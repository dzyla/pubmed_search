import os
import json
import numpy as np
import faiss
import logging
import gc
import threading
import pandas as pd
from datetime import datetime
from pathlib import Path
import concurrent.futures
from data_handler import fetch_specific_rows, build_sorted_intervals_from_metadata
from utils import log_time

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level caches — shared across all Streamlit sessions / reruns
# ---------------------------------------------------------------------------
# FAISS binary index cache: chunk_path -> faiss.IndexBinaryFlat
# IndexBinaryFlat.search() is thread-safe for concurrent reads.
_INDEX_CACHE: dict = {}
_INDEX_LOCK = threading.Lock()

# ChunkedSearcher cache: chunk_dir -> ChunkedSearcher
# Avoids re-running _ensure_chunks_exist() (JSON reads + glob) on every query.
_SEARCHER_CACHE: dict = {}
_SEARCHER_LOCK = threading.Lock()


def _get_or_build_faiss_index(chunk_path: str, actual_rows: int, embedding_dim: int):
    """
    Returns a cached faiss.IndexBinaryFlat for the given chunk file.
    On first call the index is built from the memmap and stored in the module-level
    cache so every subsequent search skips the expensive .add() step.
    Thread-safe: two threads may build the same index concurrently on first access
    (harmless — last writer wins in the cache, memory cost is transient).
    """
    with _INDEX_LOCK:
        if chunk_path in _INDEX_CACHE:
            return _INDEX_CACHE[chunk_path]

    LOGGER.info(f"Building FAISS index for {Path(chunk_path).name} ({actual_rows:,} rows) …")
    chunk_data = np.memmap(chunk_path, dtype=np.uint8, mode="r", shape=(actual_rows, embedding_dim))
    d_bits = embedding_dim * 8
    index = faiss.IndexBinaryFlat(d_bits)
    index.add(chunk_data)
    del chunk_data
    gc.collect()

    with _INDEX_LOCK:
        _INDEX_CACHE[chunk_path] = index
    LOGGER.info(f"FAISS index cached for {Path(chunk_path).name}")
    return index


def _clear_index_cache_for_dir(chunk_dir: str):
    """Evicts all cached FAISS indexes whose path lives under chunk_dir."""
    with _INDEX_LOCK:
        stale = [k for k in _INDEX_CACHE if k.startswith(chunk_dir)]
        for k in stale:
            del _INDEX_CACHE[k]
    if stale:
        LOGGER.info(f"Evicted {len(stale)} cached FAISS index(es) for {chunk_dir}")


def get_or_create_searcher(config: dict) -> "ChunkedSearcher":
    """
    Returns a cached ChunkedSearcher, creating it on first call.
    Avoids the expensive _ensure_chunks_exist() (metadata JSON reads, glob scans)
    on every search request.
    """
    key = config.get("chunk_dir", str(id(config)))
    with _SEARCHER_LOCK:
        if key in _SEARCHER_CACHE:
            return _SEARCHER_CACHE[key]

    searcher = ChunkedSearcher(config)
    with _SEARCHER_LOCK:
        _SEARCHER_CACHE[key] = searcher

    if searcher.was_updated:
        # Chunks changed — any cached FAISS indexes for this dir are stale.
        _clear_index_cache_for_dir(key)

    return searcher


def warm_up_indexes(configs, background: bool = True):
    """
    Pre-builds FAISS indexes for all sources so the first real search is fast.
    Call this once after startup (e.g. after trigger_database_updates).

    Parameters
    ----------
    configs : list of dict
        Source configs (same list passed to combined_search_orchestrator).
    background : bool
        If True, runs in a daemon thread and returns immediately.
        If False, blocks until all indexes are built.
    """
    def _build():
        for config in configs:
            if not config:
                continue
            try:
                searcher = get_or_create_searcher(config)
                metadata = searcher.metadata
                embedding_dim = metadata.get("embedding_dim")
                if not embedding_dim:
                    continue
                for chunk_info in metadata.get("chunks", []):
                    chunk_path = os.path.join(searcher.chunk_dir, chunk_info["chunk_file"])
                    actual_rows = chunk_info.get("actual_rows")
                    if actual_rows and os.path.exists(chunk_path):
                        _get_or_build_faiss_index(chunk_path, actual_rows, embedding_dim)
            except Exception as exc:
                LOGGER.error(f"Warm-up failed for {config.get('chunk_dir', '?')}: {exc}")
        LOGGER.info("FAISS index warm-up complete.")

    if background:
        t = threading.Thread(target=_build, daemon=True, name="faiss-warmup")
        t.start()
    else:
        _build()


# ---------------------------------------------------------------------------
# Chunk creation helpers (unchanged logic, factored out)
# ---------------------------------------------------------------------------

def copy_file_into_chunk(file_info, memmap_array, offset):
    """Copies a source .npy file into the large memory-mapped chunk."""
    try:
        arr = np.load(file_info["path"], mmap_mode="r", allow_pickle=True)
        rows = file_info["rows"]
        memmap_array[offset : offset + rows] = arr[:rows]
        return {
            "source_stem": file_info["stem"],
            "chunk_local_start": offset,
            "chunk_local_end": offset + rows,
            "source_local_start": 0,
            "source_mtime": file_info.get("mtime"),  # recorded for change detection
        }
    except Exception as e:
        LOGGER.error(f"Error copying {file_info['stem']} into chunk: {e}")
        return None


class ChunkedSearcher:
    def __init__(self, config):
        self.config = config
        self.chunk_dir = config["chunk_dir"]
        self.metadata_path = config["metadata_path"]
        self.data_folder = config["data_folder"]
        self.embeddings_dir = config["embeddings_directory"]
        self.npy_pattern = config["npy_files_pattern"]
        self.combined_data_file = config.get("combined_data_file")
        self.chunk_size_bytes = config.get("chunk_size_bytes", 1 << 30)
        self.was_updated = False

        self._ensure_chunks_exist()
        self.metadata = self._load_metadata()
        # Pre-build interval list once; passed to every fetch_rows call to avoid
        # rebuilding from the metadata JSON on every query.
        self.intervals = build_sorted_intervals_from_metadata(self.metadata)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _load_metadata(self):
        try:
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        except Exception as e:
            LOGGER.error(f"Failed to load metadata at {self.metadata_path}: {e}")
            return {}

    # ------------------------------------------------------------------
    # Chunk maintenance
    # ------------------------------------------------------------------

    def _ensure_chunks_exist(self):
        if not os.path.exists(self.metadata_path):
            LOGGER.info(f"Metadata missing at {self.metadata_path}. Performing FULL regeneration.")
            self._full_regeneration()
            return

        try:
            with open(self.metadata_path, "r") as f:
                meta = json.load(f)

            if "embedding_dim" not in meta or "chunks" not in meta:
                LOGGER.warning("Metadata invalid. Performing FULL regeneration.")
                self._full_regeneration()
                return

            for chunk in meta.get("chunks", []):
                c_path = os.path.join(self.chunk_dir, chunk["chunk_file"])
                if not os.path.exists(c_path):
                    LOGGER.warning(f"Missing chunk file: {c_path}. Performing FULL regeneration.")
                    self._full_regeneration()
                    return
        except Exception as e:
            LOGGER.warning(f"Error reading metadata: {e}. Performing FULL regeneration.")
            self._full_regeneration()
            return

        # Detect new or modified source files
        try:
            source_files = sorted(Path(self.embeddings_dir).glob(self.npy_pattern))

            # Guard: if source embedding dim changed (e.g. model upgrade from bge-base→bge-small),
            # the chunk data is stale and must be fully rebuilt.
            if source_files:
                try:
                    sample = np.load(str(source_files[0]), mmap_mode="r", allow_pickle=True)
                    source_dim = sample.shape[1] if sample.ndim > 1 else 0
                    chunk_dim = meta.get("embedding_dim", 0)
                    if source_dim and chunk_dim and source_dim != chunk_dim:
                        LOGGER.warning(
                            f"Embedding dimension mismatch: source files have {source_dim} bytes/vec "
                            f"but chunk metadata records {chunk_dim} bytes/vec. "
                            "Triggering FULL regeneration to rebuild with current model."
                        )
                        self._full_regeneration()
                        return
                except Exception as dim_err:
                    LOGGER.debug(f"Dim pre-check skipped: {dim_err}")

            # Build current state: stem -> mtime
            current_info = {p.stem: p.stat().st_mtime for p in source_files}

            # Build recorded state from metadata parts: stem -> mtime (None if not stored yet)
            recorded_info: dict = {}
            for chunk in meta.get("chunks", []):
                for part in chunk.get("parts", []):
                    stem = part.get("source_stem")
                    if stem:
                        recorded_info[stem] = part.get("source_mtime")  # None for old metadata

            new_stems = set(current_info.keys()) - set(recorded_info.keys())

            # Files whose content may have changed (mtime recorded and now newer)
            modified_stems = {
                stem for stem, cur_mtime in current_info.items()
                if stem in recorded_info
                and recorded_info[stem] is not None          # skip legacy entries without mtime
                and cur_mtime > recorded_info[stem] + 1.0   # >1 s tolerance for FS precision
            }

            if modified_stems:
                LOGGER.warning(
                    f"Detected {len(modified_stems)} modified source file(s): "
                    f"{sorted(modified_stems)[:5]}{'…' if len(modified_stems) > 5 else ''}. "
                    "Triggering FULL regeneration."
                )
                self._full_regeneration()
            elif new_stems:
                LOGGER.info(f"Found {len(new_stems)} new file(s). Starting INCREMENTAL update …")
                new_file_paths = [p for p in source_files if p.stem in new_stems]
                try:
                    self._incremental_update(new_file_paths, meta)
                except Exception as inc_err:
                    LOGGER.error(f"Incremental update failed ({inc_err}). Falling back to FULL regeneration.")
                    self._full_regeneration()
            else:
                LOGGER.info("No new or modified source files — chunks are up to date.")
        except Exception as e:
            LOGGER.error(f"Error checking for source file changes: {e}")

    def _create_chunk_file(self, group_files, group_rows, c_idx, g_start, embedding_dim):
        chunk_filename = f"chunk_{c_idx}.npy"
        chunk_path = os.path.join(self.chunk_dir, chunk_filename)

        LOGGER.info(f"Creating {chunk_filename} ({group_rows:,} rows) …")

        memmap_arr = np.memmap(chunk_path, dtype=np.uint8, mode="w+", shape=(group_rows, embedding_dim))

        parts = []
        current_offset = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for f_info in group_files:
                futures.append(executor.submit(copy_file_into_chunk, f_info, memmap_arr, current_offset))
                current_offset += f_info["rows"]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    parts.append(res)

        memmap_arr.flush()
        del memmap_arr

        parts.sort(key=lambda x: x["chunk_local_start"])

        return {
            "chunk_file": chunk_filename,
            "global_start": g_start,
            "global_end": g_start + group_rows - 1,
            "actual_rows": group_rows,
            "parts": parts,
        }

    def _extend_last_chunk(self, new_file_infos: list, meta: dict, embedding_dim: int) -> bool:
        """
        Appends rows to the last existing chunk instead of creating a new tiny chunk.
        Evicts the stale FAISS index for that chunk so it gets rebuilt on next search.
        Returns True on success.
        """
        last_chunk = meta["chunks"][-1]
        chunk_path = os.path.join(self.chunk_dir, last_chunk["chunk_file"])
        if not os.path.exists(chunk_path):
            return False

        old_rows = last_chunk["actual_rows"]
        new_rows = sum(f["rows"] for f in new_file_infos)
        total_rows = old_rows + new_rows

        LOGGER.info(
            f"Extending {last_chunk['chunk_file']}: "
            f"{old_rows:,} → {total_rows:,} rows (+{new_rows:,})"
        )
        try:
            # Read existing data before overwriting
            old_data = np.memmap(
                chunk_path, dtype=np.uint8, mode="r", shape=(old_rows, embedding_dim)
            ).copy()

            # Write extended array
            extended = np.memmap(
                chunk_path, dtype=np.uint8, mode="w+", shape=(total_rows, embedding_dim)
            )
            extended[:old_rows] = old_data
            del old_data

            new_parts = list(last_chunk.get("parts", []))
            offset = old_rows
            for f_info in new_file_infos:
                part = copy_file_into_chunk(f_info, extended, offset)
                if part:
                    new_parts.append(part)
                offset += f_info["rows"]

            extended.flush()
            del extended

            # Evict stale FAISS cache entry for this chunk
            with _INDEX_LOCK:
                _INDEX_CACHE.pop(chunk_path, None)

            last_chunk["actual_rows"] = total_rows
            last_chunk["global_end"] = last_chunk["global_start"] + total_rows - 1
            last_chunk["parts"] = new_parts
            meta["total_rows"] = meta.get("total_rows", 0) + new_rows
            return True
        except Exception as e:
            LOGGER.error(f"Failed to extend last chunk: {e}")
            return False

    def _incremental_update(self, new_file_paths, meta):
        os.makedirs(self.chunk_dir, exist_ok=True)

        embedding_dim = meta["embedding_dim"]
        new_file_infos = []
        new_rows_count = 0

        for p in new_file_paths:
            try:
                arr = np.load(p, mmap_mode="r", allow_pickle=True)
                rows = arr.shape[0]
                dims = arr.shape[1] if arr.ndim > 1 else 1
                if dims != embedding_dim:
                    LOGGER.warning(f"Skipping {p.name}: dim mismatch {dims} vs {embedding_dim}")
                    continue
                new_file_infos.append({"stem": p.stem, "path": str(p), "rows": rows, "mtime": p.stat().st_mtime})
                new_rows_count += rows
            except Exception as e:
                LOGGER.error(f"Error reading new source file {p}: {e}")

        if not new_file_infos:
            LOGGER.info("No valid new rows to add.")
            return

        rows_per_chunk = self.chunk_size_bytes // embedding_dim

        # If the new data fits in the last chunk's remaining capacity, extend it
        # rather than creating a tiny new chunk.
        if meta.get("chunks"):
            last_rows = meta["chunks"][-1].get("actual_rows", 0)
            capacity = rows_per_chunk - last_rows
            if new_rows_count <= capacity:
                LOGGER.info(
                    f"New data ({new_rows_count:,} rows) fits in last chunk "
                    f"(free capacity={capacity:,}). Extending instead of creating new chunk."
                )
                if self._extend_last_chunk(new_file_infos, meta, embedding_dim):
                    with open(self.metadata_path, "w") as f:
                        json.dump(meta, f, indent=4)
                    LOGGER.info(f"Extension complete. Total rows: {meta['total_rows']:,}")
                    self.was_updated = True
                    return
                LOGGER.warning("Extension failed — falling back to new chunk creation.")

        current_chunks = meta.get("chunks", [])
        last_chunk_idx = -1
        for c in current_chunks:
            try:
                idx = int(c["chunk_file"].replace("chunk_", "").replace(".npy", ""))
                if idx > last_chunk_idx:
                    last_chunk_idx = idx
            except Exception:
                pass

        next_chunk_idx = last_chunk_idx + 1
        current_global_start = meta.get("total_rows", 0)

        current_batch_files = []
        current_batch_rows = 0
        added_chunks = []

        for f_info in new_file_infos:
            if current_batch_rows + f_info["rows"] > rows_per_chunk:
                if current_batch_files:
                    chunk_meta = self._create_chunk_file(
                        current_batch_files, current_batch_rows, next_chunk_idx, current_global_start, embedding_dim
                    )
                    added_chunks.append(chunk_meta)
                    current_global_start += current_batch_rows
                    next_chunk_idx += 1
                    current_batch_files = []
                    current_batch_rows = 0
            current_batch_files.append(f_info)
            current_batch_rows += f_info["rows"]

        if current_batch_files:
            chunk_meta = self._create_chunk_file(
                current_batch_files, current_batch_rows, next_chunk_idx, current_global_start, embedding_dim
            )
            added_chunks.append(chunk_meta)
            current_global_start += current_batch_rows

        meta["chunks"].extend(added_chunks)
        meta["total_rows"] = current_global_start

        with open(self.metadata_path, "w") as f:
            json.dump(meta, f, indent=4)

        LOGGER.info(f"Incremental update complete. Added {len(added_chunks)} chunks, {new_rows_count:,} rows.")
        self.was_updated = True

    def _full_regeneration(self):
        LOGGER.info(f"Starting FULL chunk regeneration for {self.chunk_dir} …")

        # Evict stale FAISS indexes for this directory before writing new chunks.
        _clear_index_cache_for_dir(self.chunk_dir)

        os.makedirs(self.chunk_dir, exist_ok=True)

        source_files = sorted(Path(self.embeddings_dir).glob(self.npy_pattern))
        if not source_files:
            LOGGER.error(f"No source NPY files found in {self.embeddings_dir}")
            return

        file_infos = []
        total_rows = 0
        embedding_dim = None

        LOGGER.info("Scanning all source files …")
        for p in source_files:
            try:
                arr = np.load(p, mmap_mode="r", allow_pickle=True)
                rows = arr.shape[0]
                dims = arr.shape[1] if arr.ndim > 1 else 1
                if embedding_dim is None:
                    embedding_dim = dims
                elif dims != embedding_dim:
                    LOGGER.warning(f"Dimension mismatch in {p.name}")
                    continue
                file_infos.append({"stem": p.stem, "path": str(p), "rows": rows, "mtime": p.stat().st_mtime})
                total_rows += rows
            except Exception as e:
                LOGGER.error(f"Error reading source file {p}: {e}")

        if not file_infos:
            return

        rows_per_chunk = self.chunk_size_bytes // (embedding_dim * 1)
        chunks_metadata = []
        current_chunk_files = []
        current_chunk_rows = 0
        chunk_idx = 0
        global_start = 0

        for f_info in file_infos:
            if current_chunk_rows + f_info["rows"] > rows_per_chunk:
                if current_chunk_files:
                    meta = self._create_chunk_file(
                        current_chunk_files, current_chunk_rows, chunk_idx, global_start, embedding_dim
                    )
                    chunks_metadata.append(meta)
                    global_start += current_chunk_rows
                    chunk_idx += 1
                    current_chunk_files = []
                    current_chunk_rows = 0
            current_chunk_files.append(f_info)
            current_chunk_rows += f_info["rows"]

        if current_chunk_files:
            meta = self._create_chunk_file(
                current_chunk_files, current_chunk_rows, chunk_idx, global_start, embedding_dim
            )
            chunks_metadata.append(meta)

        final_metadata = {
            "total_rows": total_rows,
            "embedding_dim": embedding_dim,
            "chunks": chunks_metadata,
        }

        with open(self.metadata_path, "w") as f:
            json.dump(final_metadata, f, indent=4)

        LOGGER.info(f"Full regeneration complete. Metadata saved to {self.metadata_path}")
        self.was_updated = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_chunk_worker(self, args):
        """
        Searches a single chunk using a *cached* FAISS index.
        The index is built once on first call and reused across all subsequent searches,
        eliminating the expensive memmap→FAISS copy on every query.
        """
        chunk_info, query_packed, limit, embedding_dim = args
        chunk_filename = chunk_info["chunk_file"]
        chunk_path = os.path.join(self.chunk_dir, chunk_filename)
        global_start = chunk_info["global_start"]
        actual_rows = chunk_info.get("actual_rows")

        if not os.path.exists(chunk_path) or not actual_rows:
            LOGGER.warning(f"Chunk file missing or empty: {chunk_path}")
            return []

        try:
            index = _get_or_build_faiss_index(chunk_path, actual_rows, embedding_dim)
            d_bits = embedding_dim * 8
            distances, indices = index.search(query_packed, limit)
            local_ids = indices[0]
            dists = distances[0]

            return [
                {
                    "corpus_id": global_start + int(local_id),
                    "score": 1.0 - (dist / d_bits),
                    "chunk_file": chunk_filename,
                }
                for local_id, dist in zip(local_ids, dists)
                if local_id != -1
            ]
        except Exception as e:
            LOGGER.error(f"Error searching chunk {chunk_filename}: {type(e).__name__}: {e}")
            return []

    def find_candidates_raw(self, query_packed, limit=100):
        """
        Scans all chunks in parallel and returns raw (corpus_id, score) candidates.
        FAISS indexes are cached — only the first call per chunk pays the build cost.
        """
        if not self.metadata or "chunks" not in self.metadata:
            LOGGER.warning("Metadata invalid or empty.")
            return []

        embedding_dim = self.metadata.get("embedding_dim")
        if not embedding_dim:
            LOGGER.error("Metadata missing 'embedding_dim'.")
            return []

        chunks_list = self.metadata.get("chunks", [])
        tasks = [(chunk_info, query_packed, limit, embedding_dim) for chunk_info in chunks_list]

        LOGGER.info(f"Searching {len(tasks)} chunk(s) …")

        all_candidates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for res in executor.map(self._search_chunk_worker, tasks):
                all_candidates.extend(res)

        return all_candidates

    def fetch_rows(self, candidates):
        """Fetches full metadata rows from Parquet for the given candidates."""
        if not candidates:
            return pd.DataFrame()
        return fetch_specific_rows(
            candidates, self.metadata, self.data_folder,
            self.combined_data_file, intervals=self.intervals,
        )


# ---------------------------------------------------------------------------
# Search orchestrator
# ---------------------------------------------------------------------------

def combined_search_orchestrator(
    query_packed, configs, top_k, start_date=None, end_date=None, use_high_quality=False
):
    """
    Orchestrates search across all sources with live deduplication.

    Optimisations vs. original:
    - Uses module-level ChunkedSearcher cache (no repeated metadata JSON reads / globs).
    - Uses module-level FAISS index cache (no repeated memmap→FAISS copies).
    - Sources are searched in parallel using a ThreadPoolExecutor.
    """

    sources_map = {
        "PubMed": configs[0],
        "BioRxiv": configs[1],
        "MedRxiv": configs[2],
        "arXiv": configs[3],
    }

    searchers: dict = {}
    all_global_candidates = []

    # Generous raw limit so filtering/dedup has enough candidates to work with.
    raw_retrieval_limit = max(5000, top_k * 100)

    # --- 1. Retrieve raw candidates from all sources in parallel ---
    def _search_source(source_name, config):
        searcher = get_or_create_searcher(config)
        candidates = searcher.find_candidates_raw(query_packed, limit=raw_retrieval_limit)
        for c in candidates:
            c["source"] = source_name
        return source_name, searcher, candidates

    valid_sources = [(name, cfg) for name, cfg in sources_map.items() if cfg]

    with log_time("Scanning All Sources (parallel)"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(valid_sources)) as executor:
            futures = {
                executor.submit(_search_source, name, cfg): name
                for name, cfg in valid_sources
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    source_name, searcher, candidates = future.result()
                    searchers[source_name] = searcher
                    all_global_candidates.extend(candidates)
                    LOGGER.info(f"  {source_name}: {len(candidates):,} raw candidates")
                except Exception as e:
                    LOGGER.error(f"Error searching source: {e}")

    # --- 2. Sort globally by score ---
    all_global_candidates.sort(key=lambda x: x["score"], reverse=True)

    final_valid_rows = []
    seen_identifiers: set = set()

    # --- 3. Batched fetch + filter + live dedup ---
    batch_size = max(50, top_k)

    for i in range(0, len(all_global_candidates), batch_size):
        if len(final_valid_rows) >= top_k:
            break

        batch_candidates = all_global_candidates[i : i + batch_size]

        candidates_by_source: dict = {}
        for c in batch_candidates:
            candidates_by_source.setdefault(c["source"], []).append(c)

        batch_dfs = []
        for source_name, subset in candidates_by_source.items():
            searcher = searchers.get(source_name)
            if not searcher:
                continue
            try:
                df_subset = searcher.fetch_rows(subset)
                if df_subset.empty:
                    continue

                df_subset["source"] = source_name

                col_map = {c.lower(): c for c in df_subset.columns}

                if "server" in col_map:
                    df_subset.rename(columns={col_map["server"]: "journal"}, inplace=True)
                elif "journal-ref" in col_map:
                    df_subset["journal"] = df_subset[col_map["journal-ref"]]
                elif "journal" not in col_map and "source_title" in col_map:
                    df_subset["journal"] = df_subset[col_map["source_title"]]

                if "date" not in df_subset.columns:
                    if "update_date" in df_subset.columns:
                        df_subset["date"] = df_subset["update_date"]
                    elif "posted" in df_subset.columns:
                        df_subset["date"] = df_subset["posted"]

                required_cols = ["doi", "title", "authors", "date", "abstract", "score", "source", "journal"]
                for col in required_cols:
                    if col not in df_subset.columns:
                        df_subset[col] = None

                batch_dfs.append(df_subset[required_cols])
            except Exception as e:
                LOGGER.error(f"Error fetching batch for {source_name}: {e}")

        if not batch_dfs:
            continue

        combined_batch_df = pd.concat(batch_dfs, ignore_index=True)

        # Date filter
        if start_date and end_date:
            date_mask = pd.to_datetime(combined_batch_df["date"], errors="coerce").between(
                pd.to_datetime(start_date), pd.to_datetime(end_date)
            )
            combined_batch_df = combined_batch_df[date_mask]

        # Quality filter
        if use_high_quality:
            qual_mask = combined_batch_df["abstract"].str.len() > 75
            combined_batch_df = combined_batch_df[qual_mask.fillna(False)]

        # Live dedup
        for _, row in combined_batch_df.iterrows():
            if len(final_valid_rows) >= top_k:
                break

            doi = str(row.get("doi", "")).strip()
            title = str(row.get("title", "")).strip().lower()

            if doi and len(doi) > 5 and doi in seen_identifiers:
                continue
            if (not doi or len(doi) < 5) and title in seen_identifiers:
                continue

            if doi and len(doi) > 5:
                seen_identifiers.add(doi)
            if title:
                seen_identifiers.add(title)

            final_valid_rows.append(row)

    if not final_valid_rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(final_valid_rows)

    # Label BioRxiv/MedRxiv preprints that ended up in PubMed
    if "journal" in result_df.columns:
        result_df["journal"] = result_df["journal"].astype(str).fillna("")
        bio_mask = (result_df["source"] == "PubMed") & result_df["journal"].str.contains(
            r"biorxiv", case=False, na=False
        )
        result_df.loc[bio_mask, "source"] = "BioRxiv"
        med_mask = (result_df["source"] == "PubMed") & result_df["journal"].str.contains(
            r"medrxiv", case=False, na=False
        )
        result_df.loc[med_mask, "source"] = "MedRxiv"

    if "doi" in result_df.columns:
        preprint_mask = (result_df["source"] == "PubMed") & result_df["doi"].str.contains(
            "10.1101", na=False
        )
        result_df.loc[preprint_mask, "source"] = "BioRxiv"

    return result_df.reset_index(drop=True)


def trigger_database_updates(configs):
    """
    Checks all configurations for new files and triggers incremental updates.
    Returns True if any database was actually updated.
    Populates the ChunkedSearcher cache as a side-effect.
    """
    updates_found = False
    valid_configs = [c for c in configs if c]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_or_create_searcher, cfg): cfg for cfg in valid_configs}

        for future in concurrent.futures.as_completed(futures):
            try:
                searcher = future.result()
                if searcher.was_updated:
                    updates_found = True
                    LOGGER.info("Update detected for a source.")
            except Exception as e:
                LOGGER.error(f"Error during startup update check: {e}")

    return updates_found
