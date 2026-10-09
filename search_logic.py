import os
import re
import json
import time
import numpy as np
import faiss
import logging
import gc
import threading
import pandas as pd
from pathlib import Path
import concurrent.futures
from collections import OrderedDict
from aux_index import AuxIndex
from data_handler import fetch_specific_rows, build_sorted_intervals_from_metadata, clear_parquet_cache
from identifiers import identifier_tokens, token_hash
from utils import log_time

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_SOURCE_NAMES = ("PubMed", "BioRxiv", "MedRxiv", "arXiv", "ClinicalTrials")

# ---------------------------------------------------------------------------
# Module-level caches — shared across all Streamlit sessions / reruns
# ---------------------------------------------------------------------------
# FAISS binary index cache: chunk_path -> faiss.IndexBinaryFlat
# IndexBinaryFlat.search() is thread-safe for concurrent reads.
_INDEX_CACHE: dict = {}
_INDEX_EVENTS: dict = {}    # chunk_path -> threading.Event (build-in-progress sentinel)
_INDEX_LOCK = threading.Lock()

# ChunkedSearcher cache: chunk_dir -> ChunkedSearcher
# Avoids re-running _ensure_chunks_exist() (JSON reads + glob) on every query.
_SEARCHER_CACHE: dict = {}
_SEARCHER_EVENTS: dict = {}  # chunk_dir -> threading.Event (build-in-progress sentinel)
_SEARCHER_LOCK = threading.Lock()

# Search result cache: (query, filters, sources) -> result DataFrame.
# Repeat searches (reloads, shared links, API retries) skip the 4–8 s scan.
# Any index change bumps _INDEX_GENERATION, which is part of the key.
_RESULT_CACHE: "OrderedDict" = OrderedDict()
_RESULT_CACHE_MAX = 128
_RESULT_CACHE_LOCK = threading.Lock()
_INDEX_GENERATION = 0

# How often trigger_database_updates() runs a full FS scan.
# Set via MSS_UPDATE_INTERVAL_S env var; 0 disables rate-limiting.
_UPDATE_CHECK_INTERVAL_S: float = float(os.environ.get("MSS_UPDATE_INTERVAL_S", "3600"))
_LAST_UPDATE_CHECK_TS: float = 0.0


def _get_or_build_faiss_index(chunk_path: str, actual_rows: int, embedding_dim: int):
    """
    Returns a cached faiss.IndexBinaryFlat for the given chunk file.

    Uses Event-based double-checked locking: the first thread to see a cache miss
    builds the index while all other concurrent threads wait on an Event instead of
    building redundant copies — eliminating the memory spike that would otherwise
    occur when N threads all arrive before the index is ready.
    """
    while True:
        with _INDEX_LOCK:
            if chunk_path in _INDEX_CACHE:
                return _INDEX_CACHE[chunk_path]
            if chunk_path not in _INDEX_EVENTS:
                event = threading.Event()
                _INDEX_EVENTS[chunk_path] = event
                break           # this thread is the builder
            event = _INDEX_EVENTS[chunk_path]
        # Another thread is building — wait outside the lock to avoid deadlock
        event.wait()

    try:
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
    finally:
        # Always unblock waiters, even on exception
        with _INDEX_LOCK:
            _INDEX_EVENTS.pop(chunk_path, None)
        event.set()


def _invalidate_result_cache():
    """Drops cached search results; called whenever any index content changes."""
    global _INDEX_GENERATION
    with _RESULT_CACHE_LOCK:
        _INDEX_GENERATION += 1
        _RESULT_CACHE.clear()


def _clear_index_cache_for_dir(chunk_dir: str):
    """Evicts all cached FAISS indexes whose path lives under chunk_dir."""
    _invalidate_result_cache()
    with _INDEX_LOCK:
        stale = [k for k in _INDEX_CACHE if k.startswith(chunk_dir)]
        for k in stale:
            del _INDEX_CACHE[k]
    if stale:
        LOGGER.info(f"Evicted {len(stale)} cached FAISS index(es) for {chunk_dir}")


def get_or_create_searcher(config: dict) -> "ChunkedSearcher":
    """
    Returns a cached ChunkedSearcher, creating it on first call.

    Uses the same Event-based double-checked locking as _get_or_build_faiss_index
    so concurrent callers never construct duplicate searchers for the same dir.
    """
    key = config.get("chunk_dir", str(id(config)))
    while True:
        with _SEARCHER_LOCK:
            if key in _SEARCHER_CACHE:
                return _SEARCHER_CACHE[key]
            if key not in _SEARCHER_EVENTS:
                event = threading.Event()
                _SEARCHER_EVENTS[key] = event
                break
            event = _SEARCHER_EVENTS[key]
        event.wait()

    try:
        searcher = ChunkedSearcher(config)
        with _SEARCHER_LOCK:
            _SEARCHER_CACHE[key] = searcher

        if searcher.was_updated:
            _clear_index_cache_for_dir(key)

        return searcher
    finally:
        with _SEARCHER_LOCK:
            _SEARCHER_EVENTS.pop(key, None)
        event.set()


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
# Chunk creation helpers
# ---------------------------------------------------------------------------

def _write_json_atomic(path: str, data: dict):
    """
    Writes JSON via a temp file + os.replace so readers in other processes
    (Streamlit and search_api share these files) never see a half-written file.
    """
    tmp_path = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def copy_file_into_chunk(file_info, memmap_array, offset):
    """Copies a source .npy file into the large memory-mapped chunk."""
    try:
        arr = np.load(file_info["path"], mmap_mode="r", allow_pickle=False)
        rows = file_info["rows"]
        memmap_array[offset : offset + rows] = arr[:rows]
        return {
            "source_stem": file_info["stem"],
            "chunk_local_start": offset,
            "chunk_local_end": offset + rows,
            "source_local_start": 0,
            "source_mtime": file_info.get("mtime"),
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
        self._meta_mtime = self._current_meta_mtime()
        # Pre-build interval list once; passed to every fetch_rows call to avoid
        # rebuilding from the metadata JSON on every query.
        self.intervals = build_sorted_intervals_from_metadata(self.metadata)
        self.data_problem = self._check_alignment()

        # Auxiliary indexes (identifier postings; PubMed superseded rows).
        self.source_name = config.get("source_name")
        root = config.get("aux_index_root")
        self.aux = AuxIndex(root, self.source_name) if root and self.source_name else None
        self._superseded_bits = None
        self.superseded_count = 0
        self._aux_stem_starts = None
        self._refresh_aux(force=True)

    # ------------------------------------------------------------------
    # Auxiliary indexes
    # ------------------------------------------------------------------

    def _stem_starts(self) -> np.ndarray:
        """Global row id of row 0 of each aux-index file stem (-1 if not indexed here)."""
        starts = {iv["source_stem"]: iv["global_start"] - iv["source_local_start"] for iv in self.intervals}
        return np.array([starts.get(s, -1) for s in self.aux.stems], dtype=np.int64)

    def _refresh_aux(self, force: bool = False) -> bool:
        """Loads a newer aux index version; rebuilds row mappings when it or the chunks changed."""
        if self.aux is None:
            return False
        changed = self.aux.refresh()
        if not (changed or force) or not self.aux.loaded:
            return changed
        self._aux_stem_starts = self._stem_starts()
        total = int(self.metadata.get("total_rows", 0))
        bits, count = None, 0
        if self.aux.superseded is not None and total:
            sid, row = self.aux.superseded
            start = self._aux_stem_starts[sid]
            g = start + row.astype(np.int64)
            ok = (start >= 0) & (g < total)
            mask = np.zeros(total, dtype=bool)
            mask[g[ok]] = True
            count = int(mask.sum())
            bits = np.packbits(mask)
        self._superseded_bits, self.superseded_count = bits, count
        if count:
            LOGGER.info(f"{self.source_name}: hiding {count:,} superseded record versions")
        return True

    def is_superseded(self, gids: np.ndarray) -> np.ndarray:
        gids = np.asarray(gids, dtype=np.int64)
        if self._superseded_bits is None or not len(gids):
            return np.zeros(len(gids), dtype=bool)
        return ((self._superseded_bits[gids >> 3] >> (7 - (gids & 7))) & 1).astype(bool)

    def codes_for(self, gids):
        """Stored binary codes for global row ids (from the in-memory FAISS indexes)."""
        gids = np.asarray(gids, dtype=np.int64)
        dim = int(self.metadata.get("embedding_dim", 48))
        out = np.zeros((len(gids), dim), dtype=np.uint8)
        found = np.zeros(len(gids), dtype=bool)
        for chunk in self.metadata.get("chunks", []):
            gs, n = chunk["global_start"], chunk.get("actual_rows", 0)
            m = (gids >= gs) & (gids < gs + n)
            if not m.any():
                continue
            index = _get_or_build_faiss_index(os.path.join(self.chunk_dir, chunk["chunk_file"]), n, dim)
            out[m] = np.stack([index.reconstruct(int(g - gs)) for g in gids[m]])
            found[m] = True
        return out, found

    def exact_candidates(self, token_hashes: dict, query_float=None, query_packed=None) -> list:
        """
        Documents containing the query's identifier tokens, scored like semantic
        candidates (float-query rescoring of their stored codes) plus a boost
        that grows with the share of query identifiers they contain.
        """
        if self.aux is None or not self.aux.loaded or not token_hashes or self._aux_stem_starts is None:
            return []
        per_token = {}
        for tok, h in token_hashes.items():
            p = self.aux.postings(h)
            if p is None:
                continue
            sid, row = p
            start = self._aux_stem_starts[sid]
            ok = start >= 0
            per_token[tok] = np.unique(start[ok] + row[ok].astype(np.int64))
        if not per_token:
            return []
        all_g = np.unique(np.concatenate(list(per_token.values())))
        all_g = all_g[~self.is_superseded(all_g)]
        if not len(all_g):
            return []
        hits = np.stack([np.isin(all_g, v) for v in per_token.values()], axis=1)   # docs x tokens
        order = np.argsort(-hits.sum(axis=1), kind="stable")[:EXACT_MAX_DOCS]
        all_g, hits = all_g[order], hits[order]
        codes, found = self.codes_for(all_g)
        all_g, hits, codes = all_g[found], hits[found], codes[found]
        if query_float is not None:
            scores = rescore_with_float_query(query_float, codes)
        else:
            scores = 1.0 - np.unpackbits(codes ^ np.asarray(query_packed).reshape(1, -1), axis=1).sum(1) / (codes.shape[1] * 8)
        share = hits.sum(axis=1) / len(token_hashes)
        boost = np.where(share >= 1.0, EXACT_BOOST, EXACT_BOOST * 0.5 * share)
        tokens = list(per_token)
        return [
            {"corpus_id": int(g), "score": float(sc + b),
             "matched_terms": [tokens[j] for j in np.flatnonzero(h)]}
            for g, sc, b, h in zip(all_g, scores, boost, hits)
        ]

    def _check_alignment(self):
        """
        For sources whose metadata lives in one combined parquet (bioRxiv,
        medRxiv), row i of the parquet must describe embedding row i. A row-count
        mismatch (e.g. an update that wrote the .npy but not the parquet) would
        show the wrong paper for every hit, so the source is taken out of search.
        Returns a problem description, or None.
        """
        if not self.combined_data_file or not os.path.exists(self.combined_data_file):
            return None
        stems = {p.get("source_stem") for c in self.metadata.get("chunks", []) for p in c.get("parts", [])}
        if len(stems) != 1 or os.path.exists(os.path.join(self.data_folder, f"{next(iter(stems))}.parquet")):
            return None
        try:
            import pyarrow.parquet as pq
            parquet_rows = pq.ParquetFile(self.combined_data_file).metadata.num_rows
        except Exception as e:
            return f"cannot read {self.combined_data_file}: {e}"
        index_rows = self.metadata.get("total_rows", 0)
        if parquet_rows != index_rows:
            problem = (f"{os.path.basename(self.combined_data_file)} has {parquet_rows:,} rows but the index "
                       f"has {index_rows:,} embeddings — source disabled until they match")
            LOGGER.error(f"Data alignment problem in {self.chunk_dir}: {problem}")
            return problem
        return None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _read_metadata_file(self, attempts: int = 3, delay_s: float = 0.5):
        """
        Reads the metadata JSON, retrying briefly so a file caught mid-write by
        another process is not mistaken for a corrupt one. Returns None on failure.
        """
        for attempt in range(1, attempts + 1):
            try:
                with open(self.metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                if attempt == attempts:
                    LOGGER.error(f"Failed to load metadata at {self.metadata_path}: {e}")
                    return None
                time.sleep(delay_s)

    def _load_metadata(self):
        return self._read_metadata_file() or {}

    def _current_meta_mtime(self) -> float:
        try:
            return os.path.getmtime(self.metadata_path)
        except OSError:
            return 0.0

    def _reload_metadata_if_externally_changed(self):
        """
        Detects when an external process (e.g. an update script) has written a new
        metadata file and reloads the in-memory state accordingly.
        """
        current_mtime = self._current_meta_mtime()
        if current_mtime > self._meta_mtime + 1.0:
            LOGGER.info(
                f"Metadata file externally modified ({self.metadata_path}) — reloading."
            )
            new_metadata = self._read_metadata_file()
            if not new_metadata or "chunks" not in new_metadata:
                # Keep serving the current index; retry on the next check.
                LOGGER.warning("Reloaded metadata is unreadable — keeping the current index.")
                return False
            self.metadata = new_metadata
            self._meta_mtime = current_mtime
            self.intervals = build_sorted_intervals_from_metadata(self.metadata)
            _clear_index_cache_for_dir(self.chunk_dir)
            LOGGER.info(
                f"Reloaded metadata: {len(self.metadata.get('chunks', []))} chunk(s), "
                f"{self.metadata.get('total_rows', 0):,} total rows."
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Chunk maintenance
    # ------------------------------------------------------------------

    def check_for_updates(self) -> bool:
        """
        Explicitly checks for new source files and updates chunks if necessary.
        Also picks up metadata changes written by external processes (update scripts).
        Returns True if chunks were updated or metadata was externally reloaded.
        """
        self.was_updated = False
        externally_changed = self._reload_metadata_if_externally_changed()
        self._ensure_chunks_exist()
        if self.data_problem and not (self.was_updated or externally_changed):
            # A sync may land the .npy before its parquet: re-check each pass
            # so the source comes back as soon as the files match again.
            self.data_problem = self._check_alignment()
            return self.data_problem is None
        if self.was_updated or externally_changed:
            if self.was_updated:
                self.metadata = self._load_metadata()
                self._meta_mtime = self._current_meta_mtime()
                self.intervals = build_sorted_intervals_from_metadata(self.metadata)
            self.data_problem = self._check_alignment()
            self._refresh_aux(force=True)
            return True
        if self._refresh_aux():
            _invalidate_result_cache()
            return True
        return False

    def _ensure_chunks_exist(self):
        if not os.path.exists(self.metadata_path):
            LOGGER.info(f"Metadata missing at {self.metadata_path}. Performing FULL regeneration.")
            self._full_regeneration()
            return

        meta = self._read_metadata_file()
        if meta is None:
            LOGGER.warning("Metadata unreadable after retries. Performing FULL regeneration.")
            self._full_regeneration()
            return

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

        # Detect new or modified source files
        try:
            source_files = sorted(Path(self.embeddings_dir).glob(self.npy_pattern))

            # Guard: if source embedding dim changed (e.g. model upgrade), rebuild.
            if source_files:
                try:
                    chunk_dim = meta.get("embedding_dim", 0)
                    sample_paths = source_files[:min(10, len(source_files))]
                    sampled_dims: dict = {}
                    for sp in sample_paths:
                        try:
                            s = np.load(str(sp), mmap_mode="r", allow_pickle=False)
                            d = s.shape[1] if s.ndim > 1 else 0
                            if d:
                                sampled_dims[d] = sampled_dims.get(d, 0) + 1
                        except Exception:
                            pass

                    if len(sampled_dims) > 1:
                        LOGGER.warning(
                            f"Mixed embedding dimensions found in source files: {sampled_dims}. "
                            "Files with the minority dimension will be skipped during chunking. "
                            "Re-embed them with the correct model to include all papers in search."
                        )

                    if sampled_dims:
                        source_dim = max(sampled_dims, key=sampled_dims.get)
                        if source_dim and chunk_dim and source_dim != chunk_dim:
                            LOGGER.warning(
                                f"Embedding dimension mismatch: majority of source files have "
                                f"{source_dim} bytes/vec but chunk metadata records {chunk_dim} bytes/vec. "
                                "Triggering FULL regeneration to rebuild with current model."
                            )
                            self._full_regeneration()
                            return
                except Exception as dim_err:
                    LOGGER.debug(f"Dim pre-check skipped: {dim_err}")

            # Build current state: stem -> mtime
            current_info = {p.stem: p.stat().st_mtime for p in source_files}

            # Build recorded state from metadata parts: stem -> mtime
            recorded_info: dict = {}
            for chunk in meta.get("chunks", []):
                for part in chunk.get("parts", []):
                    stem = part.get("source_stem")
                    if stem:
                        recorded_info[stem] = part.get("source_mtime")

            new_stems = set(current_info.keys()) - set(recorded_info.keys())

            modified_stems = {
                stem for stem, cur_mtime in current_info.items()
                if stem in recorded_info
                and recorded_info[stem] is not None
                and cur_mtime > recorded_info[stem] + 1.0
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

    def _batch_and_create_chunks(
        self,
        file_infos: list,
        rows_per_chunk: int,
        embedding_dim: int,
        start_chunk_idx: int,
        start_global_row: int,
    ) -> tuple:
        """
        Batches file_infos into chunks of at most rows_per_chunk rows and calls
        _create_chunk_file for each batch.

        Returns (list_of_chunk_metadata_dicts, final_global_row_count).
        """
        chunks: list = []
        batch_files: list = []
        batch_rows = 0
        chunk_idx = start_chunk_idx
        global_start = start_global_row

        for f_info in file_infos:
            if batch_rows + f_info["rows"] > rows_per_chunk and batch_files:
                chunks.append(
                    self._create_chunk_file(batch_files, batch_rows, chunk_idx, global_start, embedding_dim)
                )
                global_start += batch_rows
                chunk_idx += 1
                batch_files, batch_rows = [], 0
            batch_files.append(f_info)
            batch_rows += f_info["rows"]

        if batch_files:
            chunks.append(
                self._create_chunk_file(batch_files, batch_rows, chunk_idx, global_start, embedding_dim)
            )
            global_start += batch_rows

        return chunks, global_start

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
            old_data = np.memmap(
                chunk_path, dtype=np.uint8, mode="r", shape=(old_rows, embedding_dim)
            ).copy()

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

            # Only this chunk's index is stale; the others stay cached.
            with _INDEX_LOCK:
                _INDEX_CACHE.pop(chunk_path, None)
            _invalidate_result_cache()

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
                arr = np.load(p, mmap_mode="r", allow_pickle=False)
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

        # If the new data fits in the last chunk's remaining capacity, extend it.
        if meta.get("chunks"):
            last_rows = meta["chunks"][-1].get("actual_rows", 0)
            capacity = rows_per_chunk - last_rows
            if new_rows_count <= capacity:
                LOGGER.info(
                    f"New data ({new_rows_count:,} rows) fits in last chunk "
                    f"(free capacity={capacity:,}). Extending instead of creating new chunk."
                )
                if self._extend_last_chunk(new_file_infos, meta, embedding_dim):
                    _write_json_atomic(self.metadata_path, meta)
                    self._meta_mtime = self._current_meta_mtime()
                    LOGGER.info(f"Extension complete. Total rows: {meta['total_rows']:,}")
                    self.was_updated = True
                    return
                LOGGER.warning("Extension failed — falling back to new chunk creation.")

        # Determine the next chunk index from existing metadata
        last_chunk_idx = -1
        for c in meta.get("chunks", []):
            try:
                idx = int(c["chunk_file"].replace("chunk_", "").replace(".npy", ""))
                if idx > last_chunk_idx:
                    last_chunk_idx = idx
            except Exception:
                pass

        added_chunks, final_global = self._batch_and_create_chunks(
            new_file_infos,
            rows_per_chunk,
            embedding_dim,
            start_chunk_idx=last_chunk_idx + 1,
            start_global_row=meta.get("total_rows", 0),
        )

        meta["chunks"].extend(added_chunks)
        meta["total_rows"] = final_global

        _write_json_atomic(self.metadata_path, meta)

        self._meta_mtime = self._current_meta_mtime()
        LOGGER.info(f"Incremental update complete. Added {len(added_chunks)} chunks, {new_rows_count:,} rows.")
        self.was_updated = True

    def _delete_stale_chunk_files(self):
        """
        Removes chunk files on disk that are no longer referenced by the on-disk
        metadata. Called before writing a new metadata file so searches never see
        a stale chunk.
        """
        if not os.path.exists(self.metadata_path):
            return
        try:
            with open(self.metadata_path) as f:
                old_meta = json.load(f)
            for chunk in old_meta.get("chunks", []):
                old_path = os.path.join(self.chunk_dir, chunk["chunk_file"])
                if os.path.exists(old_path):
                    os.remove(old_path)
                    LOGGER.info(f"Removed stale chunk file: {chunk['chunk_file']}")
        except Exception as e:
            LOGGER.warning(f"Could not clean up stale chunk files: {e}")

    def _full_regeneration(self):
        LOGGER.info(f"Starting FULL chunk regeneration for {self.chunk_dir} …")

        _clear_index_cache_for_dir(self.chunk_dir)
        self._delete_stale_chunk_files()

        os.makedirs(self.chunk_dir, exist_ok=True)

        source_files = sorted(Path(self.embeddings_dir).glob(self.npy_pattern))
        if not source_files:
            LOGGER.error(f"No source NPY files found in {self.embeddings_dir}")
            return

        file_infos = []
        total_rows = 0
        embedding_dim = None
        skipped_dim_mismatch = 0

        LOGGER.info("Scanning all source files …")
        for p in source_files:
            try:
                arr = np.load(p, mmap_mode="r", allow_pickle=False)
                rows = arr.shape[0]
                dims = arr.shape[1] if arr.ndim > 1 else 1
                if embedding_dim is None:
                    embedding_dim = dims
                elif dims != embedding_dim:
                    skipped_dim_mismatch += 1
                    if skipped_dim_mismatch <= 5:
                        LOGGER.warning(
                            f"Skipping {p.name}: dim {dims} ≠ {embedding_dim}. "
                            "Re-embed this file with the correct model."
                        )
                    continue
                file_infos.append({"stem": p.stem, "path": str(p), "rows": rows, "mtime": p.stat().st_mtime})
                total_rows += rows
            except Exception as e:
                LOGGER.error(f"Error reading source file {p}: {e}")

        if skipped_dim_mismatch > 0:
            LOGGER.warning(
                f"Skipped {skipped_dim_mismatch} source file(s) due to embedding dimension mismatch. "
                f"Those papers are excluded from search until they are re-embedded with the "
                f"{embedding_dim}-byte model."
            )

        if not file_infos:
            return

        rows_per_chunk = self.chunk_size_bytes // embedding_dim
        chunks_metadata, _ = self._batch_and_create_chunks(
            file_infos, rows_per_chunk, embedding_dim,
            start_chunk_idx=0, start_global_row=0,
        )

        final_metadata = {
            "total_rows": total_rows,
            "embedding_dim": embedding_dim,
            "chunks": chunks_metadata,
        }

        _write_json_atomic(self.metadata_path, final_metadata)

        self._meta_mtime = self._current_meta_mtime()
        LOGGER.info(f"Full regeneration complete. Metadata saved to {self.metadata_path}")
        self.was_updated = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search_chunk_worker(self, args):
        """
        Searches a single chunk using a *cached* FAISS index.
        The index is built once on first call and reused across all subsequent searches.

        With query_float, candidates are re-ranked by the float query against
        their ±1 bits (see rescore_with_float_query).
        """
        chunk_info, query_packed, limit, embedding_dim, query_float = args
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
            valid = indices[0] != -1
            local_ids = indices[0][valid]
            scores = 1.0 - distances[0][valid] / d_bits
            if self._superseded_bits is not None and len(local_ids):
                keep = ~self.is_superseded(global_start + local_ids.astype(np.int64))
                local_ids, scores = local_ids[keep], scores[keep]

            if query_float is not None and len(local_ids):
                codes = np.stack([index.reconstruct(int(i)) for i in local_ids])
                scores = rescore_with_float_query(query_float, codes)

            return [
                {
                    "corpus_id": global_start + int(local_id),
                    "score": float(score),
                    "chunk_file": chunk_filename,
                }
                for local_id, score in zip(local_ids, scores)
            ]
        except Exception as e:
            LOGGER.error(f"Error searching chunk {chunk_filename}: {type(e).__name__}: {e}")
            return []

    def find_candidates_raw(self, query_packed, limit=100, query_float=None):
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
        if query_float is not None and len(query_float) != embedding_dim * 8:
            LOGGER.warning("Float query does not match index dimension — using Hamming scores.")
            query_float = None
        tasks = [(chunk_info, query_packed, limit, embedding_dim, query_float) for chunk_info in chunks_list]

        LOGGER.info(f"Searching {len(tasks)} chunk(s) …")

        all_candidates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for res in executor.map(self._search_chunk_worker, tasks):
                all_candidates.extend(res)

        return all_candidates

    def fetch_rows(self, candidates, columns=None):
        """Fetches metadata rows from Parquet for the given candidates (all wishlist columns by default)."""
        if not candidates:
            return pd.DataFrame()
        return fetch_specific_rows(
            candidates, self.metadata, self.data_folder,
            self.combined_data_file, intervals=self.intervals, columns=columns,
        )


# ---------------------------------------------------------------------------
# Result rows + dedup
# ---------------------------------------------------------------------------

RESULT_COLUMNS = [
    "doi", "title", "authors", "date", "abstract", "score", "source", "journal",
    "pmid", "pub_type", "published_doi", "version",
    # ClinicalTrials.gov
    "nct_id", "trial_status", "trial_phase", "has_results", "pmids",
    "corpus_id",   # global row id; within PubMed, a larger id = a newer record version
    "matched_terms",   # identifier tokens of the query found in this document
]

# Exact-term matching: candidates per source and score boost when a document
# contains every identifier of the query (scores are cosines mapped to [0, 1]).
EXACT_MAX_DOCS = 2000
EXACT_BOOST = 0.06
MAX_QUERY_IDENTIFIERS = 6


def _norm_doi(value) -> str:
    doi = str(value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi if len(doi) > 5 and doi not in ("none", "nan") else ""


def _title_key(title) -> str:
    """Normalized title for matching a preprint with its journal version."""
    key = re.sub(r"[^a-z0-9]+", " ", str(title or "").lower()).strip()
    return key if len(key) >= 30 else ""     # short titles ("Reply", "Editorial") are not unique


def _is_preprint(row: dict) -> bool:
    return _norm_doi(row.get("doi")).startswith("10.1101/") or row.get("source") in ("BioRxiv", "MedRxiv")


def _newer_version(row: dict, kept: dict) -> bool:
    try:
        return row.get("source") == kept.get("source") and int(row["corpus_id"]) > int(kept["corpus_id"])
    except (KeyError, TypeError, ValueError):
        return False


def _add_or_merge(rows: list, seen: dict, row: dict) -> bool:
    """
    Appends row to rows unless it duplicates one already kept (same DOI, or same
    title when it has no DOI). A preprint and its journal version count as
    duplicates — matched by the preprint's published DOI or, when the preprint
    record predates publication, by identical normalized title. The published
    version is kept in the better-ranked slot and the preprint DOI is recorded
    on it as preprint_doi. Returns True if appended.
    """
    # Trials have no DOI; their registry id identifies them.
    doi = _norm_doi(row.get("doi")) or str(row.get("nct_id") or "").strip().lower()
    published = _norm_doi(row.get("published_doi"))
    title = str(row.get("title") or "").strip().lower()
    title_key = _title_key(title)
    pmid = str(row.get("pmid") or "").strip()
    pmid_key = f"pmid:{pmid}" if pmid.isdigit() else ""
    ids = [i for i in (doi, published, pmid_key) if i]

    existing = next((seen[i] for i in ids if i in seen), None)
    if existing is None and not doi and title in seen:
        existing = seen[title]
    same_work_by_title = False
    if existing is None and title_key and f"t:{title_key}" in seen:
        candidate = seen[f"t:{title_key}"]
        if _is_preprint(rows[candidate]) != _is_preprint(row):
            existing, same_work_by_title = candidate, True

    if existing is not None:
        kept = rows[existing]
        if pmid_key and pmid_key in seen and seen[pmid_key] == existing \
                and _newer_version(row, kept):
            # PubMed update files re-issue revised records under the same PMID
            # (e.g. after a retraction). Show the newest version in this slot.
            rows[existing] = dict(row, score=kept["score"],
                                  preprint_doi=kept.get("preprint_doi", row.get("preprint_doi")))
            return False
        if _is_preprint(kept) and not _is_preprint(row) and (
                same_work_by_title or (doi and doi == _norm_doi(kept.get("published_doi")))):
            # Kept a preprint whose published version just showed up: swap them,
            # keeping the preprint's (higher) score so the slot's rank is unchanged.
            merged = dict(row, score=kept["score"], preprint_doi=kept.get("doi"))
            rows[existing] = merged
            for i in (doi, title):
                if i:
                    seen[i] = existing
        elif _is_preprint(row) and (same_work_by_title or (published and published == _norm_doi(kept.get("doi")))):
            kept.setdefault("preprint_doi", row.get("doi"))
        return False

    rows.append(row)
    for i in ids + ([title] if title else []) + ([f"t:{title_key}"] if title_key else []):
        seen.setdefault(i, len(rows) - 1)
    return True


# ---------------------------------------------------------------------------
# Search orchestrator
# ---------------------------------------------------------------------------

# Bit pattern of every byte value, MSB first (np.packbits order): (256, 8)
_BYTE_BITS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.float32)


def rescore_with_float_query(query_float, codes) -> np.ndarray:
    """
    Asymmetric score for binary documents: the float query against each
    document's bits mapped to ±1/√d, giving a cosine in [-1, 1], mapped to
    [0, 1] like the Hamming score. Offline eval (eval/ranking_eval.py, 2,000
    queries vs ~1.7M PubMed vectors, two seeds): top-1 hit rate +6–7 points,
    MRR@10 0.80 → 0.85 and 0.76 → 0.82 compared with Hamming ranking.

    Uses a per-query lookup table (byte position × byte value → summed query
    weights) instead of unpacking bits + BLAS: ~10x faster and no BLAS thread
    pool competing with FAISS's OpenMP threads.
    """
    n_bytes = codes.shape[1]
    q = np.asarray(query_float, dtype=np.float32).reshape(n_bytes, 8)
    table = np.einsum("jk,vk->jv", q, _BYTE_BITS)                 # (n_bytes, 256)
    dot = table[np.arange(n_bytes), codes].sum(axis=1)             # bits · q
    cosine = (2.0 * dot - q.sum()) / np.sqrt(n_bytes * 8)
    return (1.0 + cosine) / 2.0


def combined_search_orchestrator(
    query_packed, configs, top_k, start_date=None, end_date=None, use_high_quality=False,
    query_float=None, query_text=None,
):
    """
    Orchestrates search across all sources with live deduplication.

    Optimisations vs. original:
    - Uses module-level ChunkedSearcher cache (no repeated metadata JSON reads / globs).
    - Uses module-level FAISS index cache (no repeated memmap→FAISS copies).
    - Sources are searched in parallel using a ThreadPoolExecutor.
    - Identical repeat searches are served from an LRU result cache.
    """
    cache_key = (
        np.asarray(query_packed).tobytes(),
        None if query_float is None else np.asarray(query_float, dtype=np.float32).tobytes(),
        tuple(cfg.get("chunk_dir") for cfg in configs if cfg),
        top_k, start_date, end_date, bool(use_high_quality),
        tuple(sorted(identifier_tokens(query_text))) if query_text else (),
    )
    with _RESULT_CACHE_LOCK:
        cache_key += (_INDEX_GENERATION,)
        cached = _RESULT_CACHE.get(cache_key)
        if cached is not None:
            _RESULT_CACHE.move_to_end(cache_key)
            LOGGER.info("Search served from result cache.")
            return cached.copy()

    result_df = _run_search(query_packed, configs, top_k, start_date, end_date,
                            use_high_quality, query_float, query_text)

    with _RESULT_CACHE_LOCK:
        if cache_key[-1] == _INDEX_GENERATION and not result_df.empty:
            _RESULT_CACHE[cache_key] = result_df.copy()
            while len(_RESULT_CACHE) > _RESULT_CACHE_MAX:
                _RESULT_CACHE.popitem(last=False)
    return result_df


_DATE_COLUMNS = ["date", "update_date", "posted"]


def _filter_candidates_by_date(candidates, searchers, start_date, end_date) -> list:
    """
    Keeps candidates whose publication date is in range, reading only the date
    column from parquet (much cheaper than full rows with abstracts).
    Preserves the input (score) order.
    """
    start = pd.to_datetime(start_date) if start_date else None
    end = pd.to_datetime(end_date) if end_date else None
    by_source: dict = {}
    for c in candidates:
        by_source.setdefault(c["source"], []).append(c)

    passing: set = set()
    for source_name, subset in by_source.items():
        searcher = searchers.get(source_name)
        if not searcher:
            continue
        try:
            df = searcher.fetch_rows(subset, columns=_DATE_COLUMNS)
        except Exception as e:
            LOGGER.error(f"Date pre-filter failed for {source_name}: {e}")
            continue
        if df.empty:
            continue
        date_col = next((c for c in _DATE_COLUMNS if c in df.columns), None)
        if date_col is None:
            continue
        dates = pd.to_datetime(df[date_col], errors="coerce")
        mask = dates.notna()
        if start is not None:
            mask &= dates >= start
        if end is not None:
            mask &= dates <= end
        passing.update((source_name, cid) for cid in df.loc[mask, "corpus_id"])
    return [c for c in candidates if (c["source"], c["corpus_id"]) in passing]


def _candidate_batches(candidates, batch_size, date_filter, searchers):
    """
    Yields score-ordered batches of candidates for full-row fetching.
    With a date filter, candidates are first screened on the date column alone,
    probing ever larger slices so a rare date range takes a few rounds, not
    hundreds of 50-row rounds.
    """
    if not date_filter:
        for i in range(0, len(candidates), batch_size):
            yield candidates[i:i + batch_size]
        return

    start_date, end_date = date_filter
    pending: list = []
    pos, probe = 0, max(200, batch_size * 4)
    while pos < len(candidates) or pending:
        if len(pending) < batch_size and pos < len(candidates):
            chunk = candidates[pos:pos + probe]
            pos += probe
            probe = min(probe * 2, 20_000)
            pending += _filter_candidates_by_date(chunk, searchers, start_date, end_date)
            continue
        yield pending[:batch_size]
        pending = pending[batch_size:]


def _run_search(query_packed, configs, top_k, start_date, end_date, use_high_quality, query_float,
                query_text=None):
    """The uncached search behind combined_search_orchestrator."""
    tokens = sorted(identifier_tokens(query_text))[:MAX_QUERY_IDENTIFIERS] if query_text else []
    token_hashes = {t: token_hash(t) for t in tokens}
    sources_map = {
        name: cfg
        for name, cfg in zip(_SOURCE_NAMES, configs)
        if cfg
    }

    searchers: dict = {}
    all_global_candidates = []

    # Generous raw limit so filtering/dedup has enough candidates to work with.
    # Date-filtered searches screen candidates cheaply (date column only), so
    # they can afford a deeper pool — rare date ranges otherwise run dry.
    date_filter = (start_date, end_date) if (start_date or end_date) else None
    raw_retrieval_limit = max(5000, top_k * 100) * (4 if date_filter else 1)

    # --- 1. Retrieve raw candidates from all sources in parallel ---
    def _search_source(source_name, config):
        searcher = get_or_create_searcher(config)
        if searcher.data_problem:
            return source_name, searcher, []
        candidates = searcher.find_candidates_raw(
            query_packed, limit=raw_retrieval_limit, query_float=query_float
        )
        if token_hashes:
            by_id = {c["corpus_id"]: c for c in candidates}
            for e in searcher.exact_candidates(token_hashes, query_float, query_packed):
                c = by_id.get(e["corpus_id"])
                if c is None:
                    candidates.append(e)
                else:
                    c["score"] = max(c["score"], e["score"])
                    c["matched_terms"] = e["matched_terms"]
        for c in candidates:
            c["source"] = source_name
        return source_name, searcher, candidates

    valid_sources = list(sources_map.items())
    if not valid_sources:
        return pd.DataFrame()

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
    seen_identifiers: dict = {}   # identifier (DOI / title) -> index in final_valid_rows

    # --- 3. Batched fetch + filter + live dedup ---
    batch_size = max(50, top_k)

    for batch_candidates in _candidate_batches(all_global_candidates, batch_size, date_filter, searchers):
        if len(final_valid_rows) >= top_k:
            break

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

                # PubMed publication types (Review, Retracted Publication, …).
                # bioRxiv/medRxiv also have a "type" column, but it means something else.
                if source_name == "PubMed" and "type" in df_subset.columns:
                    df_subset["pub_type"] = df_subset["type"]

                # Preprints: DOI of the journal version, "NA" when unpublished.
                if "published" in df_subset.columns:
                    published = df_subset["published"].astype(str).str.strip()
                    df_subset["published_doi"] = published.where(published.str.startswith("10."), None)

                required_cols = RESULT_COLUMNS
                for col in required_cols:
                    if col not in df_subset.columns:
                        df_subset[col] = None

                batch_dfs.append(df_subset[required_cols])
            except Exception as e:
                LOGGER.error(f"Error fetching batch for {source_name}: {e}")

        if not batch_dfs:
            continue

        # Rows come back grouped by parquet file in completion order; restore score
        # order so the top_k cut-off and dedup keep the best-scoring rows.
        combined_batch_df = pd.concat(batch_dfs, ignore_index=True).sort_values(
            "score", ascending=False, kind="stable"
        )

        # Date filter (either bound may be given alone)
        if start_date or end_date:
            dates = pd.to_datetime(combined_batch_df["date"], errors="coerce")
            date_mask = dates.notna()
            if start_date:
                date_mask &= dates >= pd.to_datetime(start_date)
            if end_date:
                date_mask &= dates <= pd.to_datetime(end_date)
            combined_batch_df = combined_batch_df[date_mask]

        # Quality filter
        if use_high_quality:
            qual_mask = combined_batch_df["abstract"].str.len() > 75
            combined_batch_df = combined_batch_df[qual_mask.fillna(False)]

        # Live dedup (also merges a preprint with its published version)
        for row in combined_batch_df.to_dict("records"):
            if len(final_valid_rows) >= top_k:
                break
            _add_or_merge(final_valid_rows, seen_identifiers, row)

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


# ---------------------------------------------------------------------------
# "More like this": search with the stored vectors of example documents
# ---------------------------------------------------------------------------

class UnknownReference(ValueError):
    """A reference that does not point to an indexed document."""


def parse_ref(ref: str):
    source, _, cid = str(ref).partition(":")
    if source not in _SOURCE_NAMES or not cid.isdigit():
        raise UnknownReference(f"invalid reference {ref!r}; expected e.g. 'PubMed:123456'")
    return source, int(cid)


def similar_search(refs, all_configs, search_configs, top_k, start_date=None, end_date=None,
                   use_high_quality=False):
    """
    Finds documents similar to one or more example documents (refs like
    'PubMed:123'). The examples' stored ±1 codes are averaged into a float
    query, so several examples act as one "more like these" query. Returns
    (results DataFrame without the examples, list of example row dicts,
    packed query code).
    """
    by_name = dict(zip(_SOURCE_NAMES, all_configs))
    signs, seeds = [], []
    for ref in refs:
        source, cid = parse_ref(ref)
        cfg = by_name.get(source)
        if not cfg:
            raise UnknownReference(f"{source} is not indexed on this server")
        searcher = get_or_create_searcher(cfg)
        codes, found = searcher.codes_for([cid])
        if not found[0]:
            raise UnknownReference(f"{ref} does not exist")
        signs.append(np.unpackbits(codes[0]).astype(np.float32) * 2 - 1)
        row = searcher.fetch_rows([{"corpus_id": cid, "score": 1.0}])
        seed = row.iloc[0].to_dict() if not row.empty else {"corpus_id": cid}
        seed["source"] = source
        seeds.append(seed)

    q = np.mean(signs, axis=0)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise UnknownReference("the examples cancel each other out; choose fewer or closer examples")
    q_float = (q / norm).astype(np.float32)
    packed = np.packbits(q_float > 0)[np.newaxis, :]

    df = combined_search_orchestrator(packed, search_configs, top_k + 3 * len(seeds),
                                      start_date, end_date, use_high_quality, q_float)
    if df.empty:
        return df, seeds, packed
    seed_ids = {(sd["source"], int(sd["corpus_id"])) for sd in seeds}
    seed_keys = {k for sd in seeds for k in (_norm_doi(sd.get("doi")), _norm_doi(sd.get("published_doi")),
                                             f"pmid:{sd.get('pmid')}" if str(sd.get("pmid") or "").isdigit() else "",
                                             _title_key(sd.get("title"))) if k}

    def is_seed(r) -> bool:
        if (r.get("source"), int(r.get("corpus_id", -1))) in seed_ids:
            return True
        keys = {_norm_doi(r.get("doi")), _norm_doi(r.get("published_doi")), _norm_doi(r.get("preprint_doi")),
                f"pmid:{r.get('pmid')}" if str(r.get("pmid") or "").isdigit() else "", _title_key(r.get("title"))}
        return bool((keys - {""}) & seed_keys)

    keep = [not is_seed(r) for r in df.to_dict("records")]
    return df[keep].head(top_k).reset_index(drop=True), seeds, packed


def trigger_database_updates(configs, *, force: bool = False) -> bool:
    """
    Checks all configurations for new files and triggers incremental updates.
    Returns True if any database was actually updated.

    Rate-limited to at most once per MSS_UPDATE_INTERVAL_S seconds (default 1 h)
    so per-query calls are cheap when nothing has changed.
    Pass force=True to bypass the rate limit (not normally needed — the TS starts
    at 0.0 so the very first call always runs).
    """
    global _LAST_UPDATE_CHECK_TS

    if not force and _UPDATE_CHECK_INTERVAL_S > 0:
        now = time.monotonic()
        if now - _LAST_UPDATE_CHECK_TS < _UPDATE_CHECK_INTERVAL_S:
            LOGGER.debug(
                "trigger_database_updates: skipping (%.0f s since last check, interval %.0f s)",
                now - _LAST_UPDATE_CHECK_TS,
                _UPDATE_CHECK_INTERVAL_S,
            )
            return False

    _LAST_UPDATE_CHECK_TS = time.monotonic()

    updates_found = False
    valid_configs = [c for c in configs if c]

    def _check_and_update(cfg):
        key = cfg.get("chunk_dir", str(id(cfg)))
        with _SEARCHER_LOCK:
            was_cached = key in _SEARCHER_CACHE

        searcher = get_or_create_searcher(cfg)

        if was_cached:
            return searcher.check_for_updates()
        else:
            return searcher.was_updated

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(_check_and_update, cfg): cfg for cfg in valid_configs}

        for future in concurrent.futures.as_completed(futures):
            try:
                if future.result():
                    updates_found = True
                    LOGGER.info("Update detected for a source.")
            except Exception as e:
                LOGGER.error(f"Error during update check: {e}")

    if updates_found:
        clear_parquet_cache()

    return updates_found
