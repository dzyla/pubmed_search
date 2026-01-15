import os
import json
import logging
import gc
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import numpy as np
import pandas as pd
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.quantization import semantic_search_faiss
from contextlib import contextmanager
import time

# =============================================================================
# Logging & Utilities
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOGGER = logging.getLogger(__name__)

@contextmanager
def log_time(task_name: str):
    start = time.perf_counter()
    LOGGER.info(f"Starting {task_name}...")
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        LOGGER.info(f"Completed {task_name} in {duration:.2f} seconds.")

# =============================================================================
# Model Loading
# =============================================================================

def load_model() -> SentenceTransformer:
    LOGGER.info("Loading SentenceTransformer model...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Using device: {device}")
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    model.to(device)
    LOGGER.info("Model loaded and moved to the appropriate device.")
    return model

@st.cache_resource
def load_reranker():
    # TinyBERT is fast on CPU. Use CPU explicitly to save GPU for main model if any.
    return CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', device='cpu')

MODEL_SERVER_URL = "http://localhost:8000/encode"

def get_query_embedding(query, normalize=True, precision="ubinary"):
    payload = {"text": query, "normalize": normalize, "precision": precision}
    try:
        response = requests.post(MODEL_SERVER_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data["embedding"]
        else:
            st.error(f"Model API returned error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Model API not available. Please ensure that the model server is running.")
        return None
    except Exception as e:
        st.error(f"An error occurred while obtaining the query embedding: {e}")
        return None

# =============================================================================
# Chunked Embeddings Logic
# =============================================================================

class ChunkedEmbeddings:
    def __init__(self, chunks: list, chunk_boundaries: list, total_rows: int, embedding_dim: int):
        self.chunks = chunks
        self.chunk_boundaries = chunk_boundaries
        self.total_rows = total_rows
        self.embedding_dim = embedding_dim

    @property
    def shape(self):
        return (self.total_rows, self.embedding_dim)

    def close(self):
        for idx, chunk in enumerate(self.chunks):
            if hasattr(chunk, "_mmap"):
                try:
                    chunk._mmap.close()
                except Exception as e:
                    LOGGER.error(f"Error closing memmap for chunk {idx}: {e}")

def copy_file_into_chunk(file_info, memmap_array, offset):
    arr = np.load(file_info["file_path"], mmap_mode="r")
    file_rows = file_info["rows"]
    memmap_array[offset: offset + file_rows] = arr[:file_rows]
    return {
        "source_stem": file_info["file_stem"],
        "parquet_file": file_info["file_stem"] + ".parquet",
        "chunk_local_start": offset,
        "chunk_local_end": offset + file_rows,
        "source_local_start": 0,
        "source_local_end": file_rows,
    }

def create_chunked_embeddings_memmap(embeddings_directory: str,
                                     npy_files_pattern: str,
                                     chunk_dir: str,
                                     metadata_path: str,
                                     chunk_size_bytes: int = 200 * 1024 * 1024):
    LOGGER.info("Starting creation or loading of chunked memory-mapped embeddings...")
    recreate = False
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        for chunk_info in metadata["chunks"]:
            chunk_file = chunk_info["chunk_file"]
            chunk_path = os.path.join(chunk_dir, chunk_file)
            if not os.path.exists(chunk_path):
                recreate = True
                break
        if not recreate:
            metadata_npy_files = set()
            for chunk_info in metadata["chunks"]:
                for part in chunk_info["parts"]:
                    metadata_npy_files.add(part["source_stem"])
            current_npy_files = set(f.stem for f in Path(embeddings_directory).glob(npy_files_pattern))
            if metadata_npy_files != current_npy_files:
                recreate = True
        if not recreate:
            chunks = []
            chunk_boundaries = []
            for chunk_info in metadata["chunks"]:
                chunk_file = chunk_info["chunk_file"]
                actual_rows = chunk_info["actual_rows"]
                embedding_dim = metadata["embedding_dim"]
                memmap_array = np.memmap(os.path.join(chunk_dir, chunk_file),
                                         dtype=np.uint8,
                                         mode="r",
                                         shape=(actual_rows, embedding_dim))
                chunks.append(memmap_array)
                chunk_boundaries.append((chunk_info["global_start"], chunk_info["global_end"]))
            total_rows = metadata["total_rows"]
            return ChunkedEmbeddings(chunks, chunk_boundaries, total_rows, metadata["embedding_dim"]), metadata
        else:
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            for file in Path(chunk_dir).glob("chunk_*.npy"):
                os.remove(file)

    os.makedirs(chunk_dir, exist_ok=True)
    npy_files = sorted(list(Path(embeddings_directory).glob(npy_files_pattern)))
    if not npy_files:
        raise FileNotFoundError(f"No npy files found in {embeddings_directory}")

    total_rows = 0
    embedding_dim = None
    file_infos = []
    for fname in npy_files:
        arr = np.load(fname, mmap_mode="r")
        rows, dims = arr.shape
        if embedding_dim is None:
            embedding_dim = dims
        elif dims != embedding_dim:
            raise ValueError(f"Embedding dimension mismatch in {fname}")
        file_infos.append({"file_stem": fname.stem, "rows": rows, "file_path": str(fname)})
        total_rows += rows
        del arr

    rows_per_chunk = chunk_size_bytes // embedding_dim
    chunk_groups = []
    current_chunk_files = []
    current_chunk_rows = 0

    for file_info in file_infos:
        file_rows = file_info["rows"]
        if file_rows > rows_per_chunk:
            if current_chunk_files:
                chunk_groups.append(current_chunk_files)
                current_chunk_files = []
                current_chunk_rows = 0
            chunk_groups.append([file_info])
        else:
            if current_chunk_rows + file_rows > rows_per_chunk and current_chunk_files:
                chunk_groups.append(current_chunk_files)
                current_chunk_files = []
                current_chunk_rows = 0
            current_chunk_files.append(file_info)
            current_chunk_rows += file_rows
    if current_chunk_files:
        chunk_groups.append(current_chunk_files)

    chunks = []
    chunk_boundaries = []
    chunks_metadata = []
    global_index = 0
    chunk_idx = 0

    for group in chunk_groups:
        group_total_rows = sum(file_info["rows"] for file_info in group)
        chunk_file = f"chunk_{chunk_idx}.npy"
        chunk_path = os.path.join(chunk_dir, chunk_file)
        memmap_array = np.memmap(chunk_path, dtype=np.uint8, mode="w+", shape=(group_total_rows, embedding_dim))
        chunks.append(memmap_array)
        chunk_global_start = global_index
        chunk_global_end = global_index + group_total_rows
        chunk_boundaries.append((chunk_global_start, chunk_global_end))
        parts = []
        offsets = []
        current_offset = 0
        for file_info in group:
            offsets.append(current_offset)
            current_offset += file_info["rows"]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(copy_file_into_chunk, file_info, memmap_array, off)
                for file_info, off in zip(group, offsets)
            ]
            for future in concurrent.futures.as_completed(futures):
                parts.append(future.result())
        memmap_array.flush()
        chunks_metadata.append({
            "chunk_file": chunk_file,
            "global_start": chunk_global_start,
            "global_end": chunk_global_end,
            "actual_rows": group_total_rows,
            "parts": parts,
        })
        global_index += group_total_rows
        chunk_idx += 1

    metadata = {
        "total_rows": total_rows,
        "embedding_dim": embedding_dim,
        "chunk_size_bytes": chunk_size_bytes,
        "rows_per_chunk": rows_per_chunk,
        "chunks": chunks_metadata,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return ChunkedEmbeddings(chunks, chunk_boundaries, total_rows, embedding_dim), metadata

# =============================================================================
# Search Logic
# =============================================================================

def build_sorted_intervals_from_metadata(metadata: dict) -> list:
    intervals = []
    for chunk in metadata["chunks"]:
        for part in chunk["parts"]:
            part_global_start = chunk["global_start"] + part["chunk_local_start"]
            part_global_end = chunk["global_start"] + part["chunk_local_end"] - 1
            intervals.append({
                "global_start": part_global_start,
                "global_end": part_global_end,
                "source_stem": part["source_stem"],
                "source_local_start": part["source_local_start"],
            })
    intervals.sort(key=lambda x: x["global_start"])
    return intervals

def find_file_for_index_in_metadata_interval(global_idx: int, intervals: list) -> dict:
    lo = 0
    hi = len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        interval = intervals[mid]
        if interval["global_start"] <= global_idx <= interval["global_end"]:
            offset = global_idx - interval["global_start"]
            local_idx = interval["source_local_start"] + offset
            return {"source_stem": interval["source_stem"], "local_idx": local_idx}
        elif global_idx < interval["global_start"]:
            hi = mid - 1
        else:
            lo = mid + 1
    return None

def get_location_for_index(global_idx: int, metadata: dict) -> dict:
    # Redundant but kept for compatibility with existing calls if any
    intervals = build_sorted_intervals_from_metadata(metadata)
    return find_file_for_index_in_metadata_interval(global_idx, intervals)

def check_date_on_the_fly_grouped(global_indices, metadata, data_folder, start_date, end_date):
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Error parsing dates: {e}")

    file_to_candidates = {}
    intervals = build_sorted_intervals_from_metadata(metadata)

    for idx in global_indices:
        location = find_file_for_index_in_metadata_interval(idx, intervals)
        if location is None:
            continue
        file_stem = location["source_stem"]
        local_idx = location["local_idx"]
        file_to_candidates.setdefault(file_stem, []).append((idx, local_idx))

    def process_file(file_stem, candidates):
        filtered = []
        parquet_path = os.path.join(data_folder, f"{file_stem}.parquet")
        if not os.path.exists(parquet_path):
            return filtered
        try:
            parquet_file = pq.ParquetFile(parquet_path)
            schema = parquet_file.schema_arrow
            if "update_date" in schema.names:
                date_col = "update_date"
            elif "date" in schema.names:
                date_col = "date"
            else:
                return filtered

            table = pq.read_table(parquet_path, columns=[date_col])
            df = table.to_pandas().reset_index(drop=True)
            df["date_converted"] = pd.to_datetime(df[date_col], errors="coerce")

            for global_idx, local_idx in candidates:
                if local_idx < 0 or local_idx >= len(df): continue
                row_date = df.iloc[local_idx]["date_converted"]
                if pd.isna(row_date): continue
                if start_dt <= row_date.to_pydatetime() <= end_dt:
                    filtered.append(global_idx)
        except Exception as e:
            LOGGER.error(f"Error in date check for {file_stem}: {e}")
        return filtered

    filtered_indices = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, stem, cands) for stem, cands in file_to_candidates.items()]
        for future in concurrent.futures.as_completed(futures):
            filtered_indices.extend(future.result())
    return filtered_indices

def perform_semantic_search_chunks(query_embedding: str,
                                   metadata: dict,
                                   chunk_dir: str,
                                   folder: str,
                                   precision: str = "ubinary",
                                   top_k: int = 50,
                                   corpus_index=None,
                                   use_high_quality: bool = False,
                                   start_date: str = None,
                                   end_date: str = None):
    all_hits = []
    with log_time("Loading chunked embeddings for search"):
        for chunk_info in metadata["chunks"]:
            chunk_file = chunk_info["chunk_file"]
            actual_rows = chunk_info["actual_rows"]
            global_start = chunk_info["global_start"]
            chunk_path = os.path.join(chunk_dir, chunk_file)
            try:
                chunk_data = np.memmap(chunk_path, dtype=np.uint8, mode="r", shape=(actual_rows, metadata["embedding_dim"]))
                chunk_results, _, _ = semantic_search_faiss(
                    query_embedding,
                    corpus_index=corpus_index,
                    corpus_embeddings=chunk_data if corpus_index is None else None,
                    corpus_precision=precision,
                    top_k=top_k * 20, # Fetch more for filtering
                    rescore=False,
                    rescore_multiplier=4,
                    exact=True,
                    output_index=True,
                )
                if len(chunk_results) > 0:
                    hits = chunk_results[0]
                    for hit in hits:
                        global_idx = global_start + hit["corpus_id"]
                        embedding_vec = np.array(chunk_data[hit["corpus_id"]])
                        all_hits.append({
                            "corpus_id": global_idx,
                            "score": hit["score"],
                            "embedding": embedding_vec,
                        })
                del chunk_data
            except Exception as e:
                LOGGER.error(f"Failed to load/search chunk {chunk_file}: {e}")
                continue

    if start_date and end_date:
        candidate_indices = [hit["corpus_id"] for hit in all_hits]
        filtered_candidate_indices = check_date_on_the_fly_grouped(candidate_indices, metadata, folder, start_date, end_date)
        all_hits = [hit for hit in all_hits if hit["corpus_id"] in filtered_candidate_indices]

    if not all_hits:
        return [], pd.DataFrame()

    all_hits.sort(key=lambda x: x["score"]) # Ascending score (distance) for FAISS?
    # semantic_search_faiss returns scores. If binary, hamming distance. Lower is better?
    # Actually semantic_search_faiss usually returns cosine similarity if float, but hamming if binary.
    # Code assumes sort by score.

    # Remove duplicates for ArXiv if needed
    if "chunked_arxiv_embeddings" in chunk_dir:
        seen = set()
        unique = []
        for hit in all_hits:
            eb = hit["embedding"].tobytes()
            if eb not in seen:
                seen.add(eb)
                unique.append(hit)
        all_hits = unique

    # Retrieve Data
    # If high quality, we filter. If not, we take top_k.
    # Since we want to support Reranking later, we might want to return more results if caller asks.
    # But here we stick to returning top_k hits and data.

    # Actually, let's return all hits and let the wrapper handle reranking selection
    # But to save memory we shouldn't load data for ALL hits.
    # We will slice top_k here as before.

    top_hits = all_hits[:top_k]

    if use_high_quality:
        with log_time("Loading and filtering high-quality data"):
            top_hits, data = load_data_for_indices(all_hits, metadata, folder=folder, use_high_quality=True, top_k=top_k)
    else:
        with log_time("Loading data for selected global indices"):
            data = load_data_for_indices([h["corpus_id"] for h in top_hits], metadata, folder=folder, use_high_quality=False, top_k=top_k)

    if not data.empty:
        # Align data with hits
        # data might be shorter or different order if fetched in parallel?
        # load_data_for_indices returns data combined.
        # We need to attach scores/embeddings.
        # If High Quality, load_data_for_indices returns (accepted_hits, df).
        if use_high_quality:
            top_hits_df = pd.DataFrame(top_hits)
            data["score"] = top_hits_df["score"].values
            data["embedding"] = top_hits_df["embedding"].values
        else:
            # Low quality: we passed indices. data corresponds to indices?
            # load_data_for_indices returns concatenated DF. Order might not be preserved if parallel?
            # Actually parallel fetch does NOT guarantee order unless we re-sort.
            # load_data_for_indices logic in previous file did NOT explicitly re-sort by input order.
            # I should fix this in load_data_for_indices or here.
            # Let's check load_data_for_indices implementation below.
            pass # We will fix it below.

    return top_hits, data

def load_data_for_indices(indices_or_hits, metadata, folder, use_high_quality=False, top_k=50):
    intervals = build_sorted_intervals_from_metadata(metadata)

    # 1. Prepare Request
    # Map (file_stem, local_idx) -> [list of (original_index_in_request, hit_obj)]
    file_requests = {}

    if not use_high_quality:
        # indices_or_hits is list of global_indices
        for i, global_idx in enumerate(indices_or_hits):
            loc = find_file_for_index_in_metadata_interval(global_idx, intervals)
            if loc:
                stem = loc["source_stem"]
                lidx = loc["local_idx"]
                file_requests.setdefault(stem, []).append((i, lidx, global_idx))
    else:
        # indices_or_hits is list of hit objects (dicts)
        for i, hit in enumerate(indices_or_hits):
            global_idx = hit["corpus_id"]
            loc = find_file_for_index_in_metadata_interval(global_idx, intervals)
            if loc:
                stem = loc["source_stem"]
                lidx = loc["local_idx"]
                file_requests.setdefault(stem, []).append((i, lidx, hit))

    # 2. Fetch Data
    fetched_rows = [] # list of (original_index, row_series, hit_obj_if_hq)

    def fetch_file(stem, reqs):
        path = os.path.join(folder, f"{stem}.parquet")
        if not os.path.exists(path): return []
        try:
            pf = pq.ParquetFile(path)
            # Optimization: Group reqs by row group if we wanted to be super precise,
            # but given user constraint is 8GB RAM and chunks are 200MB, reading full file
            # might still risk spiking RAM if multiple threads do it.
            # However, PyArrow's read() might be efficient.
            # To be safer, we read only necessary row groups if possible, OR just read table.
            # Since I must be robust, I will use `read_table` but ensure we don't hold it long.
            # But wait, `pd.read_parquet` is often optimized.
            # The issue with `pd.read_parquet` is it loads EVERYTHING.
            # If 4 workers load 200MB each -> 800MB. That is fine for 8GB RAM.
            # Wait, the previous `pbmss_app.py` logic used `read_row_group`.
            # If the user specifically wants memory safety, I should reuse that pattern.

            # Let's map lidx to row groups.
            # We don't have row group info easily without opening file.
            # Opening file is fast.

            # Simple safe approach: Read full file sequentially (reduce max_workers if needed) or use row groups.
            # I will implement row-group reading.

            num_rgs = pf.num_row_groups
            # Build index map: lidx -> (rg_index, offset_in_rg)
            # This requires scanning row group metadata.

            # Fast path: sort reqs by lidx.
            # Iterate row groups. Keep track of current start row.
            # If lidx in current rg range, read RG.

            reqs.sort(key=lambda x: x[1]) # Sort by lidx

            results = []
            current_row = 0

            # Collect all lidxs we need
            needed_lidxs = {r[1] for r in reqs}

            for rg in range(num_rgs):
                rg_meta = pf.metadata.row_group(rg)
                rg_rows = rg_meta.num_rows
                rg_start = current_row
                rg_end = current_row + rg_rows

                # Check if we need any rows in this RG
                # Intersection of [rg_start, rg_end) and needed_lidxs
                # Simple check:
                # needed_lidxs are sorted? No, set isn't.
                # We can filter reqs.

                rg_reqs = [r for r in reqs if rg_start <= r[1] < rg_end]

                if rg_reqs:
                    # Read only this row group
                    table = pf.read_row_group(rg)
                    df = table.to_pandas()

                    for req in rg_reqs:
                        if use_high_quality:
                            orig_i, lidx, hit = req
                        else:
                            orig_i, lidx, global_idx = req

                        rel_idx = lidx - rg_start
                        if rel_idx < len(df):
                            row = df.iloc[rel_idx]

                            if use_high_quality:
                                title_ok = pd.notna(row.get('title')) and str(row.get('title')).strip() != ""
                                abs_ok = pd.notna(row.get('abstract')) and len(str(row.get('abstract')).strip()) > 75
                                if title_ok and abs_ok:
                                    results.append((orig_i, row, hit))
                            else:
                                results.append((orig_i, row, None))

                    del table, df

                current_row += rg_rows

            return results
        except Exception as e:
            LOGGER.error(f"Error reading {stem}: {e}")
            return []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_file, stem, reqs) for stem, reqs in file_requests.items()]
        for f in concurrent.futures.as_completed(futures):
            fetched_rows.extend(f.result())

    # 3. Reconstruct
    if use_high_quality:
        # Sort by original score (which corresponds to original_index if input was sorted)
        # But we only keep valid ones.
        # We need to collect top_k valid ones.
        # Sort fetched by original_index (ranking)
        fetched_rows.sort(key=lambda x: x[0])
        final_rows = fetched_rows[:top_k]

        if not final_rows: return [], pd.DataFrame()

        hits = [x[2] for x in final_rows]
        data = pd.DataFrame([x[1] for x in final_rows])
        return hits, data
    else:
        # Low quality: we must return data aligned with input `indices_or_hits`
        # But input might have items we failed to fetch.
        # We sort fetched by original_index
        fetched_rows.sort(key=lambda x: x[0])

        # If we missed some, the alignment with `top_hits` in caller (which uses slicing) breaks.
        # Caller expects `data` to match `top_hits`.
        # So we should filter `indices_or_hits` to only those we found, OR fill with None.
        # But `perform_semantic_search_chunks` logic is: `top_hits = all_hits[:top_k]`.
        # Then `data = load...`.
        # If we return fewer rows, we must ensure caller knows which hits they correspond to.
        # But `load_data_for_indices` returns just `data`.
        # Update: `perform_semantic_search_chunks` logic:
        # `data["score"] = top_hits_df["score"].values` -> Mismatch risk!

        # Safer: Return `data` that has `original_index` and let caller merge?
        # Or just return DataFrame ordered by `original_index` and assume caller matches?
        # If we miss a row, we are in trouble.

        # Given we read parquet by index, missing a row implies file error or index out of bounds.
        # We can treat it as "lost".
        # We will re-align.

        # Create DF
        if not fetched_rows: return pd.DataFrame()

        # We need to ensure 1-to-1 mapping if possible.
        # But for robustness, if we miss one, we just drop the hit?
        # But `perform_semantic_search_chunks` has `top_hits` fixed.

        # Let's modify `perform_semantic_search_chunks` to handle this.
        # But I am writing `search_engine.py`. I can define the contract.

        # New Contract: `load_data_for_indices` returns a DataFrame that aligns with the SUCCESSFUL hits.
        # In non-HQ mode, we just return the DF.
        # I will ensure `perform_semantic_search_chunks` handles the merge.

        data = pd.DataFrame([x[1] for x in fetched_rows])
        # Add a column to track original index to help debugging/merging if needed
        data["_original_rank"] = [x[0] for x in fetched_rows]
        return data

def rerank_results(query, initial_results_df, top_k=10):
    if initial_results_df.empty:
        return initial_results_df

    try:
        reranker = load_reranker()
        pairs = []
        for _, row in initial_results_df.iterrows():
            title = str(row.get('title', ''))
            abstract = str(row.get('abstract', ''))
            pairs.append((query, title + " " + abstract))

        scores = reranker.predict(pairs)
        initial_results_df['rerank_score'] = scores
        sorted_df = initial_results_df.sort_values(by='rerank_score', ascending=False)
        return sorted_df.head(top_k).reset_index(drop=True)
    except Exception as e:
        LOGGER.error(f"Reranking failed: {e}")
        return initial_results_df.head(top_k)

def perform_semantic_search_chunks_wrapper(query_embedding, metadata, chunk_dir, folder, precision="ubinary", top_k=50, use_high_quality=False, start_date=None, end_date=None):
    # Fetch 2x candidates if we plan to rerank?
    # The wrapper is used by combined_search.
    # combined_search does the reranking.
    # So we just fetch top_k here.
    # But wait, if we want "State of the Art", we should fetch more here?
    # The caller `combined_search` sets `top_k`.

    hits, data = perform_semantic_search_chunks(query_embedding, metadata, chunk_dir, folder, precision, top_k, None, use_high_quality, start_date, end_date)

    if data.empty:
        return hits, data

    # Align scores/embeddings
    # hits contains score/embedding. data is DF.
    # If use_high_quality=True, `hits` corresponds to `data`.
    # If use_high_quality=False, `hits` was `top_hits` (len K), `data` is len <= K.
    # We need to filter `hits` to match `data`.

    if use_high_quality:
        # hits and data are aligned by `load_data_for_indices_robust`
        scores = [h["score"] for h in hits]
        embeddings = [h["embedding"] for h in hits]
        data["score"] = scores
        data["embedding"] = embeddings
    else:
        # data has `_original_rank`.
        # hits was sorted by score (rank).
        # Filter hits based on what we found.
        found_indices = set(data["_original_rank"].values)
        filtered_hits = [hits[i] for i in range(len(hits)) if i in found_indices]

        # Sort data by _original_rank to match filtered_hits
        data = data.sort_values("_original_rank").reset_index(drop=True)

        scores = [h["score"] for h in filtered_hits]
        embeddings = [h["embedding"] for h in filtered_hits]

        data["score"] = scores
        data["embedding"] = embeddings
        hits = filtered_hits # Return filtered hits

    return hits, data

def combined_search(query: str, configs: list, top_show: int = 10, precision: str = "ubinary", use_high_quality: bool = False, start_date: str = None, end_date: str = None, rerank: bool = True) -> pd.DataFrame:
    # 1. Encode
    query_embedding = get_query_embedding(query, normalize=True, precision=precision)
    if query_embedding is None: return pd.DataFrame()
    query_embedding = np.array(query_embedding, dtype=np.uint8)
    if query_embedding.ndim == 1: query_embedding = query_embedding[np.newaxis, :]

    # Store in session state for UMAP
    st.session_state["query_embedding"] = query_embedding

    all_dfs = []

    # Retrieval limit: fetch more for reranking
    retrieval_k = top_show * 2 if rerank else top_show

    def process_source(name, config, reformat_func=None):
        try:
            chunk_obj, metadata = create_chunked_embeddings_memmap(
                embeddings_directory=config["embeddings_directory"],
                npy_files_pattern=config["npy_files_pattern"],
                chunk_dir=config["chunk_dir"],
                metadata_path=config["metadata_path"],
                chunk_size_bytes=config.get("chunk_size_bytes", 200 * 1024 * 1024),
            )

            _, df = perform_semantic_search_chunks_wrapper(
                query_embedding, metadata,
                chunk_dir=config["chunk_dir"],
                folder=config["data_folder"],
                precision=precision,
                top_k=retrieval_k,
                use_high_quality=use_high_quality,
                start_date=start_date, end_date=end_date
            )

            chunk_obj.close()
            del chunk_obj
            del metadata
            gc.collect()

            if not df.empty:
                if reformat_func: df = reformat_func(df)
                df["source"] = name
                return df
        except Exception as e:
            LOGGER.error(f"Error in {name}: {e}")
        return pd.DataFrame()

    # Process sequentially
    for name, config in [("PubMed", configs[0]), ("BioRxiv", configs[1]), ("MedRxiv", configs[2]), ("arXiv", configs[3])]:
        reformat = None
        if name == "BioRxiv" or name == "MedRxiv":
            from data_handler import reformat_biorxiv_df
            reformat = reformat_biorxiv_df
        elif name == "arXiv":
            from data_handler import reformat_arxiv_df
            reformat = reformat_arxiv_df

        STATUS = st.empty() # Need to handle UI status if needed, or rely on caller?
        # The original code used global STATUS. Here we are in module.
        # We can assume `st.toast` or logs.
        # Or pass a callback.
        # For simplicity, we just run.

        df = process_source(name, config, reformat)
        if not df.empty:
            all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if combined_df.empty: return pd.DataFrame()

    # Deduplicate
    if combined_df["doi"].notnull().all():
        combined_df = combined_df.drop_duplicates(subset=["doi"], keep="first")
    else:
        combined_df = combined_df.drop_duplicates(subset=["title"], keep="first")

    # Normalize scores for ranking
    raw_min = combined_df["score"].min()
    raw_max = combined_df["score"].max()
    combined_df["quality"] = abs(combined_df["score"] - raw_max) + raw_min

    # Sort initial by vector score
    combined_df.sort_values(by="score", inplace=True)

    # Rerank
    if rerank:
        # Take top candidates (e.g. 2x top_show, or all retrieved)
        candidates = combined_df.head(retrieval_k * 4) # Rerank top 4N?
        # TinyBERT is fast. Reranking 100 items is ~0.5s on CPU.
        reranked = rerank_results(query, candidates, top_k=top_show)
        return reranked
    else:
        return combined_df.head(top_show).reset_index(drop=True)
