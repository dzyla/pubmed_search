import os
import re
import json
import math
import time
import uuid
import gc
import ast
import sqlite3
import logging
import concurrent.futures
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

import pyarrow as pa
import pyarrow.parquet as pq

import torch
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import streamlit.components.v1 as components

from tqdm import tqdm
from sympy import N
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.manifold import MDS

from crossref.restful import Works  # Import Crossref Works for citation lookup
import doi

from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import semantic_search_faiss

from umap_pytorch import load_pumap

import google.genai as genai

# =============================================================================
# SECTION 1: Logging and General Utilities
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOGGER = logging.getLogger(__name__)

@contextmanager
def log_time(task_name: str):
    """
    Context manager that logs the start and completion of a task.
    Reports to both the Python console and the global STATUS element if it exists.
    """
    start = time.perf_counter()
    if "STATUS" in globals() and STATUS is not None:
        STATUS.info(f"Starting {task_name}...")
    LOGGER.info(f"Starting {task_name}...")
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        completion_message = f"Completed {task_name} in {duration:.2f} seconds."
        if "STATUS" in globals() and STATUS is not None:
            STATUS.info(completion_message)
        LOGGER.info(completion_message)

# =============================================================================
# SECTION 2: Database and Session Helpers
# =============================================================================

def get_current_active_users(db_path: str = "sessions_history.db", timeout: int = 300) -> int:
    """
    Inserts a record of the current user's connection into a SQLite database.
    Returns the number of active sessions.
    """
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    session_id = st.session_state.session_id
    current_time = int(time.time())
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            last_seen INTEGER
        )
    """)
    conn.commit()
    cursor.execute("""
        INSERT INTO session_history (session_id, last_seen)
        VALUES (?, ?)
    """, (session_id, current_time))
    conn.commit()
    expiration_time = current_time - timeout
    cursor.execute("""
        SELECT COUNT(*) FROM (
            SELECT session_id, MAX(last_seen) AS last_seen
            FROM session_history
            GROUP BY session_id
            HAVING last_seen >= ?
        ) AS active_sessions
    """, (expiration_time,))
    active_count = cursor.fetchone()[0]
    conn.close()
    return active_count

def get_donation_collected() -> int:
    """
    Load the donation value from the donation_data.json file.
    """
    with open('donation_data.json', 'r') as file:
        data = json.load(file)
    donation_value = data.get("donation_collected", 0)
    return int(donation_value)

# =============================================================================
# SECTION 3: API and Crossref Helpers
# =============================================================================

MODEL_SERVER_URL = "http://localhost:8000/encode"

def get_query_embedding(query, normalize=True, precision="ubinary"):
    """
    Get the query embedding from the REST API model server.
    """
    payload = {"text": query, "normalize": normalize, "precision": precision}
    response = requests.post(MODEL_SERVER_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["embedding"]
    else:
        raise Exception(
            f"Failed to get query embedding from model server: {response.status_code} {response.text}"
        )

def check_update_status():
    """
    Check whether a database update is running by calling the local update status endpoint.
    """
    try:
        response = requests.get("http://localhost:8001/status", timeout=1)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "update running":
                return True
    except Exception:
        return None
    return None

# Crossref Helper Functions
def get_citation_count(doi_str):
    """
    Retrieve the number of times a manuscript has been cited using its DOI.
    """
    works = Works()
    try:
        paper_data = works.doi(doi_str)
        return paper_data.get("is-referenced-by-count", 0)
    except Exception as e:
        LOGGER.error(f"Error fetching citation count for {doi_str}: {e}")
        return 0

def get_clean_doi(doi_str):
    """
    Clean up a DOI string to ensure it is in the correct format.
    """
    if 'arxiv.org' in doi_str:
        return doi_str
    try:
        doi_clean = doi.get_clean_doi(doi_str)
        return doi_clean
    except Exception as e:
        LOGGER.error(f"Error cleaning DOI {doi_str}: {e}")
        return doi_str

@st.cache_data(show_spinner=False)
def get_references(doi_str):
    """
    Retrieve and format the list of references cited by a manuscript.
    """
    works = Works()
    try:
        paper_data = works.doi(doi_str)
        references = paper_data.get("reference", [])
        formatted_refs = []
        for ref in references:
            if isinstance(ref, dict):
                if "unstructured" in ref:
                    formatted_refs.append(ref["unstructured"])
                else:
                    formatted_refs.append(", ".join(f"{k}: {v}" for k, v in ref.items()))
            else:
                formatted_refs.append(str(ref))
        return formatted_refs
    except Exception as e:
        LOGGER.error(f"Error fetching references for {doi_str}: {e}")
        return []

# =============================================================================
# SECTION 4: Chunked Embeddings and Semantic Search
# =============================================================================

def build_sorted_intervals_from_metadata(metadata: dict) -> list:
    """
    Build a sorted list of global intervals for binary search.
    """
    LOGGER.info("Building sorted intervals from metadata...")
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
    LOGGER.info("Completed building and sorting intervals.")
    return intervals

def find_file_for_index_in_metadata(global_idx: int, intervals: list) -> dict:
    """
    Perform binary search over cached intervals to find the file and its local index.
    """
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


def load_data_for_indices(indices_or_hits: list, metadata: dict, folder: str, use_high_quality: bool = False, top_k: int = 50):
    """
    Load data rows corresponding to global indices from Parquet files.
    
    In normal mode, the function groups indices by file and loads batches of rows.
    
    In high-quality mode, the function iterates over candidate hits (which include
    score and embedding information) in order. For each candidate, it loads only the
    corresponding row and checks that the row has a non-empty 'title' and an 'abstract'
    with length greater than a set threshold (here, 50 characters). Only accepted rows
    are added until top_k accepted entries are found.
    """
    if not use_high_quality:
        # Normal mode: indices_or_hits is a list of global indices.
        global_indices = indices_or_hits
        LOGGER.info("Loading data for given global indices from Parquet files (optimized)...")
        intervals = build_sorted_intervals_from_metadata(metadata)
        file_indices = {}
        for idx in global_indices:
            location = find_file_for_index_in_metadata(idx, intervals)
            if location is not None:
                file_name = location["source_stem"]
                file_indices.setdefault(file_name, []).append(location["local_idx"])
            else:
                LOGGER.warning(f"Global index {idx} not found in metadata intervals.")
        results = []
        for file_name, local_indices in file_indices.items():
            parquet_path = os.path.join(folder, f"{file_name}.parquet")
            LOGGER.info(f"Processing file: {parquet_path} with indices: {local_indices}")
            if not os.path.exists(parquet_path):
                LOGGER.warning(f"Parquet file not found for source {file_name} at {parquet_path}")
                continue
            parquet_file = pq.ParquetFile(parquet_path)
            num_row_groups = parquet_file.num_row_groups
            group_boundaries = []
            total_rows = 0
            for rg in range(num_row_groups):
                rg_num_rows = parquet_file.metadata.row_group(rg).num_rows
                group_boundaries.append((total_rows, total_rows + rg_num_rows - 1))
                total_rows += rg_num_rows
            starts = np.array([start for start, end in group_boundaries])
            ends = np.array([end for start, end in group_boundaries])
            rg_to_indices = {}
            for local_idx in local_indices:
                rg = np.searchsorted(ends, local_idx, side='right')
                if rg < len(group_boundaries) and local_idx >= starts[rg]:
                    relative_idx = local_idx - starts[rg]
                    rg_to_indices.setdefault(rg, []).append(relative_idx)
                else:
                    LOGGER.warning(f"Local index {local_idx} not found in any row group.")
            for rg, indices in rg_to_indices.items():
                indices.sort()
                table = parquet_file.read_row_group(rg)
                pa_indices = pa.array(indices, type=pa.int64())
                subset_table = table.take(pa_indices)
                df_subset = subset_table.to_pandas()
                df_subset['source_file'] = os.path.basename(parquet_path)
                results.append(df_subset)
        if not results:
            LOGGER.info("No data loaded. Returning empty DataFrame.")
            return pd.DataFrame()
        df_combined = pd.concat(results, ignore_index=True)
        LOGGER.info("Completed loading data for all global indices (optimized).")
        return df_combined
    else:
        # High-quality mode: indices_or_hits is a list of candidate hits.
        accepted_hits = []
        accepted_rows = []
        intervals = build_sorted_intervals_from_metadata(metadata)
        parquet_cache = {}  # cache to avoid reopening files repeatedly
        min_abstract_length = 75  # threshold for abstract length
        
        for hit in indices_or_hits:
            global_idx = hit["corpus_id"]
            location = find_file_for_index_in_metadata(global_idx, intervals)
            if location is None:
                LOGGER.warning(f"Global index {global_idx} not found in metadata intervals.")
                continue
            file_name = location["source_stem"]
            local_idx = location["local_idx"]
            parquet_path = os.path.join(folder, f"{file_name}.parquet")
            if not os.path.exists(parquet_path):
                LOGGER.warning(f"Parquet file not found for source {file_name} at {parquet_path}")
                continue
            if file_name in parquet_cache:
                parquet_file, group_boundaries, starts, ends = parquet_cache[file_name]
            else:
                parquet_file = pq.ParquetFile(parquet_path)
                num_row_groups = parquet_file.num_row_groups
                group_boundaries = []
                total_rows = 0
                for rg in range(num_row_groups):
                    rg_num_rows = parquet_file.metadata.row_group(rg).num_rows
                    group_boundaries.append((total_rows, total_rows + rg_num_rows - 1))
                    total_rows += rg_num_rows
                starts = np.array([start for start, end in group_boundaries])
                ends = np.array([end for start, end in group_boundaries])
                parquet_cache[file_name] = (parquet_file, group_boundaries, starts, ends)
            rg = np.searchsorted(ends, local_idx, side='right')
            if rg < len(group_boundaries) and local_idx >= starts[rg]:
                relative_idx = local_idx - starts[rg]
                table = parquet_file.read_row_group(rg)
                pa_indices = pa.array([relative_idx], type=pa.int64())
                subset_table = table.take(pa_indices)
                df_subset = subset_table.to_pandas()
                df_subset['source_file'] = os.path.basename(parquet_path)
                
                # Quality check: verify that 'title' exists and is non-empty and that
                # 'abstract' exists and is longer than the minimum threshold.
                title_valid = ('title' in df_subset.columns and 
                               df_subset.at[0, 'title'] is not None and 
                               str(df_subset.at[0, 'title']).strip() != "")
                abstract_valid = ('abstract' in df_subset.columns and 
                                  df_subset.at[0, 'abstract'] is not None and 
                                  len(str(df_subset.at[0, 'abstract']).strip()) > min_abstract_length)
                if title_valid and abstract_valid:
                    accepted_hits.append(hit)
                    accepted_rows.append(df_subset)
                    LOGGER.info(f"Accepted global index {global_idx} from file {file_name}.")
                else:
                    LOGGER.info(f"Rejected global index {global_idx} due to missing or insufficient title/abstract.")
            else:
                LOGGER.warning(f"Local index {local_idx} not found in any row group in file {file_name}.")
            if len(accepted_hits) >= top_k:
                break
        if not accepted_hits:
            LOGGER.info("No high quality data found. Returning empty DataFrame.")
            return [], pd.DataFrame()
        df_combined = pd.concat(accepted_rows, ignore_index=True)
        LOGGER.info("Completed loading high quality data for global indices.")
        return accepted_hits, df_combined


class ChunkedEmbeddings:
    """
    A wrapper for a list of memory-mapped embedding chunks.
    """
    def __init__(self, chunks: list, chunk_boundaries: list, total_rows: int, embedding_dim: int):
        LOGGER.info("Initializing ChunkedEmbeddings object...")
        self.chunks = chunks
        self.chunk_boundaries = chunk_boundaries
        self.total_rows = total_rows
        self.embedding_dim = embedding_dim
        LOGGER.info(f"ChunkedEmbeddings initialized with {len(chunks)} chunks, total_rows={total_rows}, embedding_dim={embedding_dim}")

    @property
    def shape(self):
        return (self.total_rows, self.embedding_dim)

    def close(self):
        """
        Close the memory-mapped files to release RAM.
        """
        for idx, chunk in enumerate(self.chunks):
            if hasattr(chunk, "_mmap"):
                try:
                    chunk._mmap.close()
                    LOGGER.info(f"Closed memmap for chunk {idx}")
                except Exception as e:
                    LOGGER.error(f"Error closing memmap for chunk {idx}: {e}")

def copy_file_into_chunk(file_info, memmap_array, offset):
    LOGGER.info(f"Copying data from file {file_info['file_stem']} into chunk at offset {offset}")
    arr = np.load(file_info["file_path"], mmap_mode="r")
    file_rows = file_info["rows"]
    memmap_array[offset: offset + file_rows] = arr[:file_rows]
    LOGGER.info(f"Copied {file_rows} rows from {file_info['file_stem']} into chunk at offset {offset}")
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
                                     chunk_size_bytes: int = 1 << 30):
    LOGGER.info("Starting creation or loading of chunked memory-mapped embeddings...")
    recreate = False
    if os.path.exists(metadata_path):
        LOGGER.info(f"Metadata file found at {metadata_path}. Loading metadata...")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        for chunk_info in metadata["chunks"]:
            chunk_file = chunk_info["chunk_file"]
            chunk_path = os.path.join(chunk_dir, chunk_file)
            if not os.path.exists(chunk_path):
                LOGGER.warning(f"Chunk file {chunk_file} not found in {chunk_dir}. Will recreate chunks.")
                recreate = True
                break
        if not recreate:
            metadata_npy_files = set()
            for chunk_info in metadata["chunks"]:
                for part in chunk_info["parts"]:
                    metadata_npy_files.add(part["source_stem"])
            current_npy_files = set(f.stem for f in Path(embeddings_directory).glob(npy_files_pattern))
            if metadata_npy_files != current_npy_files:
                LOGGER.warning("Mismatch in npy files between metadata and current directory. Will recreate chunks.")
                recreate = True
        if not recreate:
            LOGGER.info("All chunk files and npy file mappings are valid. Loading memory-mapped chunks from disk...")
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
                LOGGER.info(f"Loaded chunk from {chunk_file}: rows={actual_rows}, boundaries=({chunk_info['global_start']}, {chunk_info['global_end']})")
            total_rows = metadata["total_rows"]
            return ChunkedEmbeddings(chunks, chunk_boundaries, total_rows, metadata["embedding_dim"]), metadata
        else:
            LOGGER.info("Recreation flag triggered. Removing old metadata and chunk files.")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            for file in Path(chunk_dir).glob("chunk_*.npy"):
                os.remove(file)
    else:
        LOGGER.info(f"No metadata file found at {metadata_path}. Will create new chunked embeddings.")
    os.makedirs(chunk_dir, exist_ok=True)
    LOGGER.info(f"Chunk directory ensured at {chunk_dir}.")
    npy_files = sorted(list(Path(embeddings_directory).glob(npy_files_pattern)))
    if not npy_files:
        raise FileNotFoundError(f"No npy files found in {embeddings_directory} with pattern {npy_files_pattern}")
    LOGGER.info(f"Found {len(npy_files)} npy files matching pattern {npy_files_pattern}.")
    total_rows = 0
    embedding_dim = None
    file_infos = []
    with log_time("Processing npy files"):
        for fname in npy_files:
            LOGGER.info(f"Processing file: {fname}")
            arr = np.load(fname, mmap_mode="r")
            try:
                rows, dims = arr.shape
            except ValueError:
                try:
                    rows, _, dims = arr.shape
                except ValueError:
                    raise ValueError(f"Invalid shape for file {fname}: {arr.shape}")
            LOGGER.info(f"File {fname} has shape: ({rows}, {dims})")
            if embedding_dim is None:
                embedding_dim = dims
                LOGGER.info(f"Setting embedding dimension to {dims}")
            elif dims != embedding_dim:
                raise ValueError(f"Embedding dimension mismatch in {fname}")
            file_infos.append({"file_stem": fname.stem, "rows": rows, "file_path": str(fname)})
            total_rows += rows
            del arr
    LOGGER.info(f"Total rows: {total_rows}")
    rows_per_chunk = chunk_size_bytes // embedding_dim  # type: ignore
    LOGGER.info(f"Maximum rows per chunk calculated as: {rows_per_chunk}")
    chunk_groups = []
    current_chunk_files = []
    current_chunk_rows = 0
    LOGGER.info("Grouping npy files into chunk groups...")
    for file_info in file_infos:
        file_rows = file_info["rows"]
        if file_rows > rows_per_chunk:
            if current_chunk_files:
                chunk_groups.append(current_chunk_files)
                current_chunk_files = []
                current_chunk_rows = 0
            chunk_groups.append([file_info])
            LOGGER.info(f"File {file_info['file_stem']} is too large and placed in its own chunk.")
        else:
            if current_chunk_rows + file_rows > rows_per_chunk and current_chunk_files:
                chunk_groups.append(current_chunk_files)
                current_chunk_files = []
                current_chunk_rows = 0
            current_chunk_files.append(file_info)
            current_chunk_rows += file_rows
    if current_chunk_files:
        chunk_groups.append(current_chunk_files)
        LOGGER.info("Added final chunk group.")
    chunks = []
    chunk_boundaries = []
    chunks_metadata = []
    global_index = 0
    chunk_idx = 0
    LOGGER.info("Creating memory-mapped chunks from grouped files...")
    with log_time("Creating memmap chunks"):
        for group in chunk_groups:
            group_total_rows = sum(file_info["rows"] for file_info in group)
            chunk_file = f"chunk_{chunk_idx}.npy"
            chunk_path = os.path.join(chunk_dir, chunk_file)
            LOGGER.info(f"Creating chunk {chunk_idx}: file={chunk_file}, total_rows={group_total_rows}")
            memmap_array = np.memmap(chunk_path,
                                     dtype=np.uint8,
                                     mode="w+",
                                     shape=(group_total_rows, embedding_dim))
            chunks.append(memmap_array)
            chunk_global_start = global_index
            chunk_global_end = global_index + group_total_rows
            chunk_boundaries.append((chunk_global_start, chunk_global_end))
            parts = []
            # Precompute offsets for each file in this group sequentially.
            offsets = []
            current_offset = 0
            for file_info in group:
                offsets.append(current_offset)
                current_offset += file_info["rows"]
            # Use ThreadPoolExecutor to parallelize file copying within this chunk.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(copy_file_into_chunk, file_info, memmap_array, off)
                    for file_info, off in zip(group, offsets)
                ]
                for future in concurrent.futures.as_completed(futures):
                    part = future.result()
                    parts.append(part)
            memmap_array.flush()
            chunks_metadata.append({
                "chunk_file": chunk_file,
                "global_start": chunk_global_start,
                "global_end": chunk_global_end,
                "actual_rows": group_total_rows,
                "parts": parts,
            })
            LOGGER.info(f"Finished creating chunk {chunk_idx} with boundaries ({chunk_global_start}, {chunk_global_end})")
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
    LOGGER.info("Metadata saved successfully and chunked embeddings creation completed.")
    return ChunkedEmbeddings(chunks, chunk_boundaries, total_rows, embedding_dim), metadata

def perform_semantic_search_chunks(query_embedding: str,
                                   model,  # model not used
                                   metadata: dict,
                                   chunk_dir: str,
                                   folder: str,
                                   precision: str = "ubinary",
                                   top_k: int = 50,
                                   corpus_index=None,
                                   use_high_quality: bool = False):
    """
    Execute semantic search on chunked embeddings.
    """
    all_hits = []
    with log_time("Loading chunked embeddings for search"):
        for chunk_info in metadata["chunks"]:
            chunk_file = chunk_info["chunk_file"]
            actual_rows = chunk_info["actual_rows"]
            global_start = chunk_info["global_start"]
            chunk_path = os.path.join(chunk_dir, chunk_file)
            LOGGER.info(f"Processing chunk {chunk_file}: global indices {global_start} to {chunk_info['global_end']}, rows={actual_rows}")
            try:
                chunk_data = np.memmap(chunk_path,
                                        dtype=np.uint8,
                                        mode="r",
                                        shape=(actual_rows, metadata["embedding_dim"]))
            except Exception as e:
                LOGGER.error(f"Failed to load chunk {chunk_file}: {e}")
                continue
            chunk_results, _, _ = semantic_search_faiss(
                query_embedding,
                corpus_index=corpus_index,
                corpus_embeddings=chunk_data if corpus_index is None else None,
                corpus_precision=precision,
                top_k=top_k * 20,
                calibration_embeddings=None,
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
    if not all_hits:
        LOGGER.info("Warning: No search results found across all chunks.")
        return [], pd.DataFrame()
    all_hits.sort(key=lambda x: x["score"])
    initial_hits = len(all_hits)
    if "chunked_arxiv_embeddings" in chunk_dir:
        seen_embeddings = set()
        unique_hits = []
        for hit in all_hits:
            embedding_bytes = hit["embedding"].tobytes()
            if embedding_bytes not in seen_embeddings:
                seen_embeddings.add(embedding_bytes)
                unique_hits.append(hit)
        removed = initial_hits - len(unique_hits)
        LOGGER.info(f"Removed {removed}/{initial_hits} duplicate embeddings from arXiv results.")
        all_hits = unique_hits

    # Create a mapping from corpus_id to hit for easy lookup
    hit_dict = {hit["corpus_id"]: hit for hit in all_hits}

    # Handle high_quality mode differently
    if use_high_quality:
        LOGGER.info(f"Using high quality mode with {len(all_hits)} candidates to find top {top_k} quality results.")
        # Send all hits to load_data_for_indices which will filter for quality
        all_global_indices = [hit["corpus_id"] for hit in all_hits]
        with log_time("Loading and filtering high-quality data"):
            data = load_data_for_indices(
                all_global_indices,
                metadata,
                folder=folder,
                use_high_quality=True,
                top_k=top_k
            )

            # Filter top_hits to only include the corpus_ids that were kept after quality filtering
            if len(data) > 0 and 'corpus_id' in data.columns:
                filtered_corpus_ids = set(data['corpus_id'])
                top_hits = [hit for hit in all_hits if hit["corpus_id"] in filtered_corpus_ids]
                # Limit to top_k if needed
                top_hits = top_hits[:top_k]
            else:
                top_hits = []
    else:
        # Original behavior: just take top_k
        top_hits = all_hits[:top_k]
        LOGGER.info(f"Total merged results: {len(all_hits)}; Top {top_k} selected.")
        global_indices = [hit["corpus_id"] for hit in top_hits]
        with log_time("Loading data for selected global indices"):
            data = load_data_for_indices(global_indices, metadata, folder=folder)

    # If no results remain after filtering, return empty
    if len(data) == 0:
        LOGGER.info("No results remained after filtering/loading.")
        return [], pd.DataFrame()

    # Ensure top_hits and data are properly aligned by corpus_id
    if 'corpus_id' in data.columns:
        data_corpus_ids = set(data['corpus_id'])
        top_hits = [hit for hit in top_hits if hit["corpus_id"] in data_corpus_ids]

        # Re-sort the dataframe to match top_hits order
        hit_order = {hit["corpus_id"]: i for i, hit in enumerate(top_hits)}
        data['__sort_key'] = data['corpus_id'].map(hit_order)
        data = data.sort_values('__sort_key').drop('__sort_key', axis=1)

    # Add score and embedding to data
    score_map = {hit["corpus_id"]: hit["score"] for hit in top_hits}
    embedding_map = {hit["corpus_id"]: hit["embedding"] for hit in top_hits}

    data["score"] = data["corpus_id"].map(score_map)
    data["embedding"] = data["corpus_id"].map(embedding_map)

    LOGGER.info(f"Final search results prepared: {len(top_hits)} hits, {len(data)} data rows.")
    return top_hits, data

def perform_semantic_search_chunks(query_embedding: str,
                                   model,  # model not used
                                   metadata: dict,
                                   chunk_dir: str,
                                   folder: str,
                                   precision: str = "ubinary",
                                   top_k: int = 50,
                                   corpus_index=None,
                                   use_high_quality: bool = False): 
    """
    Execute semantic search on chunked embeddings.
    
    In normal mode, the top_k hits are selected based on score.
    In high-quality mode, the entire candidate list is passed to the data loader
    which accepts only rows with a valid title and an abstract longer than a threshold.
    """
    all_hits = []
    with log_time("Loading chunked embeddings for search"):
        for chunk_info in metadata["chunks"]:
            chunk_file = chunk_info["chunk_file"]
            actual_rows = chunk_info["actual_rows"]
            global_start = chunk_info["global_start"]
            chunk_path = os.path.join(chunk_dir, chunk_file)
            LOGGER.info(f"Processing chunk {chunk_file}: global indices {global_start} to {chunk_info['global_end']}, rows={actual_rows}")
            try:
                chunk_data = np.memmap(chunk_path,
                                        dtype=np.uint8,
                                        mode="r",
                                        shape=(actual_rows, metadata["embedding_dim"]))
            except Exception as e:
                LOGGER.error(f"Failed to load chunk {chunk_file}: {e}")
                continue
            chunk_results, _, _ = semantic_search_faiss(
                query_embedding,
                corpus_index=corpus_index,
                corpus_embeddings=chunk_data if corpus_index is None else None,
                corpus_precision=precision,
                top_k=top_k * 20,
                calibration_embeddings=None,
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

    if not all_hits:
        LOGGER.info("Warning: No search results found across all chunks.")
        return [], pd.DataFrame()

    all_hits.sort(key=lambda x: x["score"])
    initial_hits = len(all_hits)

    if "chunked_arxiv_embeddings" in chunk_dir:
        seen_embeddings = set()
        unique_hits = []
        for hit in all_hits:
            embedding_bytes = hit["embedding"].tobytes()
            if embedding_bytes not in seen_embeddings:
                seen_embeddings.add(embedding_bytes)
                unique_hits.append(hit)
        removed = initial_hits - len(unique_hits)
        LOGGER.info(f"Removed {removed}/{initial_hits} duplicate embeddings from arXiv results.")
        all_hits = unique_hits

    if not use_high_quality:
        # Normal mode: simply select the top_k hits.
        top_hits = all_hits[:top_k]
        global_indices = [hit["corpus_id"] for hit in top_hits]
        with log_time("Loading data for selected global indices"):
            data = load_data_for_indices(global_indices, metadata, folder=folder, use_high_quality=False, top_k=top_k)
    else:
        # High-quality mode: pass the full candidate list for iterative quality filtering.
        with log_time("Loading high quality data for candidate global indices"):
            top_hits, data = load_data_for_indices(all_hits, metadata, folder=folder, use_high_quality=True, top_k=top_k)

    if not data.empty:
        top_hits_df = pd.DataFrame(top_hits)
        # Create Series for the new columns using the index from top_hits_df
        score_series = pd.Series(top_hits_df["score"].values, index=top_hits_df.index)
        embedding_series = pd.Series(top_hits_df["embedding"].values, index=top_hits_df.index)
        if len(data) != len(top_hits_df):
            LOGGER.warning(f"Mismatch in lengths: data has {len(data)} rows; top_hits_df has {len(top_hits_df)} rows. Using the intersection of indices.")
            min_len = min(len(data), len(top_hits_df))
            data = data.iloc[:min_len].copy()
            score_series = score_series.iloc[:min_len]
            embedding_series = embedding_series.iloc[:min_len]
        data["score"] = score_series.values
        data["embedding"] = embedding_series.values
    LOGGER.info("Score and embedding columns added to data.")
    LOGGER.info("Final search results prepared.")
    return top_hits, data

def report_dates_from_metadata(metadata_file: str) -> dict:
    """
    Loads metadata and extracts two dates from the source_stem.
    """
    def extract_dates(source_stem: str) -> list:
        return re.findall(r"(\d{6})", source_stem)
    
    def format_date(date_str: str) -> str:
        if len(date_str) != 6:
            return date_str
        return f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:]}"
    
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    result = {"second_date_from_second_part": None, "second_date_from_last_part": None}
    
    if not metadata.get("chunks"):
        LOGGER.info("No chunks found in metadata.")
        return result
    chunk = metadata["chunks"][0]
    parts = chunk.get("parts", [])

    parts = sorted(parts, key=lambda x: x.get("source_stem", ""))
    
    if parts and isinstance(parts[-1], dict):
        source_stem_last = parts[-1].get("source_stem", "")
        dates_last = extract_dates(source_stem_last)
        if len(dates_last) >= 2:
            result["second_date_from_last_part"] = format_date(dates_last[1])
            LOGGER.info(f"Second date found in last part's source_stem: {result['second_date_from_last_part']}")
        else:
            LOGGER.warning("Not enough date strings found in the last part's source_stem.")
    else:
        LOGGER.warning("The metadata does not contain a valid last part.")
    return result

# =============================================================================
# SECTION 5: DataFrame Reformatting Functions
# =============================================================================

def reformat_biorxiv_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat a BioRxiv DataFrame so its columns match the PubMed DataFrame.
    """
    df = df.copy()
    if "server" in df.columns:
        df.rename(columns={"server": "journal"}, inplace=True)
    required_cols = ["doi", "title", "authors", "date", "version", "type", "journal", "abstract", "score", "embedding"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    df = df[required_cols]
    return df

def reformat_arxiv_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat an arXiv DataFrame so its columns match the other datasets.
    """
    df = df.copy()
    df["doi"] = df["doi"].fillna("").astype(str).str.strip()
    df.loc[df["doi"] == "", "doi"] = df["id"].apply(lambda x: f"https://arxiv.org/abs/{x}")
    df["date"] = pd.to_datetime(df["update_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["journal"] = df["journal-ref"]
    def get_version_count(versions):
        try:
            if pd.isnull(versions):
                return None
            if isinstance(versions, list):
                return len(versions)
            versions_list = ast.literal_eval(versions)
            return len(versions_list)
        except Exception:
            return None
    df["version"] = df["versions"].apply(get_version_count)
    df["type"] = "preprint"
    if "score" not in df.columns:
        df["score"] = None
    if "embedding" not in df.columns:
        df["embedding"] = None
    required_cols = ["doi", "title", "authors", "date", "version", "type", "journal", "abstract", "score", "embedding"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    df = df[required_cols]
    return df

# =============================================================================
# SECTION 6: Model Loading and Plotting Functions
# =============================================================================

def load_model() -> SentenceTransformer:
    """
    Load the SentenceTransformer model and send it to the appropriate device.
    """
    LOGGER.info("Loading SentenceTransformer model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Using device: {device}")
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    model.to(device)
    LOGGER.info("Model loaded and moved to the appropriate device.")
    return model

@st.cache_resource
def load_pumap_model_and_image(model_path: str, image_path: str) -> tuple:
    """
    Load the PuMAP model and its UMAP image.
    """
    LOGGER.info("Loading PuMAP model and UMAP image...")
    model = load_pumap(model_path)
    image = np.load(image_path)
    LOGGER.info("PuMAP model and UMAP image loaded.")
    return model, image

def plot_embedding_network_advanced(query: str, query_embedding, final_results: pd.DataFrame, metric: str = "hamming") -> go.Figure:
    """
    Create a 2D plot where distances reflect pairwise similarities.
    """
    if "embedding" not in final_results.columns:
        st.error("Embeddings not found in final results.")
        return go.Figure()
    if "citations" not in final_results.columns:
        final_results["citations"] = final_results["doi"].apply(lambda doi: get_citation_count(doi) if pd.notnull(doi) else 0)
    marker_sizes = np.log1p(final_results["citations"]) * 5
    query_marker_size = 15
    paper_embeddings = np.stack(final_results["embedding"].values)  # type: ignore
    combined_embeddings = np.vstack([query_embedding, paper_embeddings])
    distances = pairwise_distances(combined_embeddings, metric=metric)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(distances)
    labels = ["Query"] + final_results["title"].tolist()
    types = ["Query"] + ["Paper"] * len(final_results)
    marker_size_arr = np.concatenate(([query_marker_size], marker_sizes.values))  # type: ignore
    citations_arr = np.concatenate(([0], final_results["citations"].values))  # type: ignore
    min_size = 3
    factor = 5
    marker_size_arr = np.log1p(citations_arr.astype(float)) * factor + min_size
    df_plot = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "title": labels,
        "type": types,
        "marker_size": marker_size_arr,
        "citations": citations_arr,
    })
    fig = px.scatter(
        df_plot,
        x="x",
        y="y",
        color="type",
        size="marker_size",
        size_max=30,
        hover_data={"title": True, "citations": True, "x": False, "y": False},
        title="2D Relationship Plot (Marker size ~ citations)"
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color="DarkSlateGrey")),
        hovertemplate="<b>%{customdata[0]}</b><br>Citations: %{customdata[1]}<extra></extra>",
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        title_font=dict(size=18, color="black"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    for i in range(1, len(df_plot)):
        fig.add_shape(
            type="line",
            x0=df_plot.iloc[0]["x"],
            y0=df_plot.iloc[0]["y"],
            x1=df_plot.iloc[i]["x"],
            y1=df_plot.iloc[i]["y"],
            line=dict(color="gray", width=1, dash="dot"),
        )
    return fig

# =============================================================================
# SECTION 7: Combined Search Function
# =============================================================================

def combined_search(query: str, configs: list, top_show: int = 10, precision: str = "ubinary", use_high_quality: bool = False) -> pd.DataFrame:
    """
    Execute a semantic search on multiple sources using the REST API.
    """
    message_holder = STATUS
    message_holder.info("Encoding query via REST API...")
    query_embedding = get_query_embedding(query, normalize=True, precision=precision)
    query_embedding = np.array(query_embedding, dtype=np.uint8)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding[np.newaxis, :]
    st.session_state["query_embedding"] = query_embedding

    # Load chunked embeddings for each database with time reporting.
    with log_time("Loading PubMed chunked embeddings"):
        pubmed_chunk_obj, pubmed_metadata = create_chunked_embeddings_memmap(
            embeddings_directory=pubmed_config["embeddings_directory"],
            npy_files_pattern=pubmed_config["npy_files_pattern"],
            chunk_dir=pubmed_config["chunk_dir"],
            metadata_path=pubmed_config["metadata_path"],
            chunk_size_bytes=pubmed_config.get("chunk_size_bytes", 1 << 30),
        )
    with log_time("Loading BioRxiv chunked embeddings"):
        biorxiv_chunk_obj, biorxiv_metadata = create_chunked_embeddings_memmap(
            embeddings_directory=biorxiv_config["embeddings_directory"],
            npy_files_pattern=biorxiv_config["npy_files_pattern"],
            chunk_dir=biorxiv_config["chunk_dir"],
            metadata_path=biorxiv_config["metadata_path"],
            chunk_size_bytes=biorxiv_config.get("chunk_size_bytes", 1 << 30),
        )
    with log_time("Loading MedRxiv chunked embeddings"):
        medrxiv_chunk_obj, medrxiv_metadata = create_chunked_embeddings_memmap(
            embeddings_directory=medrxiv_config["embeddings_directory"],
            npy_files_pattern=medrxiv_config["npy_files_pattern"],
            chunk_dir=medrxiv_config["chunk_dir"],
            metadata_path=medrxiv_config["metadata_path"],
            chunk_size_bytes=medrxiv_config.get("chunk_size_bytes", 1 << 30),
        )
    with log_time("Loading arXiv chunked embeddings"):
        arxiv_chunk_obj, arxiv_metadata = create_chunked_embeddings_memmap(
            embeddings_directory=arxiv_config["embeddings_directory"],
            npy_files_pattern=arxiv_config["npy_files_pattern"],
            chunk_dir=arxiv_config["chunk_dir"],
            metadata_path=arxiv_config["metadata_path"],
            chunk_size_bytes=arxiv_config.get("chunk_size_bytes", 1 << 30),
        )

    top_k = top_show

    # Perform semantic searches with timing.
    with log_time("Performing PubMed semantic search"):
        _, pubmed_df = perform_semantic_search_chunks(
            query_embedding, None, pubmed_metadata,
            chunk_dir=pubmed_config["chunk_dir"],
            folder=pubmed_config["data_folder"],
            precision=precision, top_k=top_k,
            use_high_quality=use_high_quality,
        )
    pubmed_df["source"] = "PubMed"

    with log_time("Performing BioRxiv semantic search"):
        _, biorxiv_df = perform_semantic_search_chunks(
            query_embedding, None, biorxiv_metadata,
            chunk_dir=biorxiv_config["chunk_dir"],
            folder=biorxiv_config["data_folder"],
            precision=precision, top_k=top_k,
        )
    with log_time("Performing MedRxiv semantic search"):
        _, medrxiv_df = perform_semantic_search_chunks(
            query_embedding, None, medrxiv_metadata,
            chunk_dir=medrxiv_config["chunk_dir"],
            folder=medrxiv_config["data_folder"],
            precision=precision, top_k=top_k,
        )
    with log_time("Performing arXiv semantic search"):
        _, arxiv_df = perform_semantic_search_chunks(
            query_embedding, None, arxiv_metadata,
            chunk_dir=arxiv_config["chunk_dir"],
            folder=arxiv_config["data_folder"],
            precision=precision, top_k=top_k,
        )

    biorxiv_df = reformat_biorxiv_df(biorxiv_df)
    biorxiv_df["source"] = "BioRxiv"
    medrxiv_df = reformat_biorxiv_df(medrxiv_df)
    medrxiv_df["source"] = "MedRxiv"
    arxiv_df = reformat_arxiv_df(arxiv_df)
    arxiv_df["source"] = "arXiv"

    LOGGER.info("Combining search results from all sources...")
    combined_df = pd.concat([pubmed_df, biorxiv_df, medrxiv_df, arxiv_df], ignore_index=True)
    raw_min = combined_df["score"].min()
    raw_max = combined_df["score"].max()
    combined_df["quality"] = abs(combined_df["score"] - raw_max) + raw_min
    combined_df.sort_values(by="score", inplace=True)
    if combined_df["doi"].notnull().all():
        combined_df = combined_df.drop_duplicates(subset=["doi"], keep="first")
    else:
        combined_df = combined_df.drop_duplicates(subset=["title"], keep="first")
    final_results = combined_df.head(top_show).reset_index(drop=True)
    message_holder.empty()
    pubmed_chunk_obj.close()
    biorxiv_chunk_obj.close()
    medrxiv_chunk_obj.close()
    gc.collect()
    return final_results

# =============================================================================
# SECTION 8: Style, Logo, and AI Summary Functions
# =============================================================================

def define_style():
    """
    Define custom CSS styles for the Streamlit app.
    """
    st.markdown(
        """
        <style>
            .stExpander > .stButton > button {
                width: 100%;
                border: none;
                background-color: #f0f2f6;
                color: #333;
                text-align: left;
                padding: 15px;
                font-size: 18px;
                border-radius: 10px;
                margin-top: 5px;
            }
            .stExpander > .stExpanderContent {
                padding-left: 10px;
                padding-top: 10px;
            }
            a {
                color: #26557b;
                text-decoration: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def logo(db_update_date, db_size_bio, db_size_pubmed, db_size_med, db_size_arxiv):
    """
    Display the logos and database information with proper attribution and a liability disclaimer.
    """
    active_users = get_current_active_users()
    # Logo image URLs
    pubmed_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/US-NLM-PubMed-Logo.svg/720px-US-NLM-PubMed-Logo.svg.png?20080121063734"
    biorxiv_logo = "https://www.biorxiv.org/sites/default/files/biorxiv_logo_homepage.png"
    medarxiv_logo = "https://www.medrxiv.org/sites/default/files/medRxiv_homepage_logo.png"
    arxiv_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/ArXiv_logo_2022.svg/640px-ArXiv_logo_2022.svg.png"
    
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <!-- Logos with clickable links and labels -->
            <div style="display: flex; justify-content: center; align-items: center; gap: 30px;">
                <div style="text-align: center;">
                    <a href="https://pubmed.ncbi.nlm.nih.gov/" target="_blank">
                        <img src="{pubmed_logo}" alt="PubMed logo" style="max-height: 80px; object-fit: contain;">
                    </a>
                    <div style="font-size: 12px;">
                        <a href="https://pubmed.ncbi.nlm.nih.gov/" target="_blank" style="text-decoration: none; color: inherit;">PubMed</a>
                    </div>
                </div>
                <div style="text-align: center;">
                    <a href="https://www.biorxiv.org/" target="_blank">
                        <img src="{biorxiv_logo}" alt="BioRxiv logo" style="max-height: 80px; object-fit: contain;">
                    </a>
                    <div style="font-size: 12px;">
                        <a href="https://www.biorxiv.org/" target="_blank" style="text-decoration: none; color: inherit;">BioRxiv</a>
                    </div>
                </div>
                <div style="text-align: center;">
                    <a href="https://www.medrxiv.org/" target="_blank">
                        <img src="{medarxiv_logo}" alt="medRxiv logo" style="max-height: 80px; object-fit: contain;">
                    </a>
                    <div style="font-size: 12px;">
                        <a href="https://www.medrxiv.org/" target="_blank" style="text-decoration: none; color: inherit;">medRxiv</a>
                    </div>
                </div>
                <div style="text-align: center;">
                    <a href="https://arxiv.org/" target="_blank">
                        <img src="{arxiv_logo}" alt="arXiv logo" style="max-height: 80px; object-fit: contain;">
                    </a>
                    <div style="font-size: 12px;">
                        <a href="https://arxiv.org/" target="_blank" style="text-decoration: none; color: inherit;">arXiv</a>
                    </div>
                </div>
            </div>
            <!-- Title, database info, and disclaimer -->
            <div style="text-align: center; margin-top: 10px;">
                <h3 style="color: black; margin: 0; font-weight: 400;">Manuscript Semantic Search [MSS]</h3>
                <p style="font-size: 16px; color: #555; margin: 5px 0 0 0;">
                    Last database update: {db_update_date} | Active users: <b>{active_users}</b><br>
                    Database size: PubMed: {int(db_size_pubmed):,} entries / BioRxiv: {int(db_size_bio):,} / MedRxiv: {int(db_size_med):,} / arXiv: {int(db_size_arxiv):,}
                </p>
                <p style="font-size: 9px; color: #777; margin-top: 15px;">
                    Disclaimer: This website is not affiliated with, endorsed by, or sponsored by PubMed, BioRxiv, medRxiv, or arXiv. The logos shown are the property of their respective owners and are used solely for informational purposes. All liability for any legal claims or issues arising from the display or use of these logos is expressly disclaimed.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )




LLM_prompt = """You are a research assistant tasked with summarizing a collection of abstracts extracted from a database of 39 million academic entries. Your goal is to synthesize a concise, clear, and insightful summary that captures the main themes, common findings, and noteworthy trends across the abstracts. 

Instructions:
- Read the provided abstracts carefully.
- Shortly digest each abstract's content.
- Identify and list the central topics and findings.
- Highlight any recurring patterns or shared insights.
- Keep the summary succinct (1-3 short paragraphs) without using overly technical jargon.
- Do not include any external links or references.
- Format the response in markdown, using bullet points where appropriate.

Now, review the abstracts provided below and generate your summary.
"""

def summarize_abstract(abstracts, instructions, api_key, model_name="gemini-2.0-flash-lite-preview-02-05"):
    """
    Summarizes the provided abstracts using Google's Gemini Flash model via the genai library.
    """
    from google.genai import types
    if not api_key:
        return "API key not provided. Please obtain your own API key at https://aistudio.google.com/apikey"
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    formatted_text = "\n".join(f"{idx + 1}. {abstract}" for idx, abstract in enumerate(abstracts))
    prompt = f"{instructions}\n\n{formatted_text}"
    content_part = types.Part.from_text(text=prompt)
    config = types.GenerateContentConfig(temperature=1, top_p=0.95, top_k=64, max_output_tokens=8192)
    try:
        response = client.models.generate_content(model=model_name, contents=content_part, config=config)
        summary = response.text
    except Exception as e:
        summary = f"Google Flash model not available or usage limit exceeded: {e}"
    return summary

# =============================================================================
# SECTION 9: Configurations and Database Size Loader
# =============================================================================

@st.cache_data
def load_configs_and_db_sizes(config_yaml_path="config_mss.yaml"):
    """
    Load configurations from a YAML file and retrieve database sizes from JSON metadata files.
    """
    with open(config_yaml_path, "r") as f:
        config_data = yaml.safe_load(f)
    pubmed_config = config_data.get("pubmed_config", {})
    biorxiv_config = config_data.get("biorxiv_config", {})
    medrxiv_config = config_data.get("medrxiv_config", {})
    arxiv_config = config_data.get("arxiv_config", {})
    configs = [pubmed_config, biorxiv_config, medrxiv_config, arxiv_config]
    def load_db_size(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata.get("total_rows", "N/A")
        except Exception:
            return 0
    pubmed_db_size = load_db_size(pubmed_config.get("metadata_path", ""))
    biorxiv_db_size = load_db_size(biorxiv_config.get("metadata_path", ""))
    medrxiv_db_size = load_db_size(medrxiv_config.get("metadata_path", ""))
    arxiv_db_size = load_db_size(arxiv_config.get("metadata_path", ""))
    return {
        "pubmed_config": pubmed_config,
        "biorxiv_config": biorxiv_config,
        "medrxiv_config": medrxiv_config,
        "arxiv_config": arxiv_config,
        "configs": configs,
        "pubmed_db_size": pubmed_db_size,
        "biorxiv_db_size": biorxiv_db_size,
        "medrxiv_db_size": medrxiv_db_size,
        "arxiv_db_size": arxiv_db_size
    }

# =============================================================================
# SECTION 10: Streamlit App Code
# =============================================================================

st.set_page_config(
    page_title="MSS",
    page_icon="📜",
)

results = load_configs_and_db_sizes()
pubmed_config = results["pubmed_config"]
biorxiv_config = results["biorxiv_config"]
medrxiv_config = results["medrxiv_config"]
arxiv_config = results["arxiv_config"]
configs = results["configs"]
pubmed_db_size = results["pubmed_db_size"]
biorxiv_db_size = results["biorxiv_db_size"]
medrxiv_db_size = results["medrxiv_db_size"]
arxiv_db_size = results["arxiv_db_size"]

define_style()

try:
    last_biorxiv_date = report_dates_from_metadata(biorxiv_config["metadata_path"]).get("second_date_from_last_part", "N/A")
except:
    last_biorxiv_date = "N/A"

logo(last_biorxiv_date, biorxiv_db_size, pubmed_db_size, medrxiv_db_size, arxiv_db_size)

status = check_update_status()
if status:
    st.info("Database update in progress. Search might be slow...")

use_ai = False

with st.form("search_form"):
    query = st.text_input("Enter your search query:", max_chars=8192)
    col1, col2 = st.columns(2)
    with col1:
        num_to_show = st.number_input("Number of results to show:", min_value=1, max_value=50, value=10)
    with col2:
        if st.session_state.get("use_ai_checkbox"):
            ai_api_provided = st.text_input("Google AI API Key", value="", help="Obtain your own API key at https://aistudio.google.com/apikey", type="password")
        else:
            ai_api_provided = None
        use_high_quality = st.checkbox("Use high-quality search?", value=False, 
                                       help="Enable this option for more accurate results, only including entries with full abstracts and titles. This might not return all results.")
    submitted = st.form_submit_button("Search 🔍")

col1, col2 = st.columns(2)
ncol1, ncol2 = st.columns(2)
use_ai = col1.checkbox("Use AI generated summary?", key="use_ai_checkbox")

# Define a global STATUS element for user messages.
STATUS = st.empty()

if submitted and query:
    with st.spinner("Searching..."):
        search_start_time = datetime.now()
        final_results = combined_search(query, configs, top_show=num_to_show, precision="ubinary", use_high_quality=use_high_quality)
        total_time = datetime.now() - search_start_time
        st.markdown(f"<h6 style='text-align: center; color: #7882af;'>Search completed in {total_time.total_seconds():.2f} seconds</h6>", unsafe_allow_html=True)
        st.session_state["final_results"] = final_results
        st.session_state["search_query"] = query
        st.session_state["num_to_show"] = num_to_show
        st.session_state["use_ai"] = st.session_state.get("use_ai_checkbox", False)
        st.session_state["ai_api_provided"] = ai_api_provided
        st.session_state["use_high_quality"] = use_high_quality
else:
    final_results = st.session_state.get("final_results", pd.DataFrame())

if not final_results.empty:
    # Get sort option.
    sort_option = col2.segmented_control("Sort results by:", options=["Relevance", "Publication Date", "Citations"], key="sort_option")
    
    # Make a copy of the final results.
    sorted_results = st.session_state["final_results"].copy()
    
    # Retrieve the DOI list from the DataFrame.
    all_doi = sorted_results["doi"].tolist()
    
    # Check if the DOI list in session state is different.
    if st.session_state.get('doi_list') != all_doi:
        st.session_state['doi_list'] = all_doi
        st.session_state['citations'] = None
        st.session_state['clean_doi'] = None

    # Compute citations if not cached.
    if st.session_state.get('citations') is None:
        with st.spinner("Fetching citation counts..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                all_citations = list(executor.map(get_citation_count, all_doi))
            st.session_state['citations'] = all_citations
    else:
        all_citations = st.session_state.get('citations')

    # Compute clean DOI if not cached.
    if st.session_state.get('clean_doi') is None:
        with st.spinner("Cleaning DOI list..."):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                clean_doi_list = list(executor.map(get_clean_doi, all_doi))
            st.session_state['clean_doi'] = clean_doi_list
    else:
        clean_doi_list = st.session_state.get('clean_doi')

    # Update the DataFrame with computed values.
    sorted_results["citations"] = all_citations
    sorted_results["doi"] = clean_doi_list

    # Sort the DataFrame based on the selected option.
    if sort_option == "Publication Date":
        sorted_results["date_parsed"] = pd.to_datetime(sorted_results["date"], errors="coerce")
        sorted_results = sorted_results.sort_values(by="date_parsed", ascending=False).reset_index(drop=True)
        sorted_results.drop(columns=["date_parsed"], inplace=True)
    elif sort_option == "Citations":
        sorted_results = sorted_results.sort_values(by="citations", ascending=False).reset_index(drop=True)
    else:
        sorted_results = sorted_results.sort_values(by="score", ascending=True).reset_index(drop=True)

    abstracts_for_summary = []
    for idx, row in sorted_results.iterrows():
        citations = row["citations"]
        expander_title = f"{idx + 1}. {row['title']}\n\n _(Score: {row['quality']:.2f} | Date: {row['date']} | Citations: {citations})_"
        doi_link = f"https://doi.org/{row['doi']}" if "arxiv.org" not in row['doi'] else row['doi']
        with st.expander(expander_title):
            col_a, col_b, col_c = st.columns(3)
            col_a.markdown(f"**Relative Score:** {row['quality']:.2f}")
            col_b.markdown(f"**Source:** {row['source']}")
            col_c.markdown(f"**Citations:** {citations}")
            st.markdown(f"**Authors:** {row['authors']}")
            col_d, col_e = st.columns(2)
            col_d.markdown(f"**Date:** {row['date']}")
            col_e.markdown(f"**Journal/Server:** {row.get('journal', 'N/A')}")
            abstracts_for_summary.append(row["abstract"])
            st.markdown(f"**Abstract:**\n{row['abstract']}")
            st.markdown(f"**[Full Text Read]({doi_link})** 🔗")
    try:
        sorted_results["Date"] = pd.to_datetime(sorted_results["date"])
        if "citations" not in sorted_results.columns:
            sorted_results["citations"] = sorted_results["doi"].apply(lambda doi: get_citation_count(doi) if pd.notnull(doi) else 0)
        plot_data = {
            "Date": sorted_results["Date"],
            "Title": sorted_results["title"],
            "Relative Score": sorted_results["quality"],
            "DOI": sorted_results["doi"],
            "Source": sorted_results["source"],
            "citations": sorted_results["citations"],
        }
        plot_df = pd.DataFrame(plot_data)
        plot_df["marker_size"] = np.log1p(plot_df["citations"]) * 5 + 3
        fig_scatter = px.scatter(
            plot_df,
            x="Date",
            y="Relative Score",
            size="marker_size",
            hover_data={"Title": True, "DOI": True, "citations": True, "marker_size": False},
            color="Source",
            title="Publication Dates and Relative Score (Marker size ~ citations)",
        )
        fig_scatter.update_layout(legend=dict(title="Source"))
    except Exception as e:
        st.error(f"Error in plotting Score vs Year: {str(e)}")
    tabs = st.tabs(["Score vs Year", "Abstract Map"])
    with tabs[0]:
        st.plotly_chart(fig_scatter, use_container_width=True)
    with tabs[1]:
        pumap, hist_data = load_pumap_model_and_image("param_umap_model.h5", "hist2d.npz")
        try:
            hist = hist_data["hist"]
            xedges = hist_data["xedges"]
            yedges = hist_data["yedges"]
        except Exception as e:
            st.error(f"Error loading histogram data: {e}")
        x_min, x_max = float(xedges[0]), float(xedges[-1])
        y_min, y_max = float(yedges[0]), float(yedges[-1])
        heatmap = go.Heatmap(
            z=np.sqrt(hist.T),
            x=xedges,
            y=yedges,
            colorscale="balance",
            reversescale=False,
            showscale=False,
            opacity=1,
        )
        
        
        if "embedding" in final_results.columns:
            raw_embeddings = np.stack(final_results["embedding"].values)  # type: ignore
            raw_embeddings_tensor = torch.from_numpy(raw_embeddings).float()
            new_2d = pumap.transform(raw_embeddings_tensor)
            scatter_x = new_2d[:, 0]
            scatter_y = new_2d[:, 1]
        else:
            st.error("final_results does not contain an 'embedding' column with high-dimensional embeddings.")
            scatter_x, scatter_y = [], []
        
        # Define a custom color mapping for each source with blue/purple hues.
        source_color_map = {
            "PubMed": "#4e79a7",    # Blue
            "BioRxiv": "#ff6a6a",    # Purple
            "MedRxiv": "#0f4d93",    # Light Blue
            "arXiv": "#847c6f",      # Medium Purple
            "Query": "#1e1e1e",      # Dark Blue
        }
        # Create a list of colors for each point based on its 'source'
        marker_colors = [source_color_map.get(source, "#2a7aba") for source in final_results["source"]]
        
        scatter = go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode="markers",
            marker=dict(
                color=marker_colors,
                size=12,
                line=dict(width=1, color="white"),
                opacity=1,
            ),
            text=final_results.apply(lambda row: (
                f"<b>{row['title']}</b><br>"
                f"Date: {row['date']}<br>"
                f"Authors: {row['authors']}<br>"
                f"Score: {row['score']:.2f}"
            ), axis=1),
            hoverinfo="text",
            name="Found Abstracts",
        )
        fig_abstract_map = go.Figure(data=[heatmap, scatter])
        fig_abstract_map.update_layout(
            title="Abstract Map with 2D Embeddings",
            xaxis=dict(title="Component 1", range=[x_min, x_max], showgrid=False, zeroline=False),
            yaxis=dict(title="Component 2", range=[y_min, y_max], showgrid=False, zeroline=False),
            plot_bgcolor="white",
            height=600,
            width=800,
        )
        st.plotly_chart(fig_abstract_map, use_container_width=True)
    if st.session_state.get("use_ai") and abstracts_for_summary:
        with st.spinner("Generating AI summary..."):
            ai_gen_start = time.time()
            if st.session_state.get("ai_api_provided"):
                st.markdown("**AI Summary of abstracts:**")
                with log_time("AI Summary Generation"):
                    summary_text = summarize_abstract(abstracts_for_summary[:9], LLM_prompt, st.session_state["ai_api_provided"])
                st.markdown(summary_text)
                LOGGER.info(summary_text)
            total_ai_time = time.time() - ai_gen_start
            st.markdown(f"**Time to generate summary:** {total_ai_time:.2f} seconds")
    STATUS.empty()
    
st.markdown(
    """
    <div style='text-align: center;'>
        <b>[MSS] Developed by <a href="https://www.dzyla.com/" target="_blank">Dawid Zyla</a></b>
        <br>
        <a href="https://github.com/dzyla/pubmed_search" target="_blank">Source code on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("---")
cost_monthly = 50
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(
        f"""
        <div style="text-align: center; padding: 10px;">
            <h3 style="margin-bottom: 5px; font-weight: normal;">Support MSS Server</h3>
            <p style="font-size: 14px; color: #555;">
                The monthly server cost is approximately ${cost_monthly}.
                Any donation helps me keep the service running.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    donation_collected = get_donation_collected()
    target_amount = 50
    progress_fraction = donation_collected / target_amount
    st.progress(progress_fraction)
    st.markdown(
        f"""
        <p style="text-align: center; font-size: 14px; color: #555; margin-bottom: 10px;">
            ${donation_collected:.2f} raised of ${target_amount:.2f}
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="text-align: center; margin-top: 5px;">
            <a href="https://www.buymeacoffee.com/dzyla" target="_blank" 
               style="
                    background-color: #3679ae;
                    color: #ffffff;
                    padding: 10px 20px;
                    border-radius: 5px;
                    text-decoration: none;
                    font-family: Lato, sans-serif;
                    font-size: 16px;
                    box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
                    transition: background-color 0.2s ease;
               "
               onMouseOver="this.style.backgroundColor='#26557b'"
               onMouseOut="this.style.backgroundColor='#26557b'">
                Buy me a coffee
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
