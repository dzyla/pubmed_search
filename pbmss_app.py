import os
from pdb import pm
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
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
    with open('donation_data.json', 'r') as file:
        data = json.load(file)
    donation_value = data.get("donation_collected", 0)
    return int(donation_value)

# =============================================================================
# SECTION 3: API and Crossref Helpers
# =============================================================================

# --- OPTIMIZATION: Global Session for faster networking ---
def create_session():
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

GLOBAL_SESSION = create_session()

# Update get_citation_count to use session and cache
@st.cache_data(ttl=3600, show_spinner=False)
def get_citation_count_cached(doi_str):
    try:
        works = Works()
        paper_data = works.doi(doi_str)
        return paper_data.get("is-referenced-by-count", 0)
    except:
        return 0

# Optimized Link Checker
@st.cache_data(ttl=3600, show_spinner=False)
def get_link_info_cached(row_dict):
    # Wrapper to return a dictionary of link data we need
    # This prevents sending the whole dataframe row which breaks caching
    source = row_dict.get("source", "").lower()
    pmid = row_dict.get("version") # Assuming version holds PMID for PubMed
    doi_val = row_dict.get("doi")

    full_text = None
    if source == "pubmed" and pmid:
        url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
        try:
            r = GLOBAL_SESSION.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if "records" in data and "pmcid" in data["records"][0]:
                    full_text = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{data['records'][0]['pmcid']}/"
        except:
            pass
    elif "arxiv" in str(doi_val):
        full_text = doi_val

    return full_text

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

def check_update_status():
    try:
        response = requests.get("http://localhost:8001/status", timeout=1)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "update running":
                return True
    except Exception:
        return None
    return None

def get_citation_count(doi_str):
    works = Works()
    try:
        paper_data = works.doi(doi_str)
        return paper_data.get("is-referenced-by-count", 0)
    except Exception as e:
        LOGGER.error(f"Error fetching citation count for {doi_str}: {e}")
        return 0

def get_clean_doi(doi_str):
    if 'arxiv.org' in doi_str:
        return doi_str
    try:
        doi_clean = doi.get_clean_doi(doi_str)
        return doi_clean
    except Exception as e:
        LOGGER.error(f"Error cleaning DOI {doi_str}: {e}")
        return doi_str

# Function to check if an article is available as a full document.
def get_full_text_link(row):
    """
    Check if the article is available on PubMed Central as a full document.
    Uses the PubMed Central API to convert PubMed ID to PubMed Central ID.
    If available, returns the PubMed Central link; otherwise falls back to the DOI link.
    """
    source = row.get("source", "None").lower()
    if source.lower() == "pubmed":
        pmid = row.get("version")
        doi = row.get("doi")
        if pmid:
            url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    records = data.get("records", [])
                    if records:
                        record = records[0]
                        pmcid = record.get("pmcid")
                        if pmcid:
                            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
            except Exception as e:
                st.error(f"Error contacting PubMed Central API: {e}")
        
        return None
    else:
        doi = row.get("doi")
        if doi:
            if "arxiv.org" in doi:
                return doi
            else:
                return f"https://doi.org/{doi}"
        return None


# Precalculate full text links in parallel using ThreadPoolExecutor.
def precalculate_full_text_links_parallel(df):
    """
    Process each row of the DataFrame in parallel to calculate the full text link.
    Stores the result in a new column 'full_text_link'.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(get_full_text_link, [row for _, row in df.iterrows()]))
    df["full_text_link"] = results
    return df

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
        #LOGGER.info("No chunks found in metadata.")
        return result
    chunk = metadata["chunks"][0]
    parts = chunk.get("parts", [])

    parts = sorted(parts, key=lambda x: x.get("source_stem", ""))
    
    if parts and isinstance(parts[-1], dict):
        source_stem_last = parts[-1].get("source_stem", "")
        dates_last = extract_dates(source_stem_last)
        if len(dates_last) >= 2:
            result["second_date_from_last_part"] = format_date(dates_last[1])
            #LOGGER.info(f"Second date found in last part's source_stem: {result['second_date_from_last_part']}")
        else:
            LOGGER.warning("Not enough date strings found in the last part's source_stem.")
    else:
        LOGGER.warning("The metadata does not contain a valid last part.")
    return result

@st.cache_data(show_spinner=False)
def get_references(doi_str):
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
# SECTION 4: Chunked Embeddings, Semantic Search, and Date Filtering
# =============================================================================

def get_location_for_index(global_idx: int, metadata: dict) -> dict:
    intervals = []
    for chunk in metadata["chunks"]:
        for part in chunk["parts"]:
            part_global_start = chunk["global_start"] + part["chunk_local_start"]
            part_global_end = chunk["global_start"] + part["chunk_local_end"] - 1
            intervals.append({
                "global_start": part_global_start,
                "global_end": part_global_end,
                "source_stem": part["source_stem"],
                "local_start": part["source_local_start"],
                "local_end": part["source_local_end"]
            })
    intervals.sort(key=lambda x: x["global_start"])
    for interval in intervals:
        if interval["global_start"] <= global_idx <= interval["global_end"]:
            offset = global_idx - interval["global_start"]
            local_idx = interval["local_start"] + offset
            return {"source_stem": interval["source_stem"], "local_idx": local_idx}
    return None

def check_date_on_the_fly_grouped(global_indices, metadata, data_folder, start_date, end_date):
    """
    Filters candidate global indices by reading only the date column from each corresponding
    parquet file and checking if the candidate's date falls between start_date and end_date.
    
    This function groups candidates by file and processes each file in parallel (using 4 workers).
    For arXiv files (detected by matching "arxiv_deduped" in the file stem), the column "update_date" is used;
    for others, the "date" column is used.
    
    Returns a list of candidate global indices that pass the time filter.
    """

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Error parsing dates: {e}")
    
    # Group candidates by file_stem
    file_to_candidates = {}
    for idx in global_indices:
        location = get_location_for_index(idx, metadata)
        if location is None:
            continue
        file_stem = location["source_stem"]
        local_idx = location["local_idx"]
        file_to_candidates.setdefault(file_stem, []).append((idx, local_idx))
    
    def process_file(file_stem, candidates):
        filtered = []
        parquet_path = os.path.join(data_folder, f"{file_stem}.parquet")
        if not os.path.exists(parquet_path):
            LOGGER.info(f"File not found: {parquet_path}")
            return filtered
        
        # Determine the date column by checking the parquet schema.
        try:
            parquet_file = pq.ParquetFile(parquet_path)
            schema = parquet_file.schema_arrow
            if "update_date" in schema.names:
                date_col = "update_date"
                LOGGER.info(f"Using 'update_date' column for file {file_stem}")
            elif "date" in schema.names:
                date_col = "date"
            else:
                LOGGER.error(f"Neither 'update_date' nor 'date' found in {parquet_path}")
                return filtered
        except Exception as e:
            LOGGER.error(f"Error reading schema from {parquet_path}: {e}")
            return filtered
        
        LOGGER.info(f"Reading Parquet file {parquet_path} for date filtering using column: {date_col}")
        try:
            table = pq.read_table(parquet_path, columns=[date_col])
            # Reset index to ensure we can use iloc.
            df = table.to_pandas().reset_index(drop=True)
            df["date_converted"] = pd.to_datetime(df[date_col], errors="coerce")
        except Exception as e:
            LOGGER.error(f"Error reading {parquet_path}: {e}")
            return filtered
        
        for global_idx, local_idx in candidates:
            if local_idx < 0 or local_idx >= len(df):
                continue
            try:
                row_date = df.iloc[local_idx]["date_converted"]
            except Exception as e:
                LOGGER.error(f"Error accessing row {local_idx} in {parquet_path}: {e}")
                continue
            if pd.isna(row_date):
                continue
            if start_dt <= row_date.to_pydatetime() <= end_dt:
                filtered.append(global_idx)
        return filtered

    filtered_indices = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, file_stem, candidates)
                   for file_stem, candidates in file_to_candidates.items()]
        for future in concurrent.futures.as_completed(futures):
            filtered_indices.extend(future.result())
    return filtered_indices

def build_sorted_intervals_from_metadata(metadata: dict) -> list:
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

    if start_date and end_date:
        candidate_indices = [hit["corpus_id"] for hit in all_hits]
        filtered_candidate_indices = check_date_on_the_fly_grouped(candidate_indices, metadata, folder, start_date, end_date)
        LOGGER.info(f"Time filter applied: {len(filtered_candidate_indices)} out of {len(candidate_indices)} hits remain.")
        all_hits = [hit for hit in all_hits if hit["corpus_id"] in filtered_candidate_indices]

    if not all_hits:
        LOGGER.info("Warning: No search results found across all chunks after time filtering.")
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

    if use_high_quality:
        LOGGER.info(f"Using high quality mode with {len(all_hits)} candidates to find top {top_k} quality results.")
        with log_time("Loading and filtering high-quality data"):
            top_hits, data = load_data_for_indices(
                all_hits,
                metadata,
                folder=folder,
                use_high_quality=True,
                top_k=top_k
            )
    else:
        top_hits = all_hits[:top_k]
        global_indices = [hit["corpus_id"] for hit in top_hits]
        with log_time("Loading data for selected global indices"):
            data = load_data_for_indices(global_indices, metadata, folder=folder, use_high_quality=False, top_k=top_k)

    if not data.empty:
        top_hits_df = pd.DataFrame(top_hits)
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
    LOGGER.info(f"Final search results prepared: {len(top_hits)} hits, {len(data)} data rows.")
    return top_hits, data

def load_data_for_indices(indices_or_hits: list, metadata: dict, folder: str, use_high_quality: bool = False, top_k: int = 50):
    if not use_high_quality:
        LOGGER.info("Loading data for given global indices from Parquet files (parallel optimized)...")
        intervals = build_sorted_intervals_from_metadata(metadata)
        file_indices = {}
        for idx in indices_or_hits:
            location = find_file_for_index_in_metadata_interval(idx, intervals)
            if location is not None:
                file_name = location["source_stem"]
                file_indices.setdefault(file_name, []).append(location["local_idx"])
            else:
                LOGGER.warning("Global index {} not found in metadata intervals.".format(idx))
        
        def process_file(file_name, local_indices):
            parquet_path = os.path.join(folder, "{}.parquet".format(file_name))
            LOGGER.info("Processing file: {} with indices: {}".format(parquet_path, local_indices))
            if not os.path.exists(parquet_path):
                LOGGER.warning("Parquet file not found for source {} at {}".format(file_name, parquet_path))
                return None
            try:
                parquet_file = pq.ParquetFile(parquet_path)
            except Exception as e:
                LOGGER.error("Error opening Parquet file {}: {}".format(parquet_path, e))
                return None
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
                    LOGGER.warning("Local index {} not found in any row group.".format(local_idx))
            results = []
            for rg, indices in rg_to_indices.items():
                indices.sort()
                try:
                    table = parquet_file.read_row_group(rg)
                except Exception as e:
                    LOGGER.error("Error reading row group {} from {}: {}".format(rg, parquet_path, e))
                    continue
                pa_indices = pa.array(indices, type=pa.int64())
                try:
                    subset_table = table.take(pa_indices)
                except Exception as e:
                    LOGGER.error("Error taking indices {} from row group {}: {}".format(indices, rg, e))
                    continue
                df_subset = subset_table.to_pandas()
                df_subset['source_file'] = os.path.basename(parquet_path)
                results.append(df_subset)
                del table, subset_table
            del parquet_file
            if results:
                return pd.concat(results, ignore_index=True)
            return None

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_file, file_name, local_indices)
                       for file_name, local_indices in file_indices.items()]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
        if not results:
            LOGGER.info("No data loaded. Returning empty DataFrame.")
            return pd.DataFrame()
        df_combined = pd.concat(results, ignore_index=True)
        LOGGER.info("Completed loading data for all global indices (parallel optimized).")
        return df_combined

    else:
        LOGGER.info("Loading high-quality data for candidate global indices (parallel optimized)...")
        accepted_hits = []
        accepted_rows = []
        intervals = build_sorted_intervals_from_metadata(metadata)
        min_abstract_length = 75
        file_hits = {}
        for hit in indices_or_hits:
            global_idx = hit["corpus_id"]
            location = find_file_for_index_in_metadata_interval(global_idx, intervals)
            if location is None:
                LOGGER.warning("Global index {} not found in metadata intervals.".format(global_idx))
                continue
            file_name = location["source_stem"]
            local_idx = location["local_idx"]
            file_hits.setdefault(file_name, []).append((hit, local_idx))
        accepted_counter = [0]
        counter_lock = threading.Lock()

        def process_file_high_quality(file_name, hits_local, accepted_counter, counter_lock):
            parquet_path = os.path.join(folder, "{}.parquet".format(file_name))
            if not os.path.exists(parquet_path):
                LOGGER.warning("Parquet file not found for source {} at {}".format(file_name, parquet_path))
                return []
            try:
                parquet_file = pq.ParquetFile(parquet_path)
            except Exception as e:
                LOGGER.error("Error opening Parquet file {}: {}".format(parquet_path, e))
                return []
            num_row_groups = parquet_file.num_row_groups
            group_boundaries = []
            total_rows = 0
            for rg in range(num_row_groups):
                rg_num_rows = parquet_file.metadata.row_group(rg).num_rows
                group_boundaries.append((total_rows, total_rows + rg_num_rows - 1))
                total_rows += rg_num_rows
            starts = np.array([start for start, end in group_boundaries])
            ends = np.array([end for start, end in group_boundaries])
            accepted = []
            for hit, local_idx in hits_local:
                with counter_lock:
                    if accepted_counter[0] >= top_k:
                        break
                rg = np.searchsorted(ends, local_idx, side='right')
                if rg < len(group_boundaries) and local_idx >= starts[rg]:
                    relative_idx = local_idx - starts[rg]
                    try:
                        table = parquet_file.read_row_group(rg)
                    except Exception as e:
                        LOGGER.error("Error reading row group {} from {}: {}".format(rg, parquet_path, e))
                        continue
                    pa_indices = pa.array([relative_idx], type=pa.int64())
                    try:
                        subset_table = table.take(pa_indices)
                    except Exception as e:
                        LOGGER.error("Error taking index {} from row group {}: {}".format(relative_idx, rg, e))
                        continue
                    df_subset = subset_table.to_pandas()
                    df_subset['source_file'] = os.path.basename(parquet_path)
                    title_valid = ('title' in df_subset.columns and
                                   df_subset.at[0, 'title'] is not None and
                                   str(df_subset.at[0, 'title']).strip() != "")
                    abstract_valid = ('abstract' in df_subset.columns and
                                      df_subset.at[0, 'abstract'] is not None and
                                      len(str(df_subset.at[0, 'abstract']).strip()) > min_abstract_length)
                    if title_valid and abstract_valid:
                        accepted.append((hit, df_subset))
                        with counter_lock:
                            accepted_counter[0] += 1
                            if accepted_counter[0] >= top_k:
                                del table, subset_table
                                break
                    else:
                        LOGGER.info("Rejected global index {} due to missing or insufficient title/abstract.".format(hit['corpus_id']))
                    del table, subset_table
                else:
                    LOGGER.warning("Local index {} not found in any row group in file {}.".format(local_idx, file_name))
            del parquet_file
            return accepted

        accepted_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_file_high_quality, file_name, hits_local, accepted_counter, counter_lock)
                       for file_name, hits_local in file_hits.items()]
            for future in concurrent.futures.as_completed(futures):
                accepted_results.extend(future.result())
        accepted_results.sort(key=lambda x: indices_or_hits.index(x[0]))
        for hit, df_subset in accepted_results:
            accepted_hits.append(hit)
            accepted_rows.append(df_subset)
            if len(accepted_hits) >= top_k:
                break
        if not accepted_hits:
            LOGGER.info("No high quality data found. Returning empty DataFrame.")
            return [], pd.DataFrame()
        df_combined = pd.concat(accepted_rows, ignore_index=True)
        LOGGER.info("Completed loading high quality data for global indices (parallel optimized).")
        return accepted_hits, df_combined

class ChunkedEmbeddings:
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
    LOGGER.info(f"Creating chunked embeddings in {chunk_dir} with pattern {npy_files_pattern}...")
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

def perform_semantic_search_chunks_wrapper(query_embedding: str,
                                           metadata: dict,
                                           chunk_dir: str,
                                           folder: str,
                                           precision: str = "ubinary",
                                           top_k: int = 50,
                                           corpus_index=None,
                                           use_high_quality: bool = False,
                                           start_date: str = None,
                                           end_date: str = None):
    return perform_semantic_search_chunks(query_embedding, metadata, chunk_dir, folder,
                                            precision, top_k, corpus_index, use_high_quality, start_date, end_date)

# =============================================================================
# SECTION 5: DataFrame Reformatting Functions
# =============================================================================

def reformat_biorxiv_df(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.copy()
    
    # check if df is empty. If so, return the mock empty dataframe with proper columns
    if df.empty:
        return pd.DataFrame(columns=["doi", "title", "authors", "date", "version", "type", "journal", "abstract", "score", "embedding"])
    
    if "doi" not in df.columns and "id" in df.columns:
        df["doi"] = df["id"].apply(lambda x: f"https://arxiv.org/abs/{x}")
    else:
        df["doi"] = df["doi"].fillna("").astype(str).str.strip()
        df.loc[df["doi"] == "", "doi"] = df["id"].apply(lambda x: f"https://arxiv.org/abs/{x}")
    df["date"] = pd.to_datetime(df["update_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["journal"] = df["journal-ref"] if "journal-ref" in df.columns else None
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
    if "versions" in df.columns:
        df["version"] = df["versions"].apply(get_version_count)
    else:
        df["version"] = None
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
    LOGGER.info("Loading SentenceTransformer model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(f"Using device: {device}")
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    model.to(device)
    LOGGER.info("Model loaded and moved to the appropriate device.")
    return model

@st.cache_resource
def load_pumap_model_and_image(model_path: str, image_path: str) -> tuple:
    LOGGER.info("Loading PuMAP model and UMAP image...")
    model = load_pumap(model_path)
    if hasattr(model, "to"):
        model.to("cpu")
    image = np.load(image_path)
    LOGGER.info("PuMAP model and UMAP image loaded.")
    return model, image

def plot_embedding_network_advanced(query: str, query_embedding, final_results: pd.DataFrame, metric: str = "hamming") -> go.Figure:
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

def combined_search(query: str, configs: list, top_show: int = 10, precision: str = "ubinary", use_high_quality: bool = False,
                    start_date: str = None, end_date: str = None) -> pd.DataFrame:

    # 1. Encode Query
    query_embedding = get_query_embedding(query, normalize=True, precision=precision)
    if query_embedding is None: return pd.DataFrame()
    query_embedding = np.array(query_embedding, dtype=np.uint8)
    if query_embedding.ndim == 1: query_embedding = query_embedding[np.newaxis, :]
    st.session_state["query_embedding"] = query_embedding

    all_dfs = []

    # Helper to Load -> Search -> Close immediately
    def process_source(name, config, reformat_func=None):
        try:
            # LOAD
            chunk_obj, metadata = create_chunked_embeddings_memmap(
                embeddings_directory=config["embeddings_directory"],
                npy_files_pattern=config["npy_files_pattern"],
                chunk_dir=config["chunk_dir"],
                metadata_path=config["metadata_path"],
                chunk_size_bytes=config.get("chunk_size_bytes", 536870912), # Reduced to 512MB for safety
            )
            # SEARCH
            # Note: We fetch 'top_show' from EACH DB to ensure we have enough good candidates
            _, df = perform_semantic_search_chunks_wrapper(
                query_embedding, metadata,
                chunk_dir=config["chunk_dir"],
                folder=config["data_folder"],
                precision=precision, top_k=top_show,
                use_high_quality=use_high_quality,
                start_date=start_date, end_date=end_date
            )
            # CLOSE & CLEANUP
            chunk_obj.close()
            del chunk_obj
            del metadata
            gc.collect()

            if not df.empty:
                df["source"] = name
                if reformat_func: df = reformat_func(df)
                return df
        except Exception as e:
            LOGGER.error(f"Error in {name}: {e}")
        return pd.DataFrame()

    # Sequential execution to prevent RAM saturation
    STATUS.info("Searching PubMed...")
    all_dfs.append(process_source("PubMed", pubmed_config))

    STATUS.info("Searching BioRxiv...")
    all_dfs.append(process_source("BioRxiv", biorxiv_config, reformat_biorxiv_df))

    STATUS.info("Searching MedRxiv...")
    all_dfs.append(process_source("MedRxiv", medrxiv_config, reformat_biorxiv_df))

    STATUS.info("Searching arXiv...")
    all_dfs.append(process_source("arXiv", arxiv_config, reformat_arxiv_df))

    STATUS.info("Ranking results...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    if combined_df.empty: return pd.DataFrame()

    # Normalize scores
    raw_min = combined_df["score"].min()
    raw_max = combined_df["score"].max()
    combined_df["quality"] = abs(combined_df["score"] - raw_max) + raw_min
    combined_df.sort_values(by="score", inplace=True)
    
    # Deduplicate
    if combined_df["doi"].notnull().all():
        combined_df = combined_df.drop_duplicates(subset=["doi"], keep="first")
    else:
        combined_df = combined_df.drop_duplicates(subset=["title"], keep="first")

    STATUS.empty()
    return combined_df.head(top_show).reset_index(drop=True)

# =============================================================================
# SECTION 8: Style, Logo, and AI Summary Functions
# =============================================================================

def define_style():
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
    active_users = get_current_active_users()
    pubmed_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/US-NLM-PubMed-Logo.svg/720px-US-NLM-PubMed-Logo.svg.png?20080121063734"
    biorxiv_logo = "https://www.biorxiv.org/sites/default/files/biorxiv_logo_homepage.png"
    medarxiv_logo = "https://www.medrxiv.org/sites/default/files/medRxiv_homepage_logo.png"
    arxiv_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/ArXiv_logo_2022.svg/640px-ArXiv_logo_2022.svg.png"
    
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
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

#@st.cache_data
def load_configs_and_db_sizes(config_yaml_path="config_mss.yaml"):
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

# --- Search Form ---
with st.form("search_form"):
    query = st.text_area("Enter your search query:", max_chars=8192, height=68)
    col1, col2 = st.columns(2)
    with col1:
        num_to_show = st.number_input("Number of results to show:", min_value=1, max_value=50, value=10)
    with col2:
        use_high_quality = st.toggle("Use high-quality search?", value=True,
                                     help="Enable for more accurate results (only entries with full abstracts and titles). For all results, disable this option.")
    # When date filtering is activated, include date inputs in the form.
    if st.session_state.get("date_filter_toggle", False):
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date_input = st.date_input("Start Date", value=datetime(2020, 1, 1))
        with col_date2:
            end_date_input = st.date_input("End Date", value=datetime.today())
        start_date_str = start_date_input.strftime("%Y-%m-%d")
        end_date_str = end_date_input.strftime("%Y-%m-%d")
    else:
        start_date_str = None
        end_date_str = None
    col3 = st.container()
    with col3:
        if st.session_state.get("use_ai_checkbox"):
            ai_col1, ai_col2 = st.columns(2)
            with ai_col1:
                ai_api_provided = st.text_input("Google AI API Key", value="", help="Obtain your own API key at https://aistudio.google.com/apikey", type="password")
            with ai_col2:
                ai_model_name = st.selectbox("AI Model", options=["gemini-2.0-flash-lite-preview-02-05", "gemini-2.0-flash"], index=0)
                ai_abstracts_count = st.number_input("Abstracts for Summary", min_value=1, max_value=20, value=9)
        else:
            ai_api_provided = None
            ai_model_name = "gemini-2.0-flash-lite-preview-02-05"
            ai_abstracts_count = 9
    submitted = st.form_submit_button("Search :material/search:", type="primary")

# --- Outside the Form: Toggles for AI Summary and Date Filter ---
col1, col2 = st.columns(2)
use_ai = col1.toggle("Use AI generated summary?", key="use_ai_checkbox")
activate_date_filter = col2.toggle("Use Date Filter", value=False, key="date_filter_toggle")
STATUS = st.empty()

st.markdown("---")

col1, col2 = st.columns(2)

if submitted and query:
    with st.spinner("Searching..."):
        search_start_time = datetime.now()
        final_results = combined_search(query, configs, top_show=num_to_show, precision="ubinary",
                                        use_high_quality=use_high_quality,
                                        start_date=start_date_str, end_date=end_date_str)
        total_time = datetime.now() - search_start_time
        st.markdown(f"<h6 style='text-align: center; color: #7882af;'>Search completed in {total_time.total_seconds():.2f} seconds</h6>", unsafe_allow_html=True)
        st.session_state["final_results"] = final_results
        st.session_state["search_query"] = query
        st.session_state["num_to_show"] = num_to_show
        st.session_state["use_ai"] = use_ai
        st.session_state["ai_api_provided"] = ai_api_provided
        st.session_state["use_high_quality"] = use_high_quality
else:
    final_results = st.session_state.get("final_results", pd.DataFrame())

if not final_results.empty:
    
    sort_option = col2.segmented_control(
        "Sort results by:", 
        options=["Relevance", "Publication Date", "Citations"], 
        key="sort_option"
    )
    # Ensure sorted_results is defined
    sorted_results = st.session_state["final_results"].copy()

    # Clean DOIs upfront (fast)
    sorted_results["doi"] = sorted_results["doi"].apply(get_clean_doi)

    # Sort the results.
    if sort_option == "Publication Date":
        sorted_results["date_parsed"] = pd.to_datetime(sorted_results["date"], errors="coerce")
        sorted_results = sorted_results.sort_values(by="date_parsed", ascending=False).reset_index(drop=True)
        sorted_results.drop(columns=["date_parsed"], inplace=True)
    elif sort_option == "Citations":
        # We need citations for sorting. Since we fetch them async now,
        # this might be tricky if we want instant sort before fetch.
        # But if the user clicks sort, we might need to wait or use cached.
        # For now, let's assume we use what we have or fetch if needed.
        # However, the async loop fetches them later.
        # If we sort by citations, we probably need them.
        # Let's check if citations are in the dataframe.
        if "citations" not in sorted_results.columns:
             # This part is tricky with async architecture.
             # If we want to sort by citations, we must have them.
             # The previous logic fetched all citations upfront.
             # Now we fetch them in the background.
             # If the user selects "Citations" sort, we should probably fetch them.
             with st.spinner("Fetching citations for sorting..."):
                 with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                     # Reuse the cached function
                     citations_map = list(executor.map(get_citation_count_cached, sorted_results["doi"]))
                 sorted_results["citations"] = citations_map

        sorted_results = sorted_results.sort_values(by="citations", ascending=False).reset_index(drop=True)
    else:
        # Default relevance (score)
        sorted_results = sorted_results.sort_values(by="score", ascending=True).reset_index(drop=True)

    st.markdown(f"#### Found {len(sorted_results)} relevant abstracts")

    # Precalculate full text links in parallel using ThreadPoolExecutor if not already there
    # This addresses "restore search for full documents and return links" while trying to keep it fast
    # We rely on cached helper functions.

    # We fetch citations and links for the displayed results
    # Ideally we should do this before rendering to "revert to the previous style"
    # But to speed it up we use parallel execution.

    displayed_rows = sorted_results.to_dict('records')

    with st.spinner("Fetching metadata..."):
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Prepare tasks
            future_to_idx = {}
            for i, row in enumerate(displayed_rows):
                # Submit citation fetch
                future_cit = executor.submit(get_citation_count_cached, row['doi'])
                # Submit link fetch
                row_dict = {"source": row["source"], "version": row["version"], "doi": row["doi"]}
                future_link = executor.submit(get_link_info_cached, row_dict)
                future_to_idx[future_cit] = (i, 'citations')
                future_to_idx[future_link] = (i, 'full_text_link')

            # Wait for all
            for future in concurrent.futures.as_completed(future_to_idx):
                idx, kind = future_to_idx[future]
                res = future.result()
                displayed_rows[idx][kind] = res

    # Convert back to DF if needed, or just iterate list
    # Revert to original expander style
    abstracts_for_summary = []

    for idx, row in enumerate(displayed_rows):
        citations = row.get("citations", 0)
        # Handle cases where citations is None
        if citations is None: citations = 0

        doi_val = row["doi"]
        doi_link = f"https://doi.org/{doi_val}" if "arxiv.org" not in doi_val else doi_val
        full_text_link = row.get("full_text_link")

        final_link = full_text_link if full_text_link is not None else doi_link
        full_text_notification = ":material/import_contacts:" if full_text_link is not None else ""
        has_full_text = full_text_link is not None

        expander_title = (
            f"{idx + 1}\. {row['title']}\n\n"
            f"_(Score: {row['quality']:.2f} | Date: {row['date']} | Citations: {citations})_"
        )
        if full_text_notification:
            expander_title += f" | {full_text_notification}"

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
            # Show links
            link_cols = st.columns(3)

            if has_full_text:
                link_cols[0].markdown(f"**[:material/import_contacts: Read Full Text]({final_link})**")
            link_cols[1].markdown(f"**[:material/link: View on Publisher Website]({doi_link})**")
        
        
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
    tabs = st.tabs(["Score vs Year", "Abstract Map", "References"])
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
            colorscale="dense",
            reversescale=False,
            showscale=False,
            opacity=1,
        )
        
        if "embedding" in final_results.columns:
            # Create raw embeddings array from final_results
            current_raw_embeddings = np.stack(final_results["embedding"].values)  # type: ignore
            
            # Check if we already have stored raw embeddings in session state
            if "stored_raw_embeddings" in st.session_state:
                # Compare stored embeddings with the current embeddings
                if np.array_equal(st.session_state["stored_raw_embeddings"], current_raw_embeddings):
                    # Use the stored UMAP result if the embeddings have not changed
                    new_2d = st.session_state["stored_umap_result"]
                else:
                    # Convert the new raw embeddings to a tensor and recalculate UMAP
                    current_tensor = torch.from_numpy(current_raw_embeddings).float()
                    new_2d = pumap.transform(current_tensor)
                    # Update session state with the new raw embeddings and UMAP result
                    st.session_state["stored_raw_embeddings"] = current_raw_embeddings
                    st.session_state["stored_umap_result"] = new_2d
            else:
                # First time calculation: convert to tensor and calculate UMAP
                current_tensor = torch.from_numpy(current_raw_embeddings).float()
                new_2d = pumap.transform(current_tensor)
                # Store the raw embeddings and UMAP result in session state
                st.session_state["stored_raw_embeddings"] = current_raw_embeddings
                st.session_state["stored_umap_result"] = new_2d

            # Extract UMAP dimensions for plotting
            scatter_x = new_2d[:, 0]
            scatter_y = new_2d[:, 1]
        else:
            st.error("final_results does not contain an 'embedding' column with high-dimensional embeddings.")
            scatter_x, scatter_y = [], []
        
        source_color_map = {
            "PubMed": "#4e79a7",
            "BioRxiv": "#ff6a6a",
            "MedRxiv": "#0f4d93",
            "arXiv": "#847c6f",
            "Query": "#1e1e1e",
        }
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
    
    with tabs[2]:
        st.markdown("#### References")
        # show authors, title, date, doi and link to the paper
        LOGGER.info(f"Showing references... {sorted_results.columns}")
        for idx, row in sorted_results.iterrows():
            doi = row["doi"]
            doi_link = f"https://doi.org/{doi}" if "arxiv.org" not in doi else doi
            
            authors = row["authors"] if row["authors"] else "Unknown"
            title = row["title"] if row["title"] else "No title available"
            date = row["date"] if row["date"] else "No date available"
            journal = row["journal"] if row["journal"] else "No journal available"
            if doi_link:
                doi_link = f"[{doi_link}]({doi_link})"
            else:
                doi_link = "No DOI available"
            st.markdown(f"**{idx + 1}\.** {authors}<br>{title} {date} {journal} <br> {doi_link}", unsafe_allow_html=True)
            
    if use_ai and abstracts_for_summary:
        with st.spinner("Generating AI summary..."):
            ai_gen_start = time.time()
            if st.session_state.get("ai_api_provided"):
                ai_count = st.session_state.get("ai_abstracts_count", 9)
                ai_model = st.session_state.get("ai_model_name", "gemini-2.0-flash-lite-preview-02-05")
                st.markdown(f"**AI Summary of top {ai_count} abstracts (Model: {ai_model}):**")
                with log_time("AI Summary Generation"):
                    summary_text = summarize_abstract(abstracts_for_summary[:ai_count], LLM_prompt, st.session_state["ai_api_provided"], model_name=ai_model)
                st.markdown(summary_text)
                LOGGER.info(summary_text)
            total_ai_time = time.time() - ai_gen_start
            st.markdown(f"**Time to generate summary:** {total_ai_time:.2f} seconds")
    STATUS.empty()

elif submitted and final_results.empty:
    st.warning("#### No results found. Please try a different query.")

st.markdown("---")
c1, c2 = st.columns([2, 1])

c1.markdown(
    """
    <div style='text-align: center;'>
        <b>[MSS] Developed by <a href="https://www.dzyla.com/" target="_blank">Dawid Zyla</a></b>
        |
        <a href="https://github.com/dzyla/pubmed_search" target="_blank">Source code on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)

c2.markdown(
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
