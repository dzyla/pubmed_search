import os
import json
import numpy as np
import faiss
import logging
import gc
import pandas as pd
from datetime import datetime
from pathlib import Path
import concurrent.futures
from data_handler import fetch_specific_rows
from utils import log_time

LOGGER = logging.getLogger(__name__)

def copy_file_into_chunk(file_info, memmap_array, offset):
    """
    Helper function to copy a source .npy file into the large memory-mapped chunk.
    """
    try:
        # allow_pickle=True added for safety with certain numpy versions/file formats
        arr = np.load(file_info["path"], mmap_mode="r", allow_pickle=True)
        rows = file_info["rows"]
        
        # Check dtype compatibility
        if arr.dtype != np.uint8 and memmap_array.dtype == np.uint8:
             pass 

        # Determine the slice in the chunk
        memmap_array[offset : offset + rows] = arr[:rows]
        
        return {
            "source_stem": file_info["stem"],
            "chunk_local_start": offset,
            "chunk_local_end": offset + rows,
            "source_local_start": 0
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
        # capture combined file if present
        self.combined_data_file = config.get("combined_data_file")
        
        # Default 1GB if not set
        self.chunk_size_bytes = config.get("chunk_size_bytes", 1 << 30)

        # 1. Ensure chunks exist (Regenerate if needed)
        self._ensure_chunks_exist()
        
        # 2. Load Metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            LOGGER.error(f"Failed to load metadata at {self.metadata_path}: {e}")
            return {}

    def _ensure_chunks_exist(self):
        """
        Checks if metadata and chunk files exist. If not, regenerates them from source embeddings.
        """
        recreate = False
        
        # Check metadata existence
        if not os.path.exists(self.metadata_path):
            LOGGER.info(f"Metadata missing at {self.metadata_path}. Triggering regeneration.")
            recreate = True
        else:
            # Validate existing metadata
            try:
                with open(self.metadata_path, 'r') as f:
                    meta = json.load(f)
                
                # Check for critical keys and EMPTY chunks
                if "embedding_dim" not in meta or "chunks" not in meta:
                    LOGGER.warning("Metadata missing 'embedding_dim' or 'chunks'. Triggering regeneration.")
                    recreate = True
                elif len(meta["chunks"]) == 0:
                     LOGGER.warning("Metadata exists but has 0 chunks. Triggering regeneration.")
                     recreate = True
                
                if not recreate:
                    # Check if all chunks listed in metadata actually exist
                    for chunk in meta.get("chunks", []):
                        c_path = os.path.join(self.chunk_dir, chunk["chunk_file"])
                        if not os.path.exists(c_path):
                            LOGGER.warning(f"Missing chunk file: {c_path}. Triggering regeneration.")
                            recreate = True
                            break
                
                if not recreate:
                    # Check for new files in embeddings directory not in metadata
                    try:
                        source_files = sorted(Path(self.embeddings_dir).glob(self.npy_pattern))
                        current_stems = set(p.stem for p in source_files)
                        
                        recorded_stems = set()
                        for chunk in meta.get("chunks", []):
                            for part in chunk.get("parts", []):
                                recorded_stems.add(part.get("source_stem"))
                        
                        if not current_stems.issubset(recorded_stems):
                            LOGGER.info("Found new files in embedding directory not in metadata. Triggering regeneration.")
                            recreate = True
                    except Exception as e:
                        LOGGER.warning(f"Error checking for new files: {e}")

            except Exception as e:
                LOGGER.warning(f"Error reading existing metadata: {e}. Triggering regeneration.")
                recreate = True

        if not recreate:
            return

        LOGGER.info(f"Starting chunk regeneration for {self.chunk_dir}...")
        os.makedirs(self.chunk_dir, exist_ok=True)
        
        # 1. Find all source NPY files
        source_files = sorted(Path(self.embeddings_dir).glob(self.npy_pattern))
        if not source_files:
            LOGGER.error(f"No source NPY files found in {self.embeddings_dir} with pattern {self.npy_pattern}")
            return

        # 2. Scan files to get total rows and dims
        file_infos = []
        total_rows = 0
        embedding_dim = None
        
        LOGGER.info(f"Scanning source files in {self.embeddings_dir}...")
        for p in source_files:
            try:
                # peek at shape without loading full data
                arr = np.load(p, mmap_mode="r", allow_pickle=True)
                rows = arr.shape[0]
                dims = arr.shape[1] if arr.ndim > 1 else 1
                
                # Check dtype
                if len(file_infos) == 0 and arr.dtype != np.uint8:
                    LOGGER.warning(f"Source file {p.name} is {arr.dtype}, but system expects uint8 (packed binary). Casting may occur.")

                if embedding_dim is None:
                    embedding_dim = dims
                elif dims != embedding_dim:
                    LOGGER.warning(f"Dimension mismatch in {p.name}: {dims} vs {embedding_dim}")
                    continue
                
                file_infos.append({
                    "stem": p.stem,
                    "path": str(p),
                    "rows": rows
                })
                total_rows += rows
            except Exception as e:
                LOGGER.error(f"Error reading source file {p}: {e}")

        if not file_infos:
            LOGGER.error("No valid source files found.")
            return
            
        LOGGER.info(f"Found {len(file_infos)} valid source files. Total rows: {total_rows}")

        # 3. Group files into chunks
        rows_per_chunk = self.chunk_size_bytes // (embedding_dim * 1) 
        
        chunks_metadata = []
        current_chunk_files = []
        current_chunk_rows = 0
        chunk_idx = 0
        global_start = 0

        # Helper to process a group of files into one chunk
        def process_chunk_group(group_files, group_rows, c_idx, g_start):
            chunk_filename = f"chunk_{c_idx}.npy"
            chunk_path = os.path.join(self.chunk_dir, chunk_filename)
            
            LOGGER.info(f"Creating {chunk_filename} ({group_rows} rows)...")
            
            # Create memmap
            memmap_arr = np.memmap(
                chunk_path, 
                dtype=np.uint8, # Packed binary
                mode='w+', 
                shape=(group_rows, embedding_dim)
            )
            
            parts = []
            current_offset = 0
            
            # Parallel copy
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for f_info in group_files:
                    futures.append(executor.submit(copy_file_into_chunk, f_info, memmap_arr, current_offset))
                    current_offset += f_info["rows"]
                
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res:
                        parts.append(res)
            
            # Flush changes to disk
            memmap_arr.flush()
            del memmap_arr
            
            # Sort parts by offset to keep metadata clean
            parts.sort(key=lambda x: x["chunk_local_start"])
            
            return {
                "chunk_file": chunk_filename,
                "global_start": g_start,
                "global_end": g_start + group_rows - 1,
                "actual_rows": group_rows,
                "parts": parts
            }

        # Grouping Logic
        for f_info in file_infos:
            if current_chunk_rows + f_info["rows"] > rows_per_chunk:
                # Process current batch
                if current_chunk_files:
                    meta = process_chunk_group(current_chunk_files, current_chunk_rows, chunk_idx, global_start)
                    chunks_metadata.append(meta)
                    global_start += current_chunk_rows
                    chunk_idx += 1
                    current_chunk_files = []
                    current_chunk_rows = 0
            
            current_chunk_files.append(f_info)
            current_chunk_rows += f_info["rows"]

        # Process final batch
        if current_chunk_files:
            meta = process_chunk_group(current_chunk_files, current_chunk_rows, chunk_idx, global_start)
            chunks_metadata.append(meta)
            global_start += current_chunk_rows

        # 4. Save Metadata
        final_metadata = {
            "total_rows": total_rows,
            "embedding_dim": embedding_dim,
            "chunks": chunks_metadata
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(final_metadata, f, indent=4)
            
        LOGGER.info(f"Regeneration complete. Metadata saved to {self.metadata_path}")

    def _search_chunk_worker(self, args):
        """
        Worker method to search a single chunk.
        Executed in parallel.
        """
        chunk_info, query_packed, limit, embedding_dim = args
        chunk_filename = chunk_info["chunk_file"]
        chunk_path = os.path.join(self.chunk_dir, chunk_filename)
        global_start = chunk_info["global_start"]
        actual_rows = chunk_info.get("actual_rows")
        
        if not os.path.exists(chunk_path):
            LOGGER.warning(f"Chunk file missing during search: {chunk_path}")
            return []
        
        if not actual_rows:
            return []

        chunk_candidates = []
        try:
            # Use np.memmap strictly
            chunk_data = np.memmap(
                chunk_path,
                dtype=np.uint8,
                mode="r",
                shape=(actual_rows, embedding_dim)
            )
            
            d_bits = embedding_dim * 8 
            
            # Initialize Binary Index
            index = faiss.IndexBinaryFlat(d_bits)
            index.add(chunk_data) 
            
            # Search
            distances, indices = index.search(query_packed, limit)
            
            local_ids = indices[0]
            dists = distances[0]
            
            for local_id, dist in zip(local_ids, dists):
                if local_id != -1:
                    score = 1.0 - (dist / d_bits)
                    chunk_candidates.append({
                        "corpus_id": global_start + int(local_id),
                        "score": score,
                        "chunk_file": chunk_filename
                    })
            
            # Clean up immediately
            del chunk_data
            del index
            
        except Exception as e:
            LOGGER.error(f"Error processing chunk {chunk_filename}: {type(e).__name__}: {e}")
        finally:
            # Force GC in thread
            gc.collect()
            
        return chunk_candidates

    def find_candidates_raw(self, query_packed, limit=100):
        """
        Scans all chunks in PARALLEL and returns a list of raw candidates (IDs and scores).
        Does NOT load data or Parquet files yet.
        """
        all_candidates = []
        
        if not self.metadata or "chunks" not in self.metadata:
            LOGGER.warning("Metadata invalid or empty.")
            return []

        # Retrieve embedding dimension from metadata
        embedding_dim = self.metadata.get("embedding_dim")
        if not embedding_dim:
             LOGGER.error("Metadata missing 'embedding_dim'. Cannot load memmaps.")
             return []

        chunks_list = self.metadata.get("chunks", [])
        
        # Prepare tasks for parallel execution
        # Each task: (chunk_info, query, limit, dim)
        tasks = []
        for chunk_info in chunks_list:
            tasks.append((chunk_info, query_packed, limit, embedding_dim))
            
        # Determine CPU count for parallel workers
        max_workers = 4 #os.cpu_count() or 4
        LOGGER.info(f"Scanning {len(tasks)} chunks with {max_workers} threads...")

        # Parallel Execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map returns results in order, which is fine
            results = executor.map(self._search_chunk_worker, tasks)
            
            for res in results:
                all_candidates.extend(res)

        return all_candidates
        
    def fetch_rows(self, candidates):
        """
        Fetches full data rows for a given list of candidates (which belong to this searcher's source).
        """
        if not candidates:
            return pd.DataFrame()
            
        df = fetch_specific_rows(candidates, self.metadata, self.data_folder, self.combined_data_file)
        return df


def combined_search_orchestrator(query_packed, configs, top_k, start_date=None, end_date=None, use_high_quality=False):
    """
    Orchestrates search with 'Live Deduplication' to ensure exactly top_k results are returned.
    """
    
    # 1. Initialize Searchers and Collect Raw Candidates
    sources_map = {
        "PubMed": configs[0],
        "BioRxiv": configs[1],
        "MedRxiv": configs[2],
        "arXiv": configs[3]
    }
    
    searchers = {}
    all_global_candidates = []
    
    # Increase raw limit significantly to account for filtering and duplicates
    # We grab 100x the requested amount to be safe
    raw_retrieval_limit = max(5000, top_k * 100)

    for source_name, config in sources_map.items():
        if not config: continue
        
        with log_time(f"Scanning {source_name} Index"):
            searcher = ChunkedSearcher(config)
            searchers[source_name] = searcher
            
            candidates = searcher.find_candidates_raw(query_packed, limit=raw_retrieval_limit)
            
            # Tag candidates with their source
            for c in candidates:
                c["source"] = source_name
            
            all_global_candidates.extend(candidates)
            LOGGER.info(f"Retrieved {len(candidates)} raw candidates from {source_name}")

    # 2. Sort Globally
    all_global_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    final_valid_rows = []
    seen_identifiers = set() # Track DOIs or Titles to prevent duplicates counting towards limit
    
    # 3. Batch Fetching & Filtering
    # We fetch slightly larger batches to reduce I/O overhead
    batch_size = max(50, top_k) 
    
    # Generator for batches
    for i in range(0, len(all_global_candidates), batch_size):
        # STOP CONDITION: Only break if we have enough UNIQUE results
        if len(final_valid_rows) >= top_k:
            break
            
        batch_candidates = all_global_candidates[i : i + batch_size]
        
        # Group batch by source to fetch efficiently
        candidates_by_source = {}
        for c in batch_candidates:
            s = c["source"]
            if s not in candidates_by_source:
                candidates_by_source[s] = []
            candidates_by_source[s].append(c)
        
        # Fetch rows for this batch
        batch_dfs = []
        for source_name, subset in candidates_by_source.items():
             searcher = searchers[source_name]
             try:
                 df_subset = searcher.fetch_rows(subset)
                 if not df_subset.empty:
                     df_subset["source"] = source_name
                     
                     # Normalize Columns
                     col_map = {c.lower(): c for c in df_subset.columns}
                     
                     if "server" in col_map: 
                         df_subset.rename(columns={col_map["server"]: "journal"}, inplace=True)
                     elif "journal-ref" in col_map:
                         df_subset["journal"] = df_subset[col_map["journal-ref"]]
                     elif "journal" not in col_map and "source_title" in col_map:
                         df_subset["journal"] = df_subset[col_map["source_title"]]
                     
                     if "date" not in df_subset.columns:
                        if "update_date" in df_subset.columns: df_subset["date"] = df_subset["update_date"]
                        elif "posted" in df_subset.columns: df_subset["date"] = df_subset["posted"]

                     required_cols = ["doi", "title", "authors", "date", "abstract", "score", "source", "journal"]
                     for col in required_cols:
                         if col not in df_subset.columns: df_subset[col] = None
                     
                     batch_dfs.append(df_subset[required_cols])
             except Exception as e:
                 LOGGER.error(f"Error fetching batch for {source_name}: {e}")

        if not batch_dfs:
            continue
            
        combined_batch_df = pd.concat(batch_dfs, ignore_index=True)
        
        # --- APPLY FILTERS ---
        
        # 1. Date Filter
        if start_date and end_date:
            date_mask = pd.to_datetime(combined_batch_df["date"], errors='coerce').between(
                pd.to_datetime(start_date), pd.to_datetime(end_date)
            )
            combined_batch_df = combined_batch_df[date_mask]
            
        # 2. Quality Filter
        if use_high_quality:
            qual_mask = combined_batch_df["abstract"].str.len() > 75
            qual_mask = qual_mask.fillna(False)
            combined_batch_df = combined_batch_df[qual_mask]

        # --- LIVE DEDUPLICATION ---
        # Only add to final list if not seen
        for _, row in combined_batch_df.iterrows():
            if len(final_valid_rows) >= top_k:
                break
                
            # Create unique key (DOI preferred, fallback to lower-case title)
            doi = str(row.get('doi', '')).strip()
            title = str(row.get('title', '')).strip().lower()
            
            # Skip if we have a valid DOI we've seen
            if doi and len(doi) > 5 and doi in seen_identifiers:
                continue
            # Skip if we have no DOI but have seen this title
            if (not doi or len(doi) < 5) and title in seen_identifiers:
                continue
                
            # Mark as seen
            if doi and len(doi) > 5: seen_identifiers.add(doi)
            if title: seen_identifiers.add(title)
            
            final_valid_rows.append(row)

    if not final_valid_rows:
        return pd.DataFrame()
    
    # Convert back to DataFrame
    result_df = pd.DataFrame(final_valid_rows)
    
    # Final Heuristics (BioRxiv labeling etc)
    if "journal" in result_df.columns:
        result_df["journal"] = result_df["journal"].astype(str).fillna("")
        mask_bio = (result_df["source"] == "PubMed") & (result_df["journal"].str.contains(r"biorxiv", case=False))
        result_df.loc[mask_bio, "source"] = "BioRxiv"
        mask_med = (result_df["source"] == "PubMed") & (result_df["journal"].str.contains(r"medrxiv", case=False))
        result_df.loc[mask_med, "source"] = "MedRxiv"
    
    if "doi" in result_df.columns:
        mask_preprint_doi = (result_df["source"] == "PubMed") & (result_df["doi"].str.contains("10.1101", na=False))
        result_df.loc[mask_preprint_doi, "source"] = "BioRxiv"

    return result_df.reset_index(drop=True)