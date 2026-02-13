import os
import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
import logging
import concurrent.futures

LOGGER = logging.getLogger(__name__)

def build_sorted_intervals_from_metadata(metadata: dict) -> list:
    intervals = []
    if "chunks" in metadata:
        for chunk in metadata["chunks"]:
            for part in chunk["parts"]:
                part_global_start = chunk["global_start"] + part["chunk_local_start"]
                part_global_end = chunk["global_start"] + part["chunk_local_end"] - 1
                
                intervals.append({
                    "global_start": part_global_start,
                    "global_end": part_global_end,
                    "source_stem": part["source_stem"],
                    "source_local_start": part["source_local_start"]
                })
    intervals.sort(key=lambda x: x["global_start"])
    return intervals

def find_file_for_index(global_idx: int, intervals: list) -> dict:
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

def fetch_specific_rows(hits: list, metadata: dict, data_folder: str, combined_data_file: str = None) -> pd.DataFrame:
    intervals = build_sorted_intervals_from_metadata(metadata)
    file_requests = {} 
    
    for hit in hits:
        global_idx = hit['corpus_id']
        location = find_file_for_index(global_idx, intervals)
        
        if location:
            fname = location['source_stem']
            local_idx = location['local_idx']
            if fname not in file_requests:
                file_requests[fname] = []
            file_requests[fname].append((local_idx, hit))
    
    final_rows = []

    def process_file_request(fname, requests):
        parquet_path = os.path.join(data_folder, f"{fname}.parquet")
        
        if not os.path.exists(parquet_path):
            if combined_data_file and os.path.exists(combined_data_file):
                parquet_path = combined_data_file
            else:
                LOGGER.error(f"Data file not found: {parquet_path}")
                return []
            
        local_indices = [req[0] for req in requests]
        
        try:
            LOGGER.info(f"Fetching data from {parquet_path}: indices {local_indices}")
            dataset = ds.dataset(parquet_path, format="parquet")
            
            # --- START FIX: Dynamic Column Selection ---
            # 1. Define a "Wishlist" of columns we might want from ANY database
            #    We EXCLUDE "source" because we add that manually later.
            #    We INCLUDE "server" (MedRxiv), "journal" (PubMed), "posted" (BioRxiv), etc.
            wishlist = {
                "title", "abstract", "date", "doi", "authors", 
                "journal", "server", "journal-ref", "published", "posted", "update_date"
            }
            
            # 2. Check what the file ACTUALLY has
            #    dataset.schema.names gives us the list of columns in this specific file
            available_cols = set(dataset.schema.names)
            
            # 3. Intersect: Fetch only what exists
            columns_to_fetch = list(wishlist.intersection(available_cols))
            
            # 4. Fetch with projection (Speed boost + Safety)
            table = dataset.scanner(columns=columns_to_fetch).take(local_indices)
            df = table.to_pandas()
            # --- END FIX ---
            
            results = []
            for i, (_, hit_info) in enumerate(requests):
                if i < len(df):
                    row = df.iloc[i].to_dict()
                    row['score'] = hit_info['score']
                    results.append(row)
            return results
            
        except Exception as e:
            LOGGER.error(f"Error reading {parquet_path}: {e}")
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for fname, reqs in file_requests.items():
            futures.append(executor.submit(process_file_request, fname, reqs))
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            final_rows.extend(res)
            
    return pd.DataFrame(final_rows)