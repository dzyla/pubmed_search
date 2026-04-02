import os
import logging
import concurrent.futures

import pyarrow.dataset as ds
import pandas as pd

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lock-free parquet dataset + column cache
#
# Maps parquet_path -> (PyArrow Dataset, columns_to_fetch list).
#
# Why no lock?
#   CPython's GIL makes dict.get() and dict[k]=v atomic.  Two threads may
#   rarely both miss the cache for the same path and both compute+store an
#   entry — that's harmless (deterministic output, last writer wins).
#   Every later call is a plain dict lookup with zero lock overhead.
#
# Memory cost: a few KB per file (schema metadata only, no row data cached).
# ---------------------------------------------------------------------------
_PARQUET_DATASET_CACHE: dict = {}

_WISHLIST = frozenset({
    "title", "abstract", "date", "doi", "authors",
    "journal", "server", "journal-ref", "published", "posted", "update_date",
})


def _get_or_open_dataset(parquet_path: str):
    """Returns cached (Dataset, columns_to_fetch). No lock — GIL is sufficient."""
    entry = _PARQUET_DATASET_CACHE.get(parquet_path)
    if entry is None:
        dataset = ds.dataset(parquet_path, format="parquet")
        cols = list(_WISHLIST.intersection(set(dataset.schema.names)))
        entry = (dataset, cols)
        _PARQUET_DATASET_CACHE[parquet_path] = entry
    return entry


# ---------------------------------------------------------------------------
# Interval helpers  (result cached in ChunkedSearcher.intervals)
# ---------------------------------------------------------------------------

def build_sorted_intervals_from_metadata(metadata: dict) -> list:
    intervals = []
    for chunk in metadata.get("chunks", []):
        g_start = chunk["global_start"]
        for part in chunk.get("parts", []):
            intervals.append({
                "global_start":       g_start + part["chunk_local_start"],
                "global_end":         g_start + part["chunk_local_end"] - 1,
                "source_stem":        part["source_stem"],
                "source_local_start": part["source_local_start"],
            })
    intervals.sort(key=lambda x: x["global_start"])
    return intervals


def find_file_for_index(global_idx: int, intervals: list) -> dict:
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        iv = intervals[mid]
        if iv["global_start"] <= global_idx <= iv["global_end"]:
            offset = global_idx - iv["global_start"]
            return {
                "source_stem": iv["source_stem"],
                "local_idx":   iv["source_local_start"] + offset,
            }
        elif global_idx < iv["global_start"]:
            hi = mid - 1
        else:
            lo = mid + 1
    return None


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def fetch_specific_rows(
    hits: list,
    metadata: dict,
    data_folder: str,
    combined_data_file: str = None,
    intervals: list = None,   # pre-built by ChunkedSearcher — avoids rebuild per query
) -> pd.DataFrame:
    """
    Fetches metadata rows from parquet files for the given FAISS hits.

    Optimisations vs original:
    - intervals pre-built once in ChunkedSearcher (no rebuild per call)
    - parquet schema + column list cached lock-free (no footer re-read)
    - to_pydict() replaces to_pandas() — skips numpy/dtype/index overhead
      for the small per-file result tables
    """
    import pandas as pd  # local import keeps module-level footprint minimal

    if intervals is None:
        intervals = build_sorted_intervals_from_metadata(metadata)

    # Group hits by source parquet file
    file_requests: dict = {}
    for hit in hits:
        loc = find_file_for_index(hit["corpus_id"], intervals)
        if loc:
            file_requests.setdefault(loc["source_stem"], []).append(
                (loc["local_idx"], hit)
            )

    if not file_requests:
        return pd.DataFrame()

    def _fetch_one(fname: str, requests: list) -> list:
        parquet_path = os.path.join(data_folder, f"{fname}.parquet")
        if not os.path.exists(parquet_path):
            if combined_data_file and os.path.exists(combined_data_file):
                parquet_path = combined_data_file
            else:
                LOGGER.error(f"Data file not found: {parquet_path}")
                return []

        local_indices = [r[0] for r in requests]
        try:
            dataset, columns_to_fetch = _get_or_open_dataset(parquet_path)
            table = dataset.scanner(columns=columns_to_fetch).take(local_indices)

            # to_pydict() is faster than to_pandas() for 1-10 row tables:
            # no numpy array allocation, no dtype inference, no DataFrame index.
            raw = table.to_pydict()
            results = []
            for i, (_, hit_info) in enumerate(requests):
                if i < len(table):
                    row = {k: raw[k][i] for k in raw}
                    row["score"] = hit_info["score"]
                    results.append(row)
            return results

        except Exception as e:
            LOGGER.error(f"Error reading {parquet_path}: {e}")
            return []

    # Fresh pool per query — same pattern as original.
    # Persistent pools add sleeping-thread wake-up latency and compete with
    # FAISS / Streamlit threads for CPU scheduler slots on 4-CPU hardware.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_fetch_one, fname, reqs)
            for fname, reqs in file_requests.items()
        ]
        final_rows: list = []
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                final_rows.extend(res)

    return pd.DataFrame(final_rows)
