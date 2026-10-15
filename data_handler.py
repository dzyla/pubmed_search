import os
import logging
import concurrent.futures

import numpy as np
import pyarrow.parquet as pq
import pandas as pd

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lock-free parquet metadata cache
#
# Maps parquet_path -> (FileMetaData, row-group start rows, wishlist columns, signature).
# A fetch opens the file with the cached footer and reads only the row groups that
# hold the wanted rows. (pyarrow's Dataset.take reads nearly the whole file for one
# row: 13 MB / 27 ms cold for a PubMed file, vs 0.2 MB / 2 ms for one 500-row group,
# measured on the server 2026-10-15.) No file stays open between fetches.
#
# Why no lock? CPython's GIL makes dict.get() and dict[k]=v atomic; two threads may
# rarely both miss and both store an entry — harmless (same content, last writer wins).
# ---------------------------------------------------------------------------
_PARQUET_META_CACHE: dict = {}

_WISHLIST = frozenset({
    "title", "abstract", "date", "doi", "authors",
    "journal", "server", "journal-ref", "published", "posted", "update_date",
    "pmid", "type", "version",
    "nct_id", "trial_status", "trial_phase", "has_results", "pmids",
    "published_pmid", "license",
    "pub_type", "pmcid",                       # OpenAlex
    "grant_id", "appl_id", "ic", "activity_code", "organization", "fiscal_year",
})


def _parquet_meta(parquet_path: str):
    """
    Returns cached (FileMetaData, row-group start rows, column names, wishlist columns).
    Re-reads the footer when the file's size or mtime changed (e.g. a weekly update
    rewrote it in place): stale row-group offsets would read garbage.
    """
    st = os.stat(parquet_path)
    signature = (st.st_mtime_ns, st.st_size)
    entry = _PARQUET_META_CACHE.get(parquet_path)
    if entry is None or entry[4] != signature:
        meta = pq.read_metadata(parquet_path)
        starts = np.cumsum([0] + [meta.row_group(i).num_rows for i in range(meta.num_row_groups)])
        names = meta.schema.to_arrow_schema().names
        entry = (meta, starts, names, [c for c in names if c in _WISHLIST], signature)
        _PARQUET_META_CACHE[parquet_path] = entry
    return entry[:4]


def read_parquet_rows(parquet_path: str, rows, columns=None):
    """
    ({column: [values]}, valid) for the given row numbers, in that order, reading only
    the row groups that contain them. Rows outside the file are not valid (values None).
    """
    meta, starts, names, wishlist = _parquet_meta(parquet_path)
    cols = wishlist if columns is None else [c for c in columns if c in names]
    rows = np.asarray(rows, dtype=np.int64)
    valid = (rows >= 0) & (rows < starts[-1])
    out = {c: [None] * len(rows) for c in cols}
    if not valid.any():
        return out, valid
    groups = np.searchsorted(starts, np.where(valid, rows, 0), side="right") - 1
    pf = pq.ParquetFile(parquet_path, metadata=meta)
    try:
        for g in np.unique(groups[valid]):
            pos = np.flatnonzero(valid & (groups == g))
            part = pf.read_row_group(int(g), columns=cols).take(rows[pos] - starts[g]).to_pydict()
            for c in cols:
                col = out[c]
                for j, v in zip(pos, part[c]):
                    col[j] = v
    finally:
        pf.close()
    return out, valid


def clear_parquet_cache():
    """Clears the cached parquet footers to allow reading new data."""
    _PARQUET_META_CACHE.clear()
    LOGGER.info("Cleared Parquet dataset cache.")


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
    columns: list = None,     # restrict to these columns (e.g. date only); default: _WISHLIST
) -> pd.DataFrame:
    """
    Fetches metadata rows from parquet files for the given FAISS hits.

    Optimisations vs original:
    - intervals pre-built once in ChunkedSearcher (no rebuild per call)
    - parquet footer + column list cached lock-free (no footer re-read)
    - reads only the row groups holding the wanted rows (read_parquet_rows)
    """
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
            raw, valid = read_parquet_rows(parquet_path, local_indices, columns)
            if not valid.all():
                LOGGER.error(f"{parquet_path}: {int((~valid).sum())} requested row(s) beyond its "
                             f"{_parquet_meta(parquet_path)[1][-1]:,} rows")
            results = []
            for i, (_, hit_info) in enumerate(requests):
                if valid[i]:
                    row = {k: raw[k][i] for k in raw}
                    row["score"] = hit_info["score"]
                    row["corpus_id"] = hit_info["corpus_id"]
                    if hit_info.get("matched_terms"):
                        row["matched_terms"] = hit_info["matched_terms"]
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
