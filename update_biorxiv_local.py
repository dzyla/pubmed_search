#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bioRxiv + medRxiv downloader with JSON block cache + LOCAL embeddings (no server).

What is new:
- Before downloading, the script looks for cached month-block JSON files by exact date
  (e.g., details_data_250701_250731.json). If they exist and are readable, we use them first.
- Only missing/corrupt blocks are downloaded, then processed.
- JSON cache dirs are PERSISTENT; we do not delete them.
- JSON -> Parquet is robust (nested fields JSON-encoded). Parquet is written atomically.
- Embeddings are local (SentenceTransformer, precision='ubinary', uint8 .npy), with GPU and OOM backoff.

Dependencies:
  pip install pyyaml requests pandas pyarrow numpy torch sentence-transformers python-dateutil

Optional (only if enabling Dropbox upload):
  pip install dropbox streamlit
"""

import os
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, List, Tuple, Dict, Optional
from dateutil.relativedelta import relativedelta

import yaml
import requests
import numpy as np
import pandas as pd

# Optional Dropbox upload
try:
    import dropbox
    import streamlit as st
    HAS_DROPBOX = True
except Exception:
    HAS_DROPBOX = False

# Torch / ST for local embeddings
import torch
from sentence_transformers import SentenceTransformer


# --------------------- YAML Configuration ---------------------
def load_config(path="/mnt/h/pubmed_semantic_search/config_mss.yaml"):
    print("Loading YAML configuration...")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    print("YAML configuration loaded.")
    return cfg


# --------------------- Generic Helpers ---------------------
def robust_mkdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def is_nested_like(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, (list, dict, tuple, set, np.ndarray)):
        return True
    return False


def to_json_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if is_nested_like(x):
        try:
            return json.dumps(x, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(x)
    if isinstance(x, (bytes, bytearray, memoryview)):
        try:
            return x.decode("utf-8", errors="replace")
        except Exception:
            return x.hex() if hasattr(x, "hex") else str(bytes(x))
    return x


def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    for col in ("title", "abstract", "category", "doi", "server", "type", "license"):
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    for col in df.columns:
        s = df[col]
        sample = s.head(100)
        needs_json = False
        try:
            for v in sample:
                if is_nested_like(v):
                    needs_json = True
                    break
        except Exception:
            needs_json = True

        if s.dtype == "object":
            try:
                if any(not isinstance(v, (str, type(None))) and not (isinstance(v, float) and pd.isna(v)) for v in sample):
                    needs_json = True
            except Exception:
                needs_json = True

        if needs_json:
            df[col] = df[col].map(to_json_str)

        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].fillna("").astype("string")

    return df


def atomic_to_parquet(df: pd.DataFrame, save_path: str):
    save_dir = os.path.dirname(save_path)
    robust_mkdir(save_dir)
    tmp_path = f"{save_path}.tmp_write"
    df.to_parquet(tmp_path, engine="pyarrow", index=False)
    os.replace(tmp_path, save_path)


def save_dataframe(df, save_path):
    df_clean = sanitize_for_parquet(df.copy())
    atomic_to_parquet(df_clean, save_path)
    print(f"Saved DataFrame to {save_path}")


def load_json_to_dataframe(json_file: Path) -> pd.DataFrame:
    with open(json_file, 'r') as file:
        data = json.load(file)
    if isinstance(data, dict) and "collection" in data and isinstance(data["collection"], list):
        data = data["collection"]
    elif isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)


def process_json_files(output_directory: str, only_files: Optional[List[Path]] = None):
    """
    Convert selected JSON files to Parquet files in output_directory.
    If only_files is None, nothing happens (explicit by design).
    """
    robust_mkdir(output_directory)
    if not only_files:
        print("No JSON files provided to process_json_files; skipping.")
        return

    print(f"Processing {len(only_files)} JSON files -> {output_directory}")
    for json_file in only_files:
        df = load_json_to_dataframe(json_file)
        parquet_filename = f"{json_file.stem}.parquet"
        save_path = os.path.join(output_directory, parquet_filename)
        save_dataframe(df, save_path)
        print(f"Processed and saved {json_file.name} -> {parquet_filename}")


def list_unembedded_parquets(df_dir: str, embed_dir: str) -> List[Path]:
    df_dir = Path(df_dir)
    embed_dir = Path(embed_dir)
    robust_mkdir(embed_dir)

    parquet_files = list(df_dir.glob("*.parquet"))
    npy_stems = {f.stem for f in embed_dir.glob("*.npy")}

    missing = []
    for pf in sorted(parquet_files):
        if pf.stem not in npy_stems:
            missing.append(pf)
            print(f"Will embed (missing .npy): {pf.name}")
        else:
            print(f"Skipping (already embedded): {pf.name}")
    return missing


def get_last_checking_date(data_folder: str) -> str:
    data_folder = Path(data_folder)
    parquet_files = list(data_folder.glob("*.parquet"))
    max_date = None
    print(f"Scanning {len(parquet_files)} parquet files in {data_folder} for last checking date...")
    for file in parquet_files:
        try:
            df = pd.read_parquet(file, columns=['date'])
            if not df.empty and "date" in df.columns:
                file_max = pd.to_datetime(df["date"], errors="coerce").max()
                if pd.notna(file_max) and (max_date is None or file_max > max_date):
                    max_date = file_max
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if max_date is None or pd.isna(max_date):
        print("No valid dates found; using default start date 1990-01-01.")
        return "1990-01-01"

    last_date_str = max_date.strftime("%Y-%m-%d")
    print(f"Last checking date determined as: {last_date_str}")
    return last_date_str


# --------------------- Month-block utils + cache logic ---------------------
def month_blocks(start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime]]:
    blocks = []
    current = start_date
    while current <= end_date:
        bs = current
        be = min(current + relativedelta(months=1) - relativedelta(days=1), end_date)
        blocks.append((bs, be))
        current = current + relativedelta(months=1)
    return blocks


def json_block_name(endpoint: str, block_start: datetime, block_end: datetime) -> str:
    return f"{endpoint}_data_{block_start.strftime('%y%m%d')}_{block_end.strftime('%y%m%d')}.json"


def validate_json_file(path: Path) -> bool:
    """
    Light validation: file exists, > 0 bytes, loads as JSON,
    and either a list or dict with 'collection' list.
    """
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with path.open("r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return True
        if isinstance(data, dict):
            if "collection" in data and isinstance(data["collection"], list):
                return True
            # Some edge cases return empty collection or different shape; accept dicts too.
            return True
        return False
    except Exception:
        return False


def _fetch_block(base_url_prefix: str, block_start: datetime, block_end: datetime, save_directory: str, endpoint_for_name: str):
    """
    Download one month block and write to cache dir if any records found.
    """
    block_interval = f"{block_start.strftime('%Y-%m-%d')}/{block_end.strftime('%Y-%m-%d')}"
    block_data = []
    cursor = 0
    print(f"Starting fetch for block {block_interval} ...")
    while True:
        url = f"{base_url_prefix}{block_interval}/{cursor}/json"
        print(f"Requesting URL: {url}")
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            print(f"HTTP {r.status_code} at cursor {cursor}; stopping this block.")
            break
        data = r.json()
        fetched = len(data.get('collection', []))
        if fetched <= 0:
            break
        block_data.extend(data['collection'])
        cursor += fetched
        print(f"Fetched {fetched}, total so far: {cursor}")

    if block_data:
        robust_mkdir(save_directory)
        out_name = json_block_name(endpoint_for_name, block_start, block_end)
        out_path = Path(save_directory) / out_name
        with out_path.open('w') as file:
            json.dump(block_data, file, indent=2)
        print(f"Saved data block to {out_path}")
        return out_path
    else:
        print(f"No records for block {block_interval}.")
        return None


def ensure_cached_jsons(
    server: str,
    interval: str,
    cache_dir: str,
    endpoint_for_name: str = "details",
    max_workers: int = 12,
) -> Tuple[List[Path], List[Path]]:
    """
    Ensure we have JSON month-block files for the interval in cache_dir.

    Returns:
      (existing_files, newly_downloaded_files)
      existing_files: list of Paths already in cache and validated
      newly_downloaded_files: list of Paths we just downloaded for missing blocks
    """
    robust_mkdir(cache_dir)
    start_date, end_date = [datetime.strptime(d, "%Y-%m-%d") for d in interval.split('/')]
    blocks = month_blocks(start_date, end_date)

    # Where to download from
    if server == "biorxiv":
        base_url_prefix = "https://api.biorxiv.org/details/biorxiv/"
    elif server == "medrxiv":
        base_url_prefix = "https://api.medrxiv.org/details/medrxiv/"
    else:
        raise ValueError("server must be 'biorxiv' or 'medrxiv'.")

    existing: List[Path] = []
    missing: List[Tuple[datetime, datetime]] = []

    # First pass: identify cached blocks
    for bs, be in blocks:
        fname = json_block_name(endpoint_for_name, bs, be)
        fpath = Path(cache_dir) / fname
        if validate_json_file(fpath):
            existing.append(fpath)
            print(f"[cache hit] {fpath.name}")
        else:
            missing.append((bs, be))
            print(f"[cache miss] {fpath.name}")

    # Download missing in parallel
    downloaded: List[Path] = []
    if missing:
        print(f"Downloading {len(missing)} missing blocks for {server} into {cache_dir} ...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for bs, be in missing:
                tasks.append(ex.submit(_fetch_block, base_url_prefix, bs, be, cache_dir, endpoint_for_name))
            for fut in as_completed(tasks):
                out = fut.result()
                if isinstance(out, Path) and validate_json_file(out):
                    downloaded.append(out)
    else:
        print("All month blocks already cached; nothing to download.")

    # We will process existing first, then newly downloaded.
    existing_sorted = sorted(existing)
    downloaded_sorted = sorted(downloaded)
    return existing_sorted, downloaded_sorted


# --------------------- Local Embedding (PubMed-style) ---------------------
def build_queries_from_df(df: pd.DataFrame) -> List[str]:
    titles = df['title'] if 'title' in df.columns else [''] * len(df)
    abstracts = df['abstract'] if 'abstract' in df.columns else [''] * len(df)
    queries = []
    for t, a in zip(titles, abstracts):
        ts = str(t) if t is not None else ''
        aa = str(a) if a is not None else ''
        s = (ts.strip() + ' ' + aa.strip()).strip()
        queries.append(s)
    return queries


@torch.inference_mode()
def embed_file_to_npy(parquet_path: Path,
                      out_dir: Path,
                      model: SentenceTransformer,
                      device_type: str,
                      initial_batch_size: int = 96,
                      ubinary_dim_hint: int = 128):
    stem = parquet_path.stem
    out_path = out_dir / f"{stem}.npy"
    if out_path.exists():
        print(f"[skip exists] {out_path.name}")
        return

    try:
        df = pd.read_parquet(parquet_path, columns=['title', 'abstract'])
    except Exception:
        df = pd.read_parquet(parquet_path)

    queries = build_queries_from_df(df)
    robust_mkdir(out_dir)

    if len(queries) == 0:
        empty = np.empty((0, ubinary_dim_hint), dtype=np.uint8)
        np.save(out_path, empty)
        print(f"[empty] {out_path.name} (0,{ubinary_dim_hint})")
        return

    bs = initial_batch_size
    t0 = time.time()
    while True:
        try:
            if device_type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    embs = model.encode(
                        queries,
                        normalize_embeddings=True,
                        precision='ubinary',
                        batch_size=bs,
                        convert_to_numpy=True,
                        show_progress_bar=True,
                    )
            else:
                embs = model.encode(
                    queries,
                    normalize_embeddings=True,
                    precision='ubinary',
                    batch_size=bs,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                )

            if not isinstance(embs, np.ndarray):
                embs = np.array(embs)

            if embs.dtype != np.uint8:
                print(f"Warning: expected uint8 from 'ubinary', got {embs.dtype}; casting to uint8.")
                embs = np.ascontiguousarray(embs).astype(np.uint8)

            if embs.ndim != 2:
                raise RuntimeError(f"Unexpected embedding ndim {embs.ndim}; expected 2.")

            embs = np.ascontiguousarray(embs, dtype=np.uint8)
            np.save(out_path, embs)
            dt = time.time() - t0
            ips = len(queries) / dt if dt > 0 else 0
            print(f"[ok] {out_path.name}: {embs.shape} saved in {dt:.2f}s ({ips:.2f} items/s)")
            return

        except RuntimeError as e:
            msg = str(e)
            if 'CUDA out of memory' in msg and bs > 4:
                new_bs = max(4, bs // 2)
                print(f"OOM at batch_size={bs}. Reducing to {new_bs} and retrying…")
                bs = new_bs
                torch.cuda.empty_cache()
                continue
            raise


def embed_directory_local(df_dir: str,
                          embed_dir: str,
                          model_name: str = 'mixedbread-ai/mxbai-embed-large-v1',
                          batch_size: int = 96):
    df_dir_p = Path(df_dir)
    embed_dir_p = Path(embed_dir)
    robust_mkdir(embed_dir_p)

    files_to_process = list_unembedded_parquets(df_dir_p, embed_dir_p)
    if not files_to_process:
        print(f"No embeddings needed for {df_dir}.")
        return

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model '{model_name}' on {device_type}…")
    model = SentenceTransformer(model_name, device=device_type)

    if device_type == 'cuda':
        try:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"Using device: {name} with {mem:.1f} GB")
        except Exception:
            pass

    for pf in files_to_process:
        embed_file_to_npy(
            parquet_path=pf,
            out_dir=embed_dir_p,
            model=model,
            device_type=device_type,
            initial_batch_size=batch_size,
            ubinary_dim_hint=128,
        )

    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------- Optional: Dropbox upload ---------------------
def connect_to_dropbox():
    if not HAS_DROPBOX:
        raise RuntimeError("Dropbox dependencies not installed.")
    dropbox_APP_KEY = st.secrets["dropbox_APP_KEY"]
    dropbox_APP_SECRET = st.secrets["dropbox_APP_SECRET"]
    dropbox_REFRESH_TOKEN = st.secrets["dropbox_REFRESH_TOKEN"]
    dbx = dropbox.Dropbox(
        app_key=dropbox_APP_KEY,
        app_secret=dropbox_APP_SECRET,
        oauth2_refresh_token=dropbox_REFRESH_TOKEN
    )
    print("Connected to Dropbox.")
    return dbx


def upload_file(dbx, file_path: Path, dropbox_file_path: str):
    try:
        dropbox_file_path = dropbox_file_path.replace('\\', '/')
        try:
            metadata = dbx.files_get_metadata(dropbox_file_path)
            dropbox_mod_time = metadata.server_modified
            local_mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if dropbox_mod_time >= local_mod_time:
                print(f"Skipped {dropbox_file_path}, Dropbox version is up-to-date.")
                return
        except dropbox.exceptions.ApiError:
            pass
        with file_path.open('rb') as f:
            dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode.overwrite)
            print(f"Uploaded {dropbox_file_path}")
    except Exception as e:
        print(f"Failed to upload {dropbox_file_path}: {str(e)}")


def upload_path(local_path: str, dropbox_path: str):
    if not HAS_DROPBOX:
        print("Dropbox upload requested but dependencies are missing; skipping.")
        return
    dbx = connect_to_dropbox()
    lp = Path(local_path)
    if lp.is_file():
        rel = lp.name
        dp = os.path.join(dropbox_path, rel).replace('\\', '/').replace('//', '/')
        upload_file(dbx, lp, dp)
    elif lp.is_dir():
        for local_file in lp.rglob('*'):
            if local_file.is_file():
                rel = local_file.relative_to(lp.parent)
                dp = os.path.join(dropbox_path, str(rel)).replace('\\', '/').replace('//', '/')
                upload_file(dbx, local_file, dp)
    else:
        print("The provided path does not exist.")


# --------------------- Main Orchestration ---------------------
def main(do_biorxiv=True, do_medrxiv=True, do_upload=False):
    cfg = load_config("/mnt/h/pubmed_semantic_search/config_mss.yaml")

    bcfg = cfg.get("biorxiv_config", {})
    mcfg = cfg.get("medrxiv_config", {})

    biorxiv_df_dir = bcfg.get("data_folder", "biorxiv_df")
    biorxiv_embed_dir = bcfg.get("embeddings_directory", "biorxiv_embed")
    medrxiv_df_dir = mcfg.get("data_folder", "medrxiv_df")
    medrxiv_embed_dir = mcfg.get("embeddings_directory", "medrxiv_embed")

    # IMPORTANT: persistent JSON cache dirs; reuse your prior names so your existing files are picked up.
    json_cache_biorxiv = "db_update_json_temp_biorxiv"   # keep same path to reuse your current JSONs
    json_cache_medrxiv = "db_update_json_temp_medrxiv"

    today = datetime.today().strftime('%Y-%m-%d')

    if do_biorxiv:
        print("\n=== bioRxiv: cached JSON -> parquet -> local embed ===")
        start_bio = get_last_checking_date(biorxiv_df_dir)
        interval_bio = f"{start_bio}/{today}"
        print(f"bioRxiv interval: {interval_bio}")

        # 1) Use cached month-block JSONs first; 2) download only missing; 3) process both
        existing_jsons, new_jsons = ensure_cached_jsons(
            server="biorxiv",
            interval=interval_bio,
            cache_dir=json_cache_biorxiv,
            endpoint_for_name="details",
            max_workers=12,
        )

        # Process cached first (fast), then newly downloaded
        process_json_files(biorxiv_df_dir, only_files=existing_jsons)
        process_json_files(biorxiv_df_dir, only_files=new_jsons)

        # Now embed
        embed_directory_local(biorxiv_df_dir, biorxiv_embed_dir)

        if do_upload:
            print("Uploading bioRxiv df + embed to Dropbox…")
            upload_path(biorxiv_df_dir, "/")
            upload_path(biorxiv_embed_dir, "/")

    if do_medrxiv:
        print("\n=== medRxiv: cached JSON -> parquet -> local embed ===")
        start_med = get_last_checking_date(medrxiv_df_dir)
        interval_med = f"{start_med}/{today}"
        print(f"medRxiv interval: {interval_med}")

        existing_jsons, new_jsons = ensure_cached_jsons(
            server="medrxiv",
            interval=interval_med,
            cache_dir=json_cache_medrxiv,
            endpoint_for_name="details",
            max_workers=12,
        )

        process_json_files(medrxiv_df_dir, only_files=existing_jsons)
        process_json_files(medrxiv_df_dir, only_files=new_jsons)

        embed_directory_local(medrxiv_df_dir, medrxiv_embed_dir)

        if do_upload:
            print("Uploading medRxiv df + embed to Dropbox…")
            upload_path(medrxiv_df_dir, "/")
            upload_path(medrxiv_embed_dir, "/")


if __name__ == "__main__":
    main(do_biorxiv=True, do_medrxiv=True, do_upload=False)
