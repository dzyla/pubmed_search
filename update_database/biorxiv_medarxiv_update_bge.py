import os
import json
import time
import shutil
import yaml
import requests
import glob
import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

# -----------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_ID = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 1024  # BGE is small, we can increase batch size
CONFIG_PATH = "/home/dzyla/pubmed_search/snowflake_code/config_mss.yaml"

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIG LOADING
# -----------------------------------------------------------------------------
print(f"Loading YAML configuration from {CONFIG_PATH}...")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config_yaml = yaml.safe_load(f)
else:
    print(f"WARNING: Config file not found at {CONFIG_PATH}. Using empty defaults.")
    config_yaml = {}

biorxiv_conf = config_yaml.get("biorxiv_config", {})
medrxiv_conf = config_yaml.get("medrxiv_config", {})

SOURCES = [
    {
        "name": "BioRxiv",
        "server": "biorxiv",
        "endpoint": "details",
        "config_section": biorxiv_conf,
        "embed_filename": "biorxiv_binary_bge.npy", # Updated filename to distinguish
        "state_filename": "fetch_state.json"
    },
    {
        "name": "MedRxiv",
        "server": "medrxiv",
        "endpoint": "details",
        "config_section": medrxiv_conf,
        "embed_filename": "medarxiv_binary_bge.npy", # Updated filename
        "state_filename": "fetch_state.json"
    }
]

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def retry_on_exception(exception, retries=5, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return func(*args, **kwargs)
                except exception as e:
                    print(f"Error in {func.__name__}: {str(e)}. Retrying ({i+1}/{retries})...")
                    time.sleep(delay)
            raise exception
        return wrapper
    return decorator

def get_state(work_dir, state_file_name):
    state_path = os.path.join(work_dir, state_file_name)
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
                return state.get('last_fetch_date', '1990-01-01')
        except Exception:
            return '1990-01-01'
    return '1990-01-01'

def update_state(work_dir, state_file_name, new_date_str):
    state_path = os.path.join(work_dir, state_file_name)
    with open(state_path, 'w') as f:
        json.dump({'last_fetch_date': new_date_str}, f)
    print(f"Updated fetch clock to: {new_date_str}")

def build_input_texts(df):
    """
    Constructs text for BGE.
    Format: "Title. Abstract"
    """
    titles = df['title'].fillna('').astype(str)
    abstracts = df['abstract'].fillna('').astype(str)
    return (titles + ". " + abstracts).tolist()

def generate_embeddings_batched(model, texts, batch_size=BATCH_SIZE, desc="Embedding"):
    """
    Generates PACKED BINARY embeddings (uint8).
    Automatically detects dimension size.
    """
    total_samples = len(texts)
    
    # Dynamic dimension calculation
    # BGE-Small (384 dims) / 8 = 48 bytes
    # Snowflake (768 dims) / 8 = 96 bytes
    float_dim = model.get_sentence_embedding_dimension()
    binary_dim = float_dim // 8
    
    binary_embeddings = np.zeros((total_samples, binary_dim), dtype=np.uint8)
    
    total_batches = (total_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, total_samples, batch_size), total=total_batches, desc=desc):
        batch_texts = texts[i : i + batch_size]
        
        # 1. Encode Float (Normalized)
        emb_float = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True # Critical for binary quantization
        )
        
        # 2. Quantize & Pack
        # Float > 0 becomes 1, else 0. Then packed into uint8.
        packed = np.packbits(emb_float > 0, axis=1)
        
        binary_embeddings[i : i + len(batch_texts)] = packed
        
    return binary_embeddings

# -----------------------------------------------------------------------------
# 3. FETCHING LOGIC
# -----------------------------------------------------------------------------
def save_data_block_json(block_data, start_date, end_date, endpoint, save_directory):
    start_yymmdd = start_date.strftime("%y%m%d")
    end_yymmdd = end_date.strftime("%y%m%d")
    filename = f"{save_directory}/{endpoint}_data_{start_yymmdd}_{end_yymmdd}.json"
    with open(filename, 'w') as file:
        json.dump(block_data, file, indent=4)

def _get_with_retry(url, retries=3, backoff=3):
    """GET with exponential backoff for any requests.exceptions.RequestException.
    SSL verification is disabled because api.medrxiv.org serves an expired certificate.
    timeout=(10, 30): 10s to establish TCP connection, 30s to receive data.
    Using the tuple form is critical — a single int only sets the read timeout and
    can leave the SSL handshake hanging indefinitely on some systems (e.g. WSL).
    """
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    for attempt in range(retries):
        print(f"    → GET {url}")
        try:
            return requests.get(url, timeout=(10, 30), verify=False)
        except requests.exceptions.RequestException as e:
            wait = backoff * (2 ** attempt)
            print(f"  ⚠ Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt + 1 < retries:
                print(f"    Waiting {wait}s before retry...")
                time.sleep(wait)
    raise requests.exceptions.RetryError(f"All {retries} retries exhausted for {url}")


def fetch_block(endpoint, server, block_start, block_end, save_directory):
    """
    Returns True on success, False if the block could not be fetched.
    Skips fetching if the JSON file already exists on disk (resumable).
    """
    if server == "biorxiv":
        base_url = f"https://api.biorxiv.org/{endpoint}/{server}/"
    else:
        base_url = f"https://api.medrxiv.org/{endpoint}/{server}/"

    # Check if this block was already fetched in a previous (possibly crashed) run.
    start_yymmdd = block_start.strftime("%y%m%d")
    end_yymmdd   = block_end.strftime("%y%m%d")
    existing_file = os.path.join(save_directory, f"{endpoint}_data_{start_yymmdd}_{end_yymmdd}.json")
    if os.path.exists(existing_file):
        print(f"Skipping {server} {block_start.date()} → {block_end.date()} (already on disk).")
        return True

    block_interval = f"{block_start.strftime('%Y-%m-%d')}/{block_end.strftime('%Y-%m-%d')}"
    block_data = []
    cursor = 0
    continue_fetching = True
    page = 0

    print(f"[{server}] Starting block {block_interval}...")
    while continue_fetching:
        url = f"{base_url}{block_interval}/{cursor}/json"
        try:
            response = _get_with_retry(url)
        except requests.exceptions.RequestException as e:
            print(f"[{server}] Network error fetching {url}: {e}")
            break

        if response.status_code != 200:
            print(f"[{server}] Failed {block_interval} at cursor {cursor}. Status: {response.status_code}")
            break

        try:
            data = response.json()
        except ValueError:
            print(f"[{server}] Invalid JSON response from {url}")
            break

        fetched_count = len(data.get('collection', []))
        if fetched_count > 0:
            block_data.extend(data['collection'])
            cursor += fetched_count
            page += 1
            # Print every 10 pages so the user can see progress without flood
            if page % 10 == 0:
                print(f"  [{server}] {block_interval} — page {page}, {cursor:,} records so far...")
        else:
            continue_fetching = False

    if block_data:
        print(f"  [{server}] Block {block_interval} done — {len(block_data):,} records.")
        save_data_block_json(block_data, block_start, block_end, endpoint, save_directory)
        return True
    print(f"  [{server}] Block {block_interval} — no records returned.")
    return False

def convert_json_to_parquet(json_dir, parquet_dir):
    json_files = list(Path(json_dir).glob("*.json"))
    if not json_files:
        return []

    print(f"Converting {len(json_files)} JSON files to Parquet...")
    parquet_files = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
            
            if not data:
                continue

            df = pd.DataFrame(data)

            problematic = ['funder', 'authors', 'rel_authors', 'category', 'published', 'version']
            for col in problematic:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    df[col] = df[col].astype(str)
                except:
                    pass

            out_name = f"{json_file.stem}.parquet"
            out_path = os.path.join(parquet_dir, out_name)
            df.to_parquet(out_path)
            parquet_files.append(out_path)
        except Exception as e:
            print(f"Error converting {json_file.name}: {e}")
            
    return parquet_files

# -----------------------------------------------------------------------------
# 4. PROCESSING & INTEGRITY LOGIC
# -----------------------------------------------------------------------------
def calculate_hamming_similarity(bits_a, bits_b):
    """
    Calculates percentage of matching bits between two packed uint8 arrays.
    """
    xor_diff = np.bitwise_xor(bits_a, bits_b)
    diff_bits = np.unpackbits(xor_diff).sum()
    
    total_bits = len(bits_a) * 8
    matching_bits = total_bits - diff_bits
    similarity = matching_bits / total_bits
    return similarity, diff_bits

def check_database_integrity(meta_path, embed_path, model):
    """
    Randomly selects 10 rows, re-calculates embeddings, and checks against NPY file.
    """
    print(f"\n--- Running Database Integrity Check ---")
    try:
        df = pd.read_parquet(meta_path)
        embeddings = np.load(embed_path)
    except Exception as e:
        print(f"Skipping integrity check (Files not ready or missing): {e}")
        return

    total_rows = len(df)
    if total_rows == 0:
        return

    indices = random.sample(range(total_rows), min(10, total_rows))
    print(f"Verifying {len(indices)} random entries...")
    
    subset_df = df.iloc[indices].copy()
    texts_to_check = build_input_texts(subset_df)
    
    passed = True
    
    for i, idx in enumerate(indices):
        text = texts_to_check[i]
        
        # 1. Fresh embedding
        emb_float = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        # 2. Quantize
        packed_fresh = np.packbits(emb_float > 0)
        # 3. Stored
        packed_stored = embeddings[idx]
        
        # 4. Compare with Tolerance
        similarity, diff_bits = calculate_hamming_similarity(packed_fresh, packed_stored)
        
        # Threshold: 95% similarity
        if similarity < 0.95:
            print(f"❌ INTEGRITY FAILURE at index {idx}!")
            print(f"   Mismatch: {diff_bits} bits differ ({similarity:.2%} match)")
            passed = False
        else:
            if diff_bits > 0:
                print(f"⚠️  Index {idx}: Matches with minor noise ({diff_bits} bits flipped). OK.")
            # else: Exact match
    
    if not passed:
        raise ValueError("Database integrity check failed! Significant embedding mismatch detected.")
    
    print(f"✅ Integrity Check Passed. Database is consistent.")

def process_source(source, model, override_start: datetime | None = None):
    name = source['name']
    server = source['server']
    conf = source['config_section']
    
    work_dir = conf.get('data_folder')
    meta_path = conf.get('combined_data_file')
    embed_dir = conf.get('embeddings_directory')
    embed_path = os.path.join(embed_dir, source['embed_filename'])

    if not work_dir or not meta_path or not embed_dir:
        print(f"[{name}] Missing critical config paths. Skipping.")
        return
        
    temp_json_dir = os.path.join(work_dir, "temp_incoming_json")
    temp_parquet_dir = os.path.join(work_dir, "temp_incoming_parquet")
    
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STATE & FILE RECOVERY LOGIC
    # ---------------------------------------------------------
    
    # 1. Check if we need to force a reset because the master file is missing
    if override_start is not None:
        start_date = override_start
        print(f"[{name}] Using --from-date override: {start_date.strftime('%Y-%m-%d')}")
    elif not os.path.exists(meta_path):
        print(f"[{name}] ⚠️ Master Parquet file missing at {meta_path}.")
        print(f"[{name}] Resetting fetch state to 2013-01-01 to ensure full download.")
        update_state(work_dir, source['state_filename'], "2013-01-01")
        start_date = datetime(2013, 1, 1)
    else:
        last_date_str = get_state(work_dir, source['state_filename'])
        start_date = datetime.strptime(last_date_str, "%Y-%m-%d")

    # 2. Check if we need to regenerate NPY files (Parquet exists, NPY missing)
    if os.path.exists(meta_path) and not os.path.exists(embed_path):
        print(f"[{name}] ⚠️ Parquet file found but Embeddings (.npy) missing.")
        print(f"[{name}] Regenerating embeddings from existing data...")
        
        try:
            df = pd.read_parquet(meta_path)
            texts = build_input_texts(df)
            print(f"[{name}] Generating embeddings for {len(texts):,} records...")
            
            embeddings = generate_embeddings_batched(model, texts, desc=f"Regenerating {name}")
            np.save(embed_path, embeddings)
            print(f"[{name}] ✅ Embeddings recreated and saved to {embed_path}.")
        except Exception as e:
            print(f"[{name}] CRITICAL ERROR regenerating embeddings: {e}")
            return # Stop here if we can't fix the files

    # ---------------------------------------------------------
    # NORMAL FETCH ROUTINE
    # ---------------------------------------------------------
    end_date = datetime.today()
    
    if start_date.date() >= end_date.date():
        print(f"[{name}] Up to date (Last fetch: {start_date.strftime('%Y-%m-%d')}). Checking integrity only...")
        check_database_integrity(meta_path, embed_path, model)
        return

    print(f"\n[{name}] Fetching updates: {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    
    # 2. Fetch Data
    os.makedirs(temp_json_dir, exist_ok=True)
    os.makedirs(temp_parquet_dir, exist_ok=True)
    
    current_date = start_date
    future_to_block = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        while current_date <= end_date:
            block_start = current_date
            block_end = min(current_date + relativedelta(months=1) - relativedelta(days=1), end_date)

            if block_start > block_end:
                break

            future = executor.submit(
                fetch_block, source['endpoint'], server, block_start, block_end, temp_json_dir
            )
            future_to_block[future] = (block_start, block_end)
            current_date += relativedelta(months=1)

        total_blocks = len(future_to_block)
        completed = 0
        failed_blocks = []
        for future in as_completed(future_to_block):
            bs, be = future_to_block[future]
            completed += 1
            try:
                ok = future.result()
            except Exception as exc:
                ok = False
                print(f"[{name}] Block {bs.date()} → {be.date()} raised: {exc}")
            status = "OK" if ok else "FAILED"
            print(f"[{name}] [{completed}/{total_blocks}] {bs.strftime('%Y-%m')} — {status}")
            if not ok:
                failed_blocks.append(bs)

    if failed_blocks:
        failed_blocks.sort()
        print(f"\n[{name}] ⚠️  {len(failed_blocks)} block(s) failed to fetch:")
        for fb in failed_blocks:
            print(f"         • {fb.strftime('%Y-%m-%d')}")

        # Persist the failure log so the user can inspect it later.
        gap_log_path = os.path.join(work_dir, "failed_blocks.log")
        with open(gap_log_path, "a") as gf:
            gf.write(f"\n--- {datetime.now().isoformat()} ---\n")
            for fb in failed_blocks:
                gf.write(f"{fb.strftime('%Y-%m-%d')}\n")
        print(f"[{name}] Failed block dates appended to: {gap_log_path}")
        print(f"[{name}] To backfill, re-run with: "
              f"--from-date {min(failed_blocks).strftime('%Y-%m-%d')} --source {server}")

        # Advance state only up to the day before the earliest failure so the
        # next run re-tries the missing months.
        safe_advance = min(failed_blocks) - relativedelta(days=1)
        if safe_advance <= start_date:
            print(f"[{name}] First block failed — state not advanced.")
            effective_end = None
        else:
            effective_end = safe_advance
            print(f"[{name}] State will advance to {effective_end.strftime('%Y-%m-%d')} (before first failure).")
    else:
        effective_end = end_date

    # 3. Convert & Load Incoming
    incoming_files = convert_json_to_parquet(temp_json_dir, temp_parquet_dir)

    if not incoming_files:
        print(f"[{name}] No new data found/converted.")
        if effective_end:
            update_state(work_dir, source['state_filename'], effective_end.strftime("%Y-%m-%d"))
        # Keep temp dirs if blocks failed so the next run can skip already-fetched ones.
        if not failed_blocks:
            shutil.rmtree(temp_json_dir, ignore_errors=True)
            shutil.rmtree(temp_parquet_dir, ignore_errors=True)
        return

    # 4. Load Master Database
    if os.path.exists(meta_path) and os.path.exists(embed_path):
        existing_df = pd.read_parquet(meta_path)
        existing_embeddings = np.load(embed_path)

        # Heal a mismatch caused by a previous crash mid-save: the parquet may
        # have more rows than the .npy if embedding was interrupted.
        if len(existing_embeddings) < len(existing_df):
            missing_count = len(existing_df) - len(existing_embeddings)
            print(f"[{name}] ⚠️  Embedding/parquet mismatch: "
                  f"{len(existing_embeddings):,} embeddings vs {len(existing_df):,} rows. "
                  f"Generating {missing_count:,} missing embeddings…")
            missing_df = existing_df.iloc[len(existing_embeddings):]
            missing_texts = build_input_texts(missing_df)
            missing_embs = generate_embeddings_batched(model, missing_texts,
                                                       desc=f"Healing {name}")
            existing_embeddings = np.vstack([existing_embeddings, missing_embs])
            np.save(embed_path, existing_embeddings)
            print(f"[{name}] ✅ Embeddings healed and saved.")

        # Prepare for deduplication
        existing_df['signature'] = existing_df['title'].fillna('') + existing_df['abstract'].fillna('')
        seen_signatures = set(existing_df['signature'].unique())
        print(f"[{name}] Master DB: {len(existing_df):,} records loaded.")
    else:
        existing_df = pd.DataFrame()
        existing_embeddings = None
        seen_signatures = set()
        print(f"[{name}] Starting fresh (or files still missing).")

    # 5. Load Incoming Data & Deduplicate
    incoming_dfs = [pd.read_parquet(f) for f in incoming_files]
    raw_df = pd.concat(incoming_dfs, ignore_index=True)
    
    for col in ['title', 'abstract']:
        if col not in raw_df.columns: raw_df[col] = ""

    raw_df['signature'] = raw_df['title'].fillna('') + raw_df['abstract'].fillna('')
    new_df = raw_df[~raw_df['signature'].isin(seen_signatures)].copy()
    
    # 6. Embed New Data
    if len(new_df) > 0:
        print(f"[{name}] Embedding {len(new_df):,} new unique records...")
        
        new_texts = build_input_texts(new_df)
        new_binary_embeddings = generate_embeddings_batched(model, new_texts, desc=f"Embedding {name}")

        # 7. Merge & Save
        new_df.drop(columns=['signature'], inplace=True)
        if 'signature' in existing_df.columns: existing_df.drop(columns=['signature'], inplace=True)
        
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        if existing_embeddings is not None:
            final_embeddings = np.vstack([existing_embeddings, new_binary_embeddings])
        else:
            final_embeddings = new_binary_embeddings
            
        print(f"[{name}] Saving Master DB with {len(final_df):,} records...")
        # Small row groups: the search app reads a few rows per query, and a
        # single 357k-row group made every bioRxiv fetch decode the whole file
        # (1.3 s -> 0.15 s per fetch with 2,000-row groups).
        final_df.to_parquet(meta_path, row_group_size=2000, write_page_index=True)
        np.save(embed_path, final_embeddings)
    else:
        print(f"[{name}] All fetched data was duplicate.")

    # 8. Cleanup & Update Clock
    # Only wipe temp dirs when all blocks succeeded; on partial failure keep
    # the JSON files so the next run can skip already-fetched months.
    if not failed_blocks:
        print(f"[{name}] Cleaning up temp files...")
        shutil.rmtree(temp_json_dir, ignore_errors=True)
        shutil.rmtree(temp_parquet_dir, ignore_errors=True)
    else:
        print(f"[{name}] Keeping temp files for {len(failed_blocks)} failed block(s) — "
              "re-run to fetch missing months.")

    if effective_end:
        update_state(work_dir, source['state_filename'], effective_end.strftime("%Y-%m-%d"))

    # 9. Integrity Check
    check_database_integrity(meta_path, embed_path, model)

# -----------------------------------------------------------------------------
# 5. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update BioRxiv/MedRxiv database.")
    parser.add_argument(
        "--from-date",
        metavar="YYYY-MM-DD",
        help=(
            "Override the saved fetch state and re-fetch from this date. "
            "Useful to backfill months that silently failed in previous runs. "
            "Deduplication prevents re-adding records already in the database."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["biorxiv", "medrxiv", "both"],
        default="both",
        help="Which source to update (default: both).",
    )
    args = parser.parse_args()

    override_start: datetime | None = None
    if args.from_date:
        try:
            override_start = datetime.strptime(args.from_date, "%Y-%m-%d")
            print(f"⚠️  --from-date override: will fetch from {args.from_date} "
                  "regardless of saved state.")
        except ValueError:
            print(f"Invalid --from-date '{args.from_date}'. Expected YYYY-MM-DD.")
            exit(1)

    active_sources = [s for s in SOURCES
                      if args.source == "both" or s["server"] == args.source]

    print(f"--- Loading Model {MODEL_ID} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = SentenceTransformer(
            MODEL_ID,
            device=device,
            model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "sdpa"},
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    for source in active_sources:
        try:
            process_source(source, model, override_start=override_start)
        except Exception as e:
            print(f"CRITICAL ERROR processing {source['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll tasks completed.")