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

@retry_on_exception(requests.exceptions.ConnectionError)
def fetch_block(endpoint, server, block_start, block_end, save_directory):
    if server == "biorxiv":
        base_url = f"https://api.biorxiv.org/{endpoint}/{server}/"
    else:
        base_url = f"https://api.medrxiv.org/{endpoint}/{server}/"
        
    block_interval = f"{block_start.strftime('%Y-%m-%d')}/{block_end.strftime('%Y-%m-%d')}"
    block_data = []
    cursor = 0
    continue_fetching = True

    print(f"Fetching {server} block {block_interval}...")
    while continue_fetching:
        url = f"{base_url}{block_interval}/{cursor}/json"
        try:
            response = requests.get(url, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching {url}: {e}")
            break

        if response.status_code != 200:
            print(f"Failed {block_interval} at cursor {cursor}. Status: {response.status_code}")
            break

        try:
            data = response.json()
        except ValueError:
            print(f"Invalid JSON response from {url}")
            break

        fetched_count = len(data.get('collection', []))
        if fetched_count > 0:
            block_data.extend(data['collection'])
            cursor += fetched_count
        else:
            continue_fetching = False

    if block_data:
        save_data_block_json(block_data, block_start, block_end, endpoint, save_directory)
        return True
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

def process_source(source, model):
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
    if not os.path.exists(meta_path):
        print(f"[{name}] ⚠️ Master Parquet file missing at {meta_path}.")
        print(f"[{name}] Resetting fetch state to 2013-01-01 to ensure full download.")
        update_state(work_dir, source['state_filename'], "2013-01-01")
        start_date = datetime(2013, 1, 1)
    else:
        # Load state normally
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
    tasks = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        while current_date <= end_date:
            block_start = current_date
            block_end = min(current_date + relativedelta(months=1) - relativedelta(days=1), end_date)
            
            if block_start > block_end:
                break
                
            tasks.append(executor.submit(
                fetch_block, source['endpoint'], server, block_start, block_end, temp_json_dir
            ))
            current_date += relativedelta(months=1)
            
        for future in as_completed(tasks):
            future.result() 

    # 3. Convert & Load Incoming
    incoming_files = convert_json_to_parquet(temp_json_dir, temp_parquet_dir)
    
    if not incoming_files:
        print(f"[{name}] No new data found/converted.")
        update_state(work_dir, source['state_filename'], end_date.strftime("%Y-%m-%d"))
        shutil.rmtree(temp_json_dir, ignore_errors=True)
        shutil.rmtree(temp_parquet_dir, ignore_errors=True)
        return

    # 4. Load Master Database
    if os.path.exists(meta_path) and os.path.exists(embed_path):
        existing_df = pd.read_parquet(meta_path)
        existing_embeddings = np.load(embed_path)
        
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
        final_df.to_parquet(meta_path)
        np.save(embed_path, final_embeddings)
    else:
        print(f"[{name}] All fetched data was duplicate.")

    # 8. Cleanup & Update Clock
    print(f"[{name}] Cleaning up temp files...")
    shutil.rmtree(temp_json_dir, ignore_errors=True)
    shutil.rmtree(temp_parquet_dir, ignore_errors=True)
    
    update_state(work_dir, source['state_filename'], end_date.strftime("%Y-%m-%d"))

    # 9. Integrity Check
    check_database_integrity(meta_path, embed_path, model)

# -----------------------------------------------------------------------------
# 5. MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    print(f"--- Loading Model {MODEL_ID} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = SentenceTransformer(
            MODEL_ID, 
            device=device,
            model_kwargs={"torch_dtype": torch.float16, "attn_implementation": "sdpa"},
            trust_remote_code=True
        )
        # Optional: Compile for B200/RTX5080 speedup
        # try:
        #    model = torch.compile(model)
        # except: pass
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    for source in SOURCES:
        try:
            process_source(source, model)
        except Exception as e:
            print(f"CRITICAL ERROR processing {source['name']}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll tasks completed.")