#!/usr/bin/env python3
import os
import re
import json
import sys
import time
import uuid
import glob
import gc
import queue
import socket
import errno
import subprocess
import threading
import multiprocessing
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from unidecode import unidecode
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Final destination folders
DEST_DF_FOLDER = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/arxiv_df'          # Where parquet chunks live
DEST_EMBED_FOLDER = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/arxiv_embed'    # Where embeddings live

# Coordination
COORDINATION_DIR = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/arxiv_temp/coordination'
TASK_QUEUE_FILE = os.path.join(COORDINATION_DIR, 'task_queue.json')
MACHINE_STATUS_DIR = os.path.join(COORDINATION_DIR, 'machine_status')
GPU_LOCK_FOLDER = os.path.join('/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/arxiv_temp/gpu_locks', socket.gethostname())

# Kaggle / Inputs
KAGGLE_DATASET_DIR = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/arxiv_temp/kaggle_arxiv'
KAGGLE_JSON_FILE = os.path.join(KAGGLE_DATASET_DIR, 'arxiv-metadata-oai-snapshot.json')

# Model
BATCH_SIZE = 256
MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"
PREFETCH_SIZE = 2
SAVE_INTERVAL = 1
LOCK_TIMEOUT = 60 * 25

# Chunking Settings
CHUNK_SIZE_ROWS = 25000  # Save a new parquet every 25k new papers found

# -----------------------------------------------------------------------------
# HARDWARE & SETUP
# -----------------------------------------------------------------------------
HOSTNAME = socket.gethostname()
MACHINE_ID = str(uuid.uuid4())[:8]
try:
    AVAILABLE_GPUS = torch.cuda.device_count()
except:
    AVAILABLE_GPUS = 0

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Check Kaggle
def is_kaggle_installed():
    try:
        subprocess.run(['kaggle', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except:
        return False

if not is_kaggle_installed():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = unidecode(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def robust_basename(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0]

def quantize_and_pack(embeddings):
    bits = (embeddings > 0)
    packed = np.packbits(bits, axis=1)
    return packed

def ensure_dirs():
    for directory in [DEST_DF_FOLDER, DEST_EMBED_FOLDER, COORDINATION_DIR,
                      GPU_LOCK_FOLDER, MACHINE_STATUS_DIR, KAGGLE_DATASET_DIR]:
        os.makedirs(directory, exist_ok=True)

# -----------------------------------------------------------------------------
# INCREMENTAL DATA PROCESSING (MEMORY OPTIMIZED)
# -----------------------------------------------------------------------------
def get_existing_ids(df_folder):
    """
    Scans all existing parquet files in the destination folder.
    Loads ONLY the 'id' column into a set to keep memory usage extremely low.
    """
    existing_ids = set()
    files = glob.glob(os.path.join(df_folder, "*.parquet"))
    
    if not files:
        return existing_ids

    print(f"Scanning {len(files)} existing parquet files for IDs...")
    
    # We can do this in parallel or sequential. Sequential is safer for RAM.
    for f in tqdm(files, desc="Indexing existing IDs"):
        try:
            # Only load the 'id' column
            df = pd.read_parquet(f, columns=['id'])
            existing_ids.update(df['id'].astype(str).values)
        except Exception as e:
            print(f"Warning: Could not read IDs from {f}: {e}")
            
    print(f"Found {len(existing_ids):,} existing documents in database.")
    return existing_ids

def download_kaggle_dataset():
    """Force download/update of the Kaggle dataset."""
    print("Checking for updates from Kaggle...")
    # --force ensures we get the newest version if it changed
    command = f"kaggle datasets download -d Cornell-University/arxiv -p {KAGGLE_DATASET_DIR} --unzip --force"
    try:
        subprocess.run(command, shell=True, check=True)
        print("Kaggle download/update complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")

def incremental_process_json(json_file, output_folder, existing_ids):
    """
    Streams the massive JSON file. 
    Filters out IDs we already have.
    Saves NEW rows into small parquet chunks.
    NEVER loads the whole dataset into RAM.
    """
    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        return

    print("Starting incremental stream processing...")
    
    new_rows = []
    chunks_created = 0
    new_docs_count = 0
    
    # Use a unique run ID so we don't overwrite files if script runs twice in one day
    run_id = int(time.time())

    with open(json_file, 'r') as f:
        # TQDM on a generator if we don't know total lines, or just run it
        for line in tqdm(f, desc="Scanning JSON stream"):
            if not line: continue
            
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = str(doc.get('id', ''))
            
            # THE CORE LOGIC: Filter against the set
            if doc_id in existing_ids:
                continue
            
            # If we are here, it's a NEW paper
            # Extract and clean only what we need
            processed_doc = {
                'id': doc_id,
                'title': clean_text(doc.get('title', '')),
                'abstract': clean_text(doc.get('abstract', '')),
                'authors': doc.get('authors', ''),
                'date': doc.get('update_date', ''), # Arxiv usually has update_date
                'categories': doc.get('categories', '')
            }
            
            new_rows.append(processed_doc)
            new_docs_count += 1

            # Flush to disk if buffer is full
            if len(new_rows) >= CHUNK_SIZE_ROWS:
                save_chunk(new_rows, output_folder, run_id, chunks_created)
                chunks_created += 1
                new_rows = [] # Clear memory
                gc.collect()

    # Save remaining rows
    if new_rows:
        save_chunk(new_rows, output_folder, run_id, chunks_created)
        chunks_created += 1

    print(f"\nIncremental processing done.")
    print(f"Found {new_docs_count:,} NEW papers.")
    print(f"Created {chunks_created} new parquet files.")

def save_chunk(rows, folder, run_id, chunk_num):
    """Helper to save a list of dicts as an optimized parquet."""
    df = pd.DataFrame(rows)
    
    # Compression settings
    compression_dict = {
        'abstract': 'ZSTD', 'title': 'ZSTD', 'authors': 'ZSTD', 
        'id': 'ZSTD', 'categories': 'ZSTD'
    }
    valid_comp = {k: v for k, v in compression_dict.items() if k in df.columns}
    
    filename = f"arxiv_update_{run_id}_chunk_{chunk_num}.parquet"
    path = os.path.join(folder, filename)
    
    df.to_parquet(
        path,
        engine='pyarrow',
        compression=valid_comp,
        index=False
    )
    print(f"Saved new chunk: {filename}")

# -----------------------------------------------------------------------------
# COORDINATION & LOCKING (Kept from original, slightly cleaned)
# -----------------------------------------------------------------------------
class SimpleLock:
    def __init__(self, lock_path):
        self.lock_path = lock_path
        self.fd = None

    def acquire(self, timeout=LOCK_TIMEOUT):
        start_time = time.time()
        while True:
            try:
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                lock_info = f"{MACHINE_ID}:{HOSTNAME}:{os.getpid()}:{int(time.time())}"
                os.write(self.fd, lock_info.encode())
                return True
            except FileExistsError:
                if time.time() - start_time > timeout:
                    return False
                time.sleep(0.5)

    def release(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if os.path.exists(self.lock_path):
            try:
                os.remove(self.lock_path)
            except: pass

def initialize_task_queue(file_paths):
    lock_path = f"{TASK_QUEUE_FILE}.lock"
    queue_lock = SimpleLock(lock_path)
    pending_count = 0
    if queue_lock.acquire(timeout=30):
        try:
            if os.path.exists(TASK_QUEUE_FILE):
                with open(TASK_QUEUE_FILE, 'r') as f:
                    queue_data = json.load(f)
            else:
                queue_data = {'pending': [], 'in_progress': {}, 'completed': [], 'failed': {}}
            
            # Filter files: Add to queue ONLY if .npy doesn't exist
            current_known = set(queue_data['pending']) | set(queue_data['completed']) | set(queue_data['in_progress'].keys())
            
            new_files = []
            for f in file_paths:
                # Check if embedding already exists
                emb_path = os.path.join(DEST_EMBED_FOLDER, f"{robust_basename(f)}.npy")
                if f not in current_known and not os.path.exists(emb_path):
                    new_files.append(f)
            
            if new_files:
                queue_data['pending'].extend(new_files)
                print(f"Added {len(new_files)} new files to the task queue")
            
            with open(TASK_QUEUE_FILE, 'w') as f:
                json.dump(queue_data, f)
            pending_count = len(queue_data['pending'])
        finally:
            queue_lock.release()
    return pending_count

def claim_next_batch(batch_size, gpu_id):
    # (Same logic as original, claims files from JSON queue)
    claimed_files = []
    lock_path = f"{TASK_QUEUE_FILE}.lock"
    queue_lock = SimpleLock(lock_path)
    if queue_lock.acquire():
        try:
            if not os.path.exists(TASK_QUEUE_FILE): return []
            with open(TASK_QUEUE_FILE, 'r') as f:
                data = json.load(f)
            if not data['pending']: return []
            
            to_claim = data['pending'][:batch_size]
            data['pending'] = data['pending'][batch_size:]
            
            for fp in to_claim:
                data['in_progress'][fp] = {'machine': HOSTNAME, 'gpu': gpu_id, 'time': int(time.time())}
                claimed_files.append(fp)
                
            with open(TASK_QUEUE_FILE, 'w') as f:
                json.dump(data, f)
        finally:
            queue_lock.release()
    return claimed_files

def mark_file_completed(file_path, success, error_msg=None):
    lock_path = f"{TASK_QUEUE_FILE}.lock"
    queue_lock = SimpleLock(lock_path)
    if queue_lock.acquire():
        try:
            with open(TASK_QUEUE_FILE, 'r') as f:
                data = json.load(f)
            if file_path in data['in_progress']:
                del data['in_progress'][file_path]
            if success:
                data['completed'].append(file_path)
            else:
                data['failed'][file_path] = str(error_msg)
            with open(TASK_QUEUE_FILE, 'w') as f:
                json.dump(data, f)
        finally:
            queue_lock.release()

# -----------------------------------------------------------------------------
# EMBEDDING WORKER (GPU)
# -----------------------------------------------------------------------------
def process_embeddings_worker(gpu_id):
    """The GPU worker that processes the queue."""
    # Affinity
    try:
        import psutil
        p = psutil.Process()
        cores = psutil.cpu_count(logical=False) or 1
        per_gpu = max(1, cores // (AVAILABLE_GPUS or 1))
        p.cpu_affinity(list(range(gpu_id * per_gpu, (gpu_id + 1) * per_gpu)))
    except: pass

    # Load Model
    print(f"[GPU {gpu_id}] Loading Model...")
    model = SentenceTransformer(
        MODEL_ID, 
        device="cuda", 
        model_kwargs={"dtype": torch.float16, "attn_implementation": "sdpa"},
        trust_remote_code=True
    )
    #model.max_seq_length = 512

    # Loop Queue
    while True:
        files = claim_next_batch(1, gpu_id) # Process 1 file at a time
        if not files:
            break
        
        file_path = files[0]
        file_stem = robust_basename(file_path)
        out_path = os.path.join(DEST_EMBED_FOLDER, f"{file_stem}.npy")
        
        try:
            df = pd.read_parquet(file_path, columns=['title', 'abstract'])
            queries = (df['title'].fillna('') + ". " + df['abstract'].fillna('')).tolist()
            
            if queries:
                emb = model.encode(queries, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
                binary = quantize_and_pack(emb)
                np.save(out_path, binary)
                print(f"[GPU {gpu_id}] Saved {len(binary)} docs -> {file_stem}")
            
            mark_file_completed(file_path, True)
            del df, queries, emb, binary
            gc.collect()
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Failed {file_stem}: {e}")
            mark_file_completed(file_path, False, str(e))

# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------
def main():
    overall_start = time.time()
    ensure_dirs()
    
    # 1. Update Raw Data (Manual/Kaggle)
    # ----------------------------------
    download_kaggle_dataset()
    
    # 2. Incremental Parsing
    # ----------------------
    # Instead of loading everything, we load IDs of what we already have
    existing_ids = get_existing_ids(DEST_DF_FOLDER)
    
    # Process the raw JSON, but only save rows that aren't in existing_ids
    incremental_process_json(KAGGLE_JSON_FILE, DEST_DF_FOLDER, existing_ids)
    
    # Clear huge ID set from memory now that parsing is done
    del existing_ids
    gc.collect()
    
    # 3. Queue Management
    # -------------------
    # Scan folder for all parquet files (both old and new)
    all_parquets = sorted(glob.glob(os.path.join(DEST_DF_FOLDER, '*.parquet')))
    if not all_parquets:
        print("No data found to process.")
        return

    # Add to queue only if .npy is missing
    pending = initialize_task_queue(all_parquets)
    print(f"Embedding Queue: {pending} files pending.")
    
    if pending == 0:
        print("All up to date.")
        return

    # 4. Distributed/Parallel Embedding
    # ---------------------------------
    if AVAILABLE_GPUS > 0:
        print(f"Spinning up workers for {AVAILABLE_GPUS} GPUs...")
        processes = []
        for i in range(AVAILABLE_GPUS):
            p = multiprocessing.Process(target=process_embeddings_worker, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        print("No GPUs found. Add CPU logic if needed (slow).")

    # remove finally the json arxiv file
    if os.path.exists(KAGGLE_JSON_FILE):
        os.remove(KAGGLE_JSON_FILE)
        print("Removed temporary Kaggle JSON file.")
    

    print(f"Total time: {time.time() - overall_start:.2f}s")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()