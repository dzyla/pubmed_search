#!/usr/bin/env python3
import os
import re
import json
import time
import uuid
import glob
import gc
import queue
import signal
import socket
import errno
import shutil
import subprocess
import threading
import multiprocessing
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from unidecode import unidecode
from sentence_transformers import SentenceTransformer

###############################
# Global configuration and directories
###############################
# Final destination folders (deduped chunks and embeddings)
DEST_DF_FOLDER = 'arxiv_df'          # Final deduped and chunked Parquet files
DEST_EMBED_FOLDER = 'arxiv_embed'
# Temporary folders for coordination and intermediate files
COORDINATION_DIR = 'coordination'
TASK_QUEUE_FILE = os.path.join(COORDINATION_DIR, 'task_queue.json')
MACHINE_STATUS_DIR = os.path.join(COORDINATION_DIR, 'machine_status')
GPU_LOCK_FOLDER = os.path.join('gpu_locks', socket.gethostname())

# Other configuration parameters
BATCH_SIZE = 64
PREFETCH_SIZE = 1
SAVE_INTERVAL = 1
LOCK_TIMEOUT = 60 * 25  # seconds

# Kaggle dataset and temporary directories
kaggle_dataset_dir = 'kaggle_arxiv'
temp_jsonl_dir = 'temp_json'           # Directory for JSONL files extraction
TEMP_DF_FOLDER = 'temp_arxiv_parquet'   # Temporary Parquet files (pre-deduplication)

# Files for combined deduplicated DataFrame and final embeddings (if needed)
COMBINED_DF_FILE = os.path.join(DEST_DF_FOLDER, 'combined_deduped.parquet')
FINAL_EMBED_FILE = os.path.join(DEST_EMBED_FOLDER, 'combined_deduped.npy')

# Load YAML configuration (if available)
CONFIG_FILE = "config_mss_new_pubmed.yaml"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {}
arxiv_conf = config.get("arxiv_config", {})

# Machine-specific information for distributed coordination
HOSTNAME = socket.gethostname()
MACHINE_ID = str(uuid.uuid4())[:8]
try:
    AVAILABLE_GPUS = torch.cuda.device_count()
    print(f"Detected {AVAILABLE_GPUS} available GPUs on {HOSTNAME}")
except Exception as e:
    AVAILABLE_GPUS = 0
    print(f"No GPUs detected on {HOSTNAME}")

###############################
# Utility functions (cleaning, basename)
###############################
def clean_text(text):
    """
    Clean text by converting to string, removing accents, replacing newlines with spaces,
    and collapsing extra spaces.
    """
    if not isinstance(text, str):
        text = str(text)
    text = unidecode(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def robust_basename(filepath: str) -> str:
    """Return the file name without its extension."""
    return os.path.splitext(os.path.basename(filepath))[0]

###############################
# Optimized Parquet Saving Function
###############################
def save_optimized_parquet(df, parquet_file, row_group_size=10000):
    """
    Save DataFrame to an optimized Parquet file for both storage efficiency and fast random access.
    
    Args:
        df: DataFrame to save
        parquet_file: Output file path
        row_group_size: Number of rows per row group (affects random access performance)
    """
    os.makedirs(os.path.dirname(parquet_file), exist_ok=True)
    # Dictionary specifying compression by column (adjust as needed)
    compression_dict = {
        'abstract': 'ZSTD',
        'title': 'ZSTD',
        'authors': 'ZSTD',
        'mesh_terms': 'ZSTD',
        'keywords': 'ZSTD',
        'chemicals': 'ZSTD',
        'journal': 'SNAPPY',
        'type': 'SNAPPY',
        'date': 'SNAPPY',
        'version': 'SNAPPY',
        'doi': 'NONE',
        'pmid': 'NONE'
    }
    # Sort DataFrame by date if available (for better compression)
    if 'date' in df.columns and not df.empty:
        df = df.sort_values('date')
    df.to_parquet(
        parquet_file,
        index=False,
        engine='pyarrow',
        compression=compression_dict,
        row_group_size=row_group_size,
        use_dictionary=True,
        data_page_size=8*1024*1024,
        write_statistics=True,
        version='2.6'
    )
    print(f"Optimized parquet file saved at {parquet_file}")

###############################
# Coordination and Distributed Processing Functions
###############################
def ensure_dirs():
    """Create all necessary directories for coordination and data storage."""
    for directory in [DEST_DF_FOLDER, DEST_EMBED_FOLDER, COORDINATION_DIR,
                      GPU_LOCK_FOLDER, MACHINE_STATUS_DIR, temp_jsonl_dir, TEMP_DF_FOLDER]:
        os.makedirs(directory, exist_ok=True)

def register_machine():
    """Register this machine in the coordination system and start heartbeat."""
    machine_info = {
        'hostname': HOSTNAME,
        'machine_id': MACHINE_ID,
        'available_gpus': AVAILABLE_GPUS,
        'start_time': int(time.time()),
        'last_heartbeat': int(time.time()),
        'status': 'active'
    }
    machine_file = os.path.join(MACHINE_STATUS_DIR, f"{HOSTNAME}_{MACHINE_ID}.json")
    with open(machine_file, 'w') as f:
        json.dump(machine_info, f)
    heartbeat_thread = threading.Thread(target=machine_heartbeat, args=(machine_file,))
    heartbeat_thread.daemon = True
    heartbeat_thread.start()
    return machine_file, heartbeat_thread

def machine_heartbeat(machine_file):
    """Periodically update the machine's heartbeat timestamp."""
    while True:
        try:
            with open(machine_file, 'r') as f:
                machine_info = json.load(f)
            machine_info['last_heartbeat'] = int(time.time())
            with open(machine_file, 'w') as f:
                json.dump(machine_info, f)
        except Exception as e:
            print(f"Heartbeat update failed: {e}")
        time.sleep(30)

def get_active_machines():
    """Return a list of currently active machines based on heartbeat."""
    active_machines = []
    current_time = int(time.time())
    heartbeat_timeout = current_time - 90
    for machine_file in glob.glob(os.path.join(MACHINE_STATUS_DIR, '*.json')):
        try:
            with open(machine_file, 'r') as f:
                machine_info = json.load(f)
            if machine_info['last_heartbeat'] > heartbeat_timeout:
                active_machines.append(machine_info)
            else:
                machine_info['status'] = 'inactive'
                with open(machine_file, 'w') as f:
                    json.dump(machine_info, f)
        except Exception:
            pass
    return active_machines

def cleanup_stale_locks():
    """Remove locks from inactive machines."""
    active_machines = get_active_machines()
    active_ids = [m['machine_id'] for m in active_machines]
    for lock_file in glob.glob(os.path.join(DEST_EMBED_FOLDER, '*.lock')):
        try:
            with open(lock_file, 'r') as f:
                lock_info = f.read()
            if ':' in lock_info:
                machine_id = lock_info.split(':')[0]
                if machine_id not in active_ids:
                    print(f"Removing stale lock: {lock_file}")
                    os.remove(lock_file)
        except Exception as e:
            print(f"Error checking lock file {lock_file}: {e}")

class SimpleLock:
    """A simple file lock for distributed coordination."""
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
                try:
                    with open(self.lock_path, 'r') as f:
                        lock_info = f.read()
                    if ':' in lock_info:
                        parts = lock_info.split(':')
                        if len(parts) >= 4:
                            lock_time = int(parts[3])
                            if time.time() - lock_time > 300:
                                print(f"Removing stale lock: {self.lock_path}")
                                os.remove(self.lock_path)
                                continue
                except Exception:
                    pass
                time.sleep(0.5)

    def release(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if os.path.exists(self.lock_path):
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass

def acquire_file_lock(file_stem):
    """Acquire a file lock for a given file stem."""
    os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
    lock_path = os.path.join(DEST_EMBED_FOLDER, f"{file_stem}.lock")
    lock = SimpleLock(lock_path)
    if lock.acquire():
        return lock
    return None

def release_file_lock(lock):
    """Release a previously acquired file lock."""
    if lock:
        lock.release()

def initialize_task_queue(file_paths):
    """
    Initialize the global task queue with all files (here, our parquet chunks)
    that need processing.
    """
    lock_path = f"{TASK_QUEUE_FILE}.lock"
    queue_lock = SimpleLock(lock_path)
    if queue_lock.acquire(timeout=30):
        try:
            if os.path.exists(TASK_QUEUE_FILE):
                with open(TASK_QUEUE_FILE, 'r') as f:
                    queue_data = json.load(f)
            else:
                queue_data = {'pending': [], 'in_progress': {}, 'completed': [], 'failed': {}}
            current_files = set(queue_data['pending']) | set(queue_data['completed']) | set(queue_data['in_progress'].keys())
            new_files = [f for f in file_paths if f not in current_files and 
                         not os.path.exists(os.path.join(DEST_EMBED_FOLDER, f"{robust_basename(f)}.npy"))]
            if new_files:
                queue_data['pending'].extend(new_files)
                print(f"Added {len(new_files)} new files to the task queue")
            with open(TASK_QUEUE_FILE, 'w') as f:
                json.dump(queue_data, f)
            pending_count = len(queue_data['pending'])
        finally:
            queue_lock.release()
        return pending_count
    else:
        print("Could not acquire lock for task queue initialization")
        return 0

def claim_next_batch(batch_size, gpu_id):
    """Claim the next batch of files from the shared task queue."""
    claimed_files = []
    lock_path = f"{TASK_QUEUE_FILE}.lock"
    queue_lock = SimpleLock(lock_path)
    if queue_lock.acquire():
        try:
            if not os.path.exists(TASK_QUEUE_FILE):
                return claimed_files
            with open(TASK_QUEUE_FILE, 'r') as f:
                queue_data = json.load(f)
            if not queue_data['pending']:
                return claimed_files
            to_claim = queue_data['pending'][:batch_size]
            queue_data['pending'] = queue_data['pending'][batch_size:]
            claim_time = int(time.time())
            for file_path in to_claim:
                queue_data['in_progress'][file_path] = {
                    'machine_id': MACHINE_ID,
                    'hostname': HOSTNAME,
                    'gpu_id': gpu_id,
                    'claimed_at': claim_time
                }
                claimed_files.append(file_path)
            with open(TASK_QUEUE_FILE, 'w') as f:
                json.dump(queue_data, f)
        finally:
            queue_lock.release()
    return claimed_files

def mark_file_completed(file_path, success, error_msg=None):
    """Mark a file as completed or failed in the task queue."""
    lock_path = f"{TASK_QUEUE_FILE}.lock"
    queue_lock = SimpleLock(lock_path)
    if queue_lock.acquire():
        try:
            if not os.path.exists(TASK_QUEUE_FILE):
                return
            with open(TASK_QUEUE_FILE, 'r') as f:
                queue_data = json.load(f)
            if file_path in queue_data['in_progress']:
                del queue_data['in_progress'][file_path]
            if success:
                queue_data['completed'].append(file_path)
            else:
                queue_data['failed'][file_path] = {
                    'error': error_msg,
                    'machine_id': MACHINE_ID,
                    'hostname': HOSTNAME,
                    'time': int(time.time())
                }
            with open(TASK_QUEUE_FILE, 'w') as f:
                json.dump(queue_data, f)
        finally:
            queue_lock.release()

def get_queue_status():
    """Return current status of the task queue."""
    status = {'pending': 0, 'in_progress': 0, 'completed': 0, 'failed': 0}
    try:
        lock_path = f"{TASK_QUEUE_FILE}.lock"
        queue_lock = SimpleLock(lock_path)
        if queue_lock.acquire(timeout=1):
            try:
                if os.path.exists(TASK_QUEUE_FILE):
                    with open(TASK_QUEUE_FILE, 'r') as f:
                        queue_data = json.load(f)
                    status['pending'] = len(queue_data['pending'])
                    status['in_progress'] = len(queue_data['in_progress'])
                    status['completed'] = len(queue_data['completed'])
                    status['failed'] = len(queue_data['failed'])
            finally:
                queue_lock.release()
    except Exception as e:
        print(f"Error reading queue status: {e}")
    return status

def pid_exists(pid):
    """Return True if the given process id exists."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            return False
        elif err.errno == errno.EPERM:
            return True
        else:
            raise
    else:
        return True

def acquire_gpu_lock(gpu_id):
    """Acquire a lock file for a given GPU with stale detection."""
    os.makedirs(GPU_LOCK_FOLDER, exist_ok=True)
    lock_file = os.path.join(GPU_LOCK_FOLDER, f"gpu_{gpu_id}.lock")
    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r") as f:
                lock_info = f.read().strip()
            if ":" in lock_info:
                parts = lock_info.split(":")
                if len(parts) >= 2:
                    pid = int(parts[0])
                    if not pid_exists(pid):
                        print(f"Removing stale GPU lock for GPU {gpu_id} (PID {pid} not found)")
                        os.remove(lock_file)
                    else:
                        print(f"GPU {gpu_id} is locked by active process {pid}. Skipping.")
                        return None
            elif lock_info.isdigit():
                pid = int(lock_info)
                if not pid_exists(pid):
                    print(f"Removing stale GPU lock for GPU {gpu_id} (PID {pid} not found)")
                    os.remove(lock_file)
                else:
                    print(f"GPU {gpu_id} is locked by active process {pid}. Skipping.")
                    return None
            else:
                print(f"GPU {gpu_id} is already locked. Skipping processing on this GPU.")
                return None
        except Exception:
            print(f"GPU {gpu_id} is already locked. Skipping processing on this GPU.")
            return None
    with open(lock_file, "w") as f:
        f.write(f"{os.getpid()}:{MACHINE_ID}:{int(time.time())}")
    return lock_file

def release_gpu_lock(lock_file):
    """Release the GPU lock by removing the lock file."""
    if lock_file and os.path.exists(lock_file):
        os.remove(lock_file)

def set_process_affinity(gpu_id):
    """Set CPU affinity for the current process (if supported) to reduce context switching."""
    try:
        import psutil
        process = psutil.Process()
        num_cores = psutil.cpu_count(logical=False)
        num_gpus = AVAILABLE_GPUS if AVAILABLE_GPUS > 0 else 1
        cores_per_gpu = max(1, num_cores // num_gpus)
        start_core = gpu_id * cores_per_gpu
        end_core = start_core + cores_per_gpu
        cpu_list = list(range(start_core, min(end_core, num_cores)))
        if cpu_list:
            process.cpu_affinity(cpu_list)
            print(f"GPU {gpu_id} worker using CPU cores: {cpu_list}")
    except Exception as e:
        print(f"Failed to set CPU affinity: {e}")

class DataPrefetcher:
    """
    Prefetch Parquet files and generate query strings (title+abstract)
    to reduce I/O wait during embedding computation.
    """
    def __init__(self, file_paths, max_prefetch=2):
        self.file_paths = file_paths
        self.max_prefetch = max_prefetch
        self.queue = queue.Queue(maxsize=max_prefetch)
        self.stop_event = threading.Event()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def _prefetch_worker(self):
        for file_path in self.file_paths:
            if self.stop_event.is_set():
                break
            try:
                df = pd.read_parquet(file_path)
                queries = [f"{(title if title else '').strip()} {(abstract if abstract else '').strip()}".strip() 
                           for title, abstract in zip(df['title'], df['abstract'])]
                self.queue.put((file_path, df, queries))
            except Exception as e:
                print(f"Error prefetching {file_path}: {e}")
                self.queue.put((file_path, None, None))
        self.queue.put((None, None, None))

    def next(self):
        return self.queue.get()

    def stop(self):
        self.stop_event.set()
        if self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1)

def process_embeddings_gpu(file_path, df, queries, model, progress_file):
    """
    Process a single parquet chunk file on GPU:
    load the DataFrame, compute embeddings for title+abstract, and save the embeddings.
    """
    file_stem = robust_basename(file_path)
    embeddings_path = os.path.join(DEST_EMBED_FOLDER, f"{file_stem}.npy")
    if os.path.exists(embeddings_path):
        print(f"Embeddings for {file_stem} already exist. Skipping.")
        mark_file_completed(file_path, True)
        return True
    lock = acquire_file_lock(file_stem)
    if lock is None:
        print(f"File {file_stem} is locked by another process. Skipping.")
        return False
    try:
        start_time = time.time()
        if not queries or len(queries) == 0:
            print(f"No valid queries found in {file_path}, skipping.")
            mark_file_completed(file_path, False, "No valid queries")
            return False
        print(f"Processing {file_stem} with {len(queries)} entries")
        query_embeddings = model.encode(
            queries,
            normalize_embeddings=True,
            precision='ubinary',
            batch_size=BATCH_SIZE,
            show_progress_bar=True
        )
        np.save(embeddings_path, query_embeddings)
        processing_time = time.time() - start_time
        items_per_sec = len(queries) / processing_time if processing_time > 0 else 0
        print(f"Saved {len(queries)} embeddings to {embeddings_path}")
        print(f"Time: {processing_time:.2f}s ({items_per_sec:.2f} items/sec)")
        with open(progress_file, 'a') as f:
            f.write(f"{file_stem},{len(queries)},{processing_time:.2f},{items_per_sec:.2f}\n")
        mark_file_completed(file_path, True)
        del query_embeddings
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing {file_path}: {error_msg}")
        mark_file_completed(file_path, False, error_msg)
        return False
    finally:
        release_file_lock(lock)

def process_embeddings_worker(gpu_id, batch_size=1):
    """
    Worker function for GPU embedding computation.
    Claims files from the shared task queue, prefetches data, and processes them.
    """
    lock_file = acquire_gpu_lock(gpu_id)
    if lock_file is None:
        return
    set_process_affinity(gpu_id)
    try:
        os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
        progress_file = os.path.join(DEST_EMBED_FOLDER, f"progress_{HOSTNAME}_gpu_{gpu_id}.csv")
        with open(progress_file, 'w') as f:
            f.write("file,num_entries,processing_time,items_per_second\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[GPU {gpu_id}] Device: {gpu_name}, Memory: {gpu_mem:.1f} GB")
        else:
            print(f"[GPU {gpu_id}] Warning: CUDA not available.")
        torch.backends.cudnn.benchmark = True
        print(f"[GPU {gpu_id}] Loading model...")
        start_time = time.time()
        model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
        model.to(device)
        try:
            import psutil
            num_cores = psutil.cpu_count(logical=False)
            cores_per_gpu = max(1, num_cores // AVAILABLE_GPUS) if AVAILABLE_GPUS > 0 else num_cores
            torch.set_num_threads(cores_per_gpu)
            print(f"[GPU {gpu_id}] Set torch thread count to {cores_per_gpu}")
        except:
            pass
        print(f"[GPU {gpu_id}] Model loaded in {time.time() - start_time:.2f}s")
        processed_count = 0
        total_time = 0
        total_entries = 0
        prefetcher = None
        while True:
            files_to_process = claim_next_batch(batch_size, gpu_id)
            if not files_to_process:
                print(f"[GPU {gpu_id}] No more files. Finishing worker.")
                break
            print(f"[GPU {gpu_id}] Claimed {len(files_to_process)} files")
            if prefetcher:
                prefetcher.stop()
            prefetcher = DataPrefetcher(files_to_process, max_prefetch=PREFETCH_SIZE)
            while True:
                file_path, df, queries = prefetcher.next()
                if file_path is None:
                    break
                try:
                    start = time.time()
                    success = process_embeddings_gpu(file_path, df, queries, model, progress_file)
                    duration = time.time() - start
                    if success:
                        processed_count += 1
                        total_time += duration
                        total_entries += len(queries) if queries else 0
                        if processed_count % SAVE_INTERVAL == 0:
                            status = get_queue_status()
                            print(f"[GPU {gpu_id}] Completed {processed_count} files. "
                                  f"Queue: {status['pending']} pending, {status['in_progress']} in progress, "
                                  f"{status['completed']} completed, {status['failed']} failed")
                except Exception as e:
                    print(f"[GPU {gpu_id}] Error processing {file_path}: {e}")
                    mark_file_completed(file_path, False, str(e))
                finally:
                    df = None
                    queries = None
            prefetcher.stop()
        if processed_count > 0 and total_time > 0:
            avg_time = total_time / processed_count
            items_per_sec = total_entries / total_time
            print(f"[GPU {gpu_id}] Completed {processed_count} files in {total_time:.2f}s "
                  f"(Avg {avg_time:.2f}s per file, {items_per_sec:.2f} items/sec)")
    except Exception as e:
        print(f"[GPU {gpu_id}] Worker error: {e}")
    finally:
        if 'prefetcher' in locals() and prefetcher:
            prefetcher.stop()
        if 'model' in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()
        release_gpu_lock(lock_file)
        print(f"[GPU {gpu_id}] Worker finished.")

def process_embeddings_cpu_arxiv(file, model):
    """
    CPU version: Process embeddings for a single Parquet file chunk on CPU.
    """
    file_stem = robust_basename(file)
    embeddings_path = os.path.join(DEST_EMBED_FOLDER, f"{file_stem}.npy")
    if os.path.exists(embeddings_path):
        print(f"Embeddings for {file_stem} already exist. Skipping.")
        mark_file_completed(file, True)
        return
    lock = acquire_file_lock(file_stem)
    if lock is None:
        print(f"File {file_stem} is locked by another process. Skipping.")
        return
    try:
        start_time = time.time()
        df = pd.read_parquet(file)
        queries = [f"{(title if title else '').strip()} {(abstract if abstract else '').strip()}".strip()
                   for title, abstract in zip(df['title'], df['abstract'])]
        print(f"Processing {file} with {len(queries)} entries (CPU)")
        query_embeddings = model.encode(
            queries,
            normalize_embeddings=True,
            precision='ubinary',
            batch_size=BATCH_SIZE,
            show_progress_bar=True
        )
        np.save(embeddings_path, query_embeddings)
        processing_time = time.time() - start_time
        print(f"Saved embeddings to {embeddings_path} in {processing_time:.2f}s")
        mark_file_completed(file, True)
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing {file}: {error_msg}")
        mark_file_completed(file, False, error_msg)
    finally:
        release_file_lock(lock)

def main_cpu_embeddings_arxiv(input_directory):
    """
    Process embeddings in CPU mode for arXiv parquet chunks.
    """
    ensure_dirs()
    machine_file, heartbeat_thread = register_machine()
    file_paths = sorted(glob.glob(os.path.join(input_directory, '*.parquet')))
    if not file_paths:
        print("No Parquet files found in the directory.")
        return
    num_pending = initialize_task_queue(file_paths)
    print(f"Task queue initialized with {num_pending} pending files")
    progress_watcher_stop = threading.Event()
    progress_watcher = threading.Thread(target=monitor_progress, args=(progress_watcher_stop,))
    progress_watcher.daemon = True
    progress_watcher.start()
    os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
    print("Loading model on CPU...")
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")
    start_time = time.time()
    processed_count = 0
    batch_size = 5
    while True:
        files_to_process = claim_next_batch(batch_size, -1)
        if not files_to_process:
            print("No more files to process. Finishing CPU processing.")
            break
        print(f"Claimed {len(files_to_process)} files for CPU processing")
        for file in tqdm(files_to_process, desc="CPU embeddings processing"):
            process_embeddings_cpu_arxiv(file, model)
            processed_count += 1
            if processed_count % SAVE_INTERVAL == 0:
                status = get_queue_status()
                print(f"Progress: completed {processed_count} files. "
                      f"Queue: {status['pending']} pending, {status['in_progress']} in progress, "
                      f"{status['completed']} completed, {status['failed']} failed")
    progress_watcher_stop.set()
    if progress_watcher.is_alive():
        progress_watcher.join(timeout=1)
    total_time = time.time() - start_time
    queue_status = get_queue_status()
    total_processed = queue_status['completed']
    print(f"CPU embedding generation completed. Processed {total_processed} files in {total_time:.2f}s")

def monitor_progress(stop_event):
    """Periodically print the status of the task queue and active machines."""
    try:
        while not stop_event.is_set():
            time.sleep(10)
            queue_status = get_queue_status()
            active_machines = get_active_machines()
            print("\n=== Cluster Status ===")
            print(f"Active machines: {len(active_machines)}")
            for machine in active_machines:
                print(f"  {machine['hostname']} (ID: {machine['machine_id']}): {machine['available_gpus']} GPUs")
            print("\n=== Queue Status ===")
            print(f"Pending: {queue_status['pending']}")
            print(f"In Progress: {queue_status['in_progress']}")
            print(f"Completed: {queue_status['completed']}")
            print(f"Failed: {queue_status['failed']}")
            if queue_status['in_progress'] > 0:
                cleanup_stale_locks()
    except Exception as e:
        print(f"Error in progress monitor: {e}")

###############################
# ArXiv-specific Processing Functions
###############################
def download_kaggle_dataset(dataset="Cornell-University/arxiv", output_dir=kaggle_dataset_dir):
    """
    Download and unzip the specified Kaggle arXiv dataset.
    Requires that the Kaggle API is installed and configured.
    """
    # Check if the JSON file is already present in the output directory
    json_path = os.path.join(output_dir, 'arxiv-metadata-oai-snapshot.json')
    if os.path.exists(json_path):
        print(f"Kaggle dataset already exists in {output_dir}.")
        return
    os.makedirs(output_dir, exist_ok=True)
    print("Downloading Kaggle arXiv dataset...")
    command = f"kaggle datasets download -d Cornell-University/arxiv -p {output_dir} --unzip"
    try:
        subprocess.run(command, shell=True, check=True)
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")

def stream_extract_to_jsonl(input_file, temp_dir, overwrite_chunks=False):
    """
    Stream the input JSON file and write each record into temporary JSONL files
    based on its category.
    """
    if overwrite_chunks and os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Removed existing chunks in {temp_dir}.")
    os.makedirs(temp_dir, exist_ok=True)
    file_handles = {}
    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Failed to decode: {line}")
                    continue
                cats = obj.get('categories', '').split()
                for cat in cats:
                    if cat not in file_handles:
                        temp_file = os.path.join(temp_dir, f'{cat}.jsonl')
                        file_handles[cat] = open(temp_file, 'a')
                    file_handles[cat].write(json.dumps(obj) + "\n")
    finally:
        for fh in file_handles.values():
            fh.close()
    print("JSONL extraction complete.")

def process_kaggle_files(input_dir, output_dir):
    """
    Process all JSONL files in the Kaggle dataset directory.
    Convert each JSONL file into a Parquet file after cleaning text.
    This function processes files in parallel.
    """
    os.makedirs(output_dir, exist_ok=True)
    json_files = list(Path(input_dir).glob("*.jsonl"))
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    if not json_files:
        print("No JSON files found in the Kaggle dataset directory.")
        return

    def process_single_file(json_file):
        try:
            df = pd.read_json(json_file, lines=True)
        except ValueError:
            with open(json_file, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        if 'title' in df.columns:
            df['title'] = df['title'].astype(str).apply(clean_text)
        if 'abstract' in df.columns:
            df['abstract'] = df['abstract'].astype(str).apply(clean_text)
        parquet_filename = f"{json_file.stem}.parquet"
        save_path = os.path.join(output_dir, parquet_filename)
        # Remove old embedding file if exists
        npy_file_path = os.path.join(DEST_EMBED_FOLDER, f"{json_file.stem}.npy")
        if os.path.exists(npy_file_path):
            os.remove(npy_file_path)
            print(f"Removed embedding file {npy_file_path} due to update")
        df.to_parquet(save_path)
        print(f"Processed and saved {json_file.name} to {parquet_filename}")
        return save_path

    # Process files in parallel using all available CPU cores
    processed_files = Parallel(n_jobs=-1)(
        delayed(process_single_file)(json_file) for json_file in json_files
    )
    return processed_files

def load_parquet_files(data_folder):
    """Return list of all Parquet files in the given folder."""
    data_folder = Path(data_folder)
    return list(data_folder.glob("*.parquet"))

def load_and_dedup_all_parquet(input_directory):
    """
    Load all Parquet files from the given directory in parallel, concatenate them,
    and remove duplicates based on the 'abstract' column.
    """
    file_paths = sorted(glob.glob(os.path.join(input_directory, '*.parquet')))
    if not file_paths:
        print("No Parquet files found in", input_directory)
        return pd.DataFrame()

    def load_single_parquet(file):
        try:
            df = pd.read_parquet(file)
            return df
        except Exception as e:
            print(f"Skipping file {file} due to error: {e}")
            return pd.DataFrame()

    # Load files in parallel
    dataframes = Parallel(n_jobs=-1)(
        delayed(load_single_parquet)(file) for file in file_paths
    )
    # Remove empty DataFrames
    dataframes = [df for df in dataframes if not df.empty]
    if not dataframes:
        print("No valid Parquet files loaded.")
        return pd.DataFrame()
    combined_df = pd.concat(dataframes, ignore_index=True)
    print("Combined dataframe shape before deduplication:", combined_df.shape)
    dedup_df = combined_df.drop_duplicates(subset='abstract')
    print("Combined dataframe shape after deduplication:", dedup_df.shape)
    return dedup_df

def chunk_dataframe(df, target_bytes=50*1024*1024):
    """
    Split the DataFrame into chunks approximating the target size in bytes.
    This uses an approximate in-memory size estimate.
    """
    total_bytes = df.memory_usage(deep=True).sum()
    num_chunks = max(1, int(np.ceil(total_bytes / target_bytes)))
    chunk_size = int(np.ceil(len(df) / num_chunks))
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    print(f"Total in-memory bytes: {total_bytes}, splitting into {len(chunks)} chunks (~{target_bytes} bytes each)")
    return chunks

def save_dataframe_chunks(chunks, output_dir):
    """
    Save each DataFrame chunk to a separate Parquet file in output_dir.
    Returns the list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_files = []
    for idx, chunk in enumerate(chunks):
        filename = os.path.join(output_dir, f"chunk_{idx}.parquet")
        chunk.to_parquet(filename, index=False)
        chunk_files.append(filename)
        print(f"Saved chunk {idx} with {len(chunk)} rows to {filename}")
    return chunk_files

###############################
# Main Processing Routines for arXiv Embeddings
###############################
def main_parallel_embeddings_arxiv(input_directory):
    """
    Process embeddings in parallel (GPU mode) for arXiv by reading parquet chunk files.
    Uses distributed coordination to claim tasks from the shared queue.
    """
    ensure_dirs()
    machine_file, heartbeat_thread = register_machine()
    file_paths = sorted(glob.glob(os.path.join(input_directory, '*.parquet')))
    if not file_paths:
        print("No Parquet files found in the directory.")
        return
    num_pending = initialize_task_queue(file_paths)
    print(f"Task queue initialized with {num_pending} pending files")
    num_gpus_to_use = AVAILABLE_GPUS
    if num_gpus_to_use == 0:
        print("No GPUs available. Switching to CPU mode.")
        main_cpu_embeddings_arxiv(input_directory)
        return
    print(f"Using {num_gpus_to_use} GPUs on {HOSTNAME}")
    for i in range(num_gpus_to_use):
        try:
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name}, Memory: {gpu_mem:.1f} GB")
        except Exception as e:
            print(f"Error getting info for GPU {i}: {e}")
    progress_watcher_stop = threading.Event()
    progress_watcher = threading.Thread(target=monitor_progress, args=(progress_watcher_stop,))
    progress_watcher.daemon = True
    progress_watcher.start()
    start_time = time.time()
    processes = []
    for gpu_id in range(num_gpus_to_use):
        p = multiprocessing.Process(target=process_embeddings_worker, args=(gpu_id, 1))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    progress_watcher_stop.set()
    if progress_watcher.is_alive():
        progress_watcher.join(timeout=1)
    total_time = time.time() - start_time
    queue_status = get_queue_status()
    total_processed = queue_status['completed']
    print(f"\nAll done! Processed {total_processed} files in {total_time:.2f}s")
    with open(machine_file, 'r') as f:
        machine_info = json.load(f)
    machine_info['status'] = 'completed'
    machine_info['end_time'] = int(time.time())
    with open(machine_file, 'w') as f:
        json.dump(machine_info, f)

def main_cpu_embeddings_arxiv(input_directory):
    """
    Process embeddings in CPU mode for arXiv parquet chunks.
    """
    ensure_dirs()
    machine_file, heartbeat_thread = register_machine()
    file_paths = sorted(glob.glob(os.path.join(input_directory, '*.parquet')))
    if not file_paths:
        print("No Parquet files found in the directory.")
        return
    num_pending = initialize_task_queue(file_paths)
    print(f"Task queue initialized with {num_pending} pending files")
    progress_watcher_stop = threading.Event()
    progress_watcher = threading.Thread(target=monitor_progress, args=(progress_watcher_stop,))
    progress_watcher.daemon = True
    progress_watcher.start()
    os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
    print("Loading model on CPU...")
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")
    start_time = time.time()
    processed_count = 0
    batch_size = 5
    while True:
        files_to_process = claim_next_batch(batch_size, -1)
        if not files_to_process:
            print("No more files to process. Finishing CPU processing.")
            break
        print(f"Claimed {len(files_to_process)} files for CPU processing")
        for file in tqdm(files_to_process, desc="CPU embeddings processing"):
            process_embeddings_cpu_arxiv(file, model)
            processed_count += 1
            if processed_count % SAVE_INTERVAL == 0:
                status = get_queue_status()
                print(f"Progress: completed {processed_count} files. "
                      f"Queue: {status['pending']} pending, {status['in_progress']} in progress, "
                      f"{status['completed']} completed, {status['failed']} failed")
    progress_watcher_stop.set()
    if progress_watcher.is_alive():
        progress_watcher.join(timeout=1)
    total_time = time.time() - start_time
    queue_status = get_queue_status()
    total_processed = queue_status['completed']
    print(f"CPU embedding generation completed. Processed {total_processed} files in {total_time:.2f}s")

def monitor_progress(stop_event):
    """Periodically print the status of the task queue and active machines."""
    try:
        while not stop_event.is_set():
            time.sleep(10)
            queue_status = get_queue_status()
            active_machines = get_active_machines()
            print("\n=== Cluster Status ===")
            print(f"Active machines: {len(active_machines)}")
            for machine in active_machines:
                print(f"  {machine['hostname']} (ID: {machine['machine_id']}): {machine['available_gpus']} GPUs")
            print("\n=== Queue Status ===")
            print(f"Pending: {queue_status['pending']}")
            print(f"In Progress: {queue_status['in_progress']}")
            print(f"Completed: {queue_status['completed']}")
            print(f"Failed: {queue_status['failed']}")
            if queue_status['in_progress'] > 0:
                cleanup_stale_locks()
    except Exception as e:
        print(f"Error in progress monitor: {e}")

def main_arxiv(embedding_mode="gpu"):
    """
    Main function for arXiv processing.
    This routine downloads and processes the dataset, deduplicates data,
    chunks the combined DataFrame into ~50 MB parts (using optimized parquet saving),
    and then calculates embeddings.
    """
    overall_start = time.time()
    ensure_dirs()
    # Checkpoint 1: Download Kaggle dataset if not present.
    kaggle_json_path = os.path.join(kaggle_dataset_dir, 'arxiv-metadata-oai-snapshot.json')
    if not os.path.exists(kaggle_json_path):
        print("Kaggle dataset not found. Downloading...")
        download_kaggle_dataset()
    else:
        print("Kaggle dataset already downloaded.")
    
    # Checkpoint 2: Extract JSONL files if not already done.
    jsonl_files = glob.glob(os.path.join(temp_jsonl_dir, "*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found. Extracting from Kaggle JSON...")
        stream_extract_to_jsonl(kaggle_json_path, temp_jsonl_dir, overwrite_chunks=False)
    else:
        print("JSONL files already extracted.")
    
    # Checkpoint 3: Process JSONL files into temporary parquet files if not present.
    temp_parquet_files = glob.glob(os.path.join(TEMP_DF_FOLDER, "*.parquet"))
    if not temp_parquet_files:
        print("No temporary parquet files found. Processing JSONL files to temporary parquet in parallel...")
        temp_parquet_files = process_kaggle_files(temp_jsonl_dir, TEMP_DF_FOLDER)
    else:
        print("Temporary parquet files already exist.")
    
    # Checkpoint 4: Deduplication and chunking.
    chunk_files = glob.glob(os.path.join(DEST_DF_FOLDER, "chunk_*.parquet"))
    if not os.path.exists(COMBINED_DF_FILE) or not chunk_files:
        print("Deduplicated and chunked parquet files not found. Deduplicating...")
        dedup_df = load_and_dedup_all_parquet(TEMP_DF_FOLDER)
        if dedup_df.empty:
            print("No data to process after deduplication.")
            return
        if 'id' in dedup_df.columns:
            dedup_df['id'] = dedup_df['id'].astype(str)
        save_optimized_parquet(dedup_df, COMBINED_DF_FILE, row_group_size=10000)
        print(f"Saved combined deduplicated DataFrame to {COMBINED_DF_FILE}")
        chunks = chunk_dataframe(dedup_df, target_bytes=50*1024*1024)
        chunk_files = save_dataframe_chunks(chunks, DEST_DF_FOLDER)
    else:
        print("Deduplicated and chunked parquet files already exist.")
    
    # Step 5: Calculate embeddings for each parquet chunk.
    if embedding_mode.lower() == "gpu":
        print("Starting GPU embedding processing for arXiv...")
        main_parallel_embeddings_arxiv(DEST_DF_FOLDER)
    else:
        print("Starting CPU embedding processing for arXiv...")
        main_cpu_embeddings_arxiv(DEST_DF_FOLDER)
    overall_end = time.time()
    print(f"Total arXiv processing time: {overall_end - overall_start:.2f} seconds")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    # Remove stale GPU locks if any exist.
    if os.path.exists(GPU_LOCK_FOLDER):
        for lock_file in glob.glob(os.path.join(GPU_LOCK_FOLDER, "*.lock")):
            try:
                os.remove(lock_file)
                print(f"Removed stale GPU lock: {lock_file}")
            except Exception:
                pass
    mode = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Running in {mode.upper()} mode")
    main_arxiv(embedding_mode=mode)
