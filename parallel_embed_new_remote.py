import os
import re
import requests
import gzip
import shutil
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import xml.etree.ElementTree as ET
from html import unescape
from datetime import datetime
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import glob
import unidecode
import multiprocessing
import time
import queue
import threading
import gc
import json
import socket
import uuid
import errno
import signal

###############################
# Global configuration and utility functions
###############################
DEST_DF_FOLDER = 'pubmed25_update_df/'
DEST_EMBED_FOLDER = 'pubmed25_update_embed/'

# New: Shared coordination directory
COORDINATION_DIR = 'coordination/'
TASK_QUEUE_FILE = os.path.join(COORDINATION_DIR, 'task_queue.json')
MACHINE_STATUS_DIR = os.path.join(COORDINATION_DIR, 'machine_status/')

# Machine-specific information
HOSTNAME = socket.gethostname()
MACHINE_ID = str(uuid.uuid4())[:8]  # Generate a unique ID for this run

# GPU lock folder
GPU_LOCK_FOLDER = f'gpu_locks/{HOSTNAME}/'

# Optimized settings
BATCH_SIZE = 64
PREFETCH_SIZE = 1
SAVE_INTERVAL = 1
LOCK_TIMEOUT = 60*25  # seconds

# Get number of available GPUs
try:
    AVAILABLE_GPUS = torch.cuda.device_count()
    print(f"Detected {AVAILABLE_GPUS} available GPUs on {HOSTNAME}")
except Exception as e:
    AVAILABLE_GPUS = 0
    print(f"No GPUs detected on {HOSTNAME}")

def robust_basename(filepath: str) -> str:
    """Return the file name stem (without extension) robustly."""
    return os.path.splitext(os.path.basename(filepath))[0]

###########################################
# New: Distributed coordination system
###########################################
def ensure_dirs():
    """Create all necessary directories"""
    for directory in [DEST_DF_FOLDER, DEST_EMBED_FOLDER, COORDINATION_DIR, 
                     GPU_LOCK_FOLDER, MACHINE_STATUS_DIR]:
        os.makedirs(directory, exist_ok=True)

def register_machine():
    """Register this machine in the coordination system"""
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
    
    # Start heartbeat thread
    heartbeat_thread = threading.Thread(target=machine_heartbeat, args=(machine_file,))
    heartbeat_thread.daemon = True
    heartbeat_thread.start()
    
    return machine_file, heartbeat_thread

def machine_heartbeat(machine_file):
    """Update the machine's heartbeat timestamp periodically"""
    while True:
        try:
            with open(machine_file, 'r') as f:
                machine_info = json.load(f)
            
            machine_info['last_heartbeat'] = int(time.time())
            
            with open(machine_file, 'w') as f:
                json.dump(machine_info, f)
                
        except Exception as e:
            print(f"Heartbeat update failed: {e}")
            
        time.sleep(30)  # Update every 30 seconds

def get_active_machines():
    """Get a list of currently active machines"""
    active_machines = []
    current_time = int(time.time())
    heartbeat_timeout = current_time - 90  # 90 seconds timeout
    
    for machine_file in glob.glob(os.path.join(MACHINE_STATUS_DIR, '*.json')):
        try:
            with open(machine_file, 'r') as f:
                machine_info = json.load(f)
            
            if machine_info['last_heartbeat'] > heartbeat_timeout:
                active_machines.append(machine_info)
            else:
                # Machine hasn't sent a heartbeat recently - consider it inactive
                machine_info['status'] = 'inactive'
                with open(machine_file, 'w') as f:
                    json.dump(machine_info, f)
        except Exception:
            pass
    
    return active_machines

def cleanup_stale_locks():
    """Check for and clean up stale locks from inactive machines"""
    active_machines = get_active_machines()
    active_ids = [m['machine_id'] for m in active_machines]
    
    # Check file locks
    for lock_file in glob.glob(os.path.join(DEST_EMBED_FOLDER, '*.lock')):
        try:
            with open(lock_file, 'r') as f:
                lock_info = f.read()
            
            if ':' in lock_info:  # New format with machine ID
                machine_id = lock_info.split(':')[0]
                if machine_id not in active_ids:
                    print(f"Removing stale lock from inactive machine: {lock_file}")
                    os.remove(lock_file)
        except Exception as e:
            print(f"Error checking lock file {lock_file}: {e}")

###########################################
# Enhanced file locking system
###########################################
class SimpleLock:
    """A simple file lock implementation for distributed systems"""
    def __init__(self, lock_path):
        self.lock_path = lock_path
        self.fd = None
        
    def acquire(self, timeout=LOCK_TIMEOUT):
        """Acquire the lock with timeout"""
        start_time = time.time()
        
        while True:
            try:
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                lock_info = f"{MACHINE_ID}:{HOSTNAME}:{os.getpid()}:{int(time.time())}"
                os.write(self.fd, lock_info.encode())
                return True
            except FileExistsError:
                # Check if the timeout has expired
                if time.time() - start_time > timeout:
                    return False
                
                # Check if the lock is stale (older than 5 minutes)
                try:
                    with open(self.lock_path, 'r') as f:
                        lock_info = f.read()
                    
                    if ':' in lock_info:
                        parts = lock_info.split(':')
                        if len(parts) >= 4:
                            lock_time = int(parts[3])
                            if time.time() - lock_time > 300:  # 5 minutes timeout
                                print(f"Removing stale lock: {self.lock_path}")
                                os.remove(self.lock_path)
                                continue
                except Exception:
                    pass
                
                # Wait a bit before retrying
                time.sleep(0.5)
    
    def release(self):
        """Release the lock"""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        
        if os.path.exists(self.lock_path):
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                # Someone else might have removed it
                pass

def acquire_file_lock(file_stem):
    """Try to acquire a lock for a file using the enhanced SimpleLock"""
    os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
    lock_path = os.path.join(DEST_EMBED_FOLDER, f"{file_stem}.lock")
    
    lock = SimpleLock(lock_path)
    if lock.acquire():
        return lock
    return None

def release_file_lock(lock):
    """Release the file lock"""
    if lock:
        lock.release()

###############################
# New: Task queue management
###############################
def initialize_task_queue(file_paths):
    """Initialize the global task queue with all files that need processing"""
    lock_path = f"{TASK_QUEUE_FILE}.lock"
    queue_lock = SimpleLock(lock_path)
    
    if queue_lock.acquire(timeout=30):  # Longer timeout for initialization
        try:
            if os.path.exists(TASK_QUEUE_FILE):
                # Queue exists, load and update it
                with open(TASK_QUEUE_FILE, 'r') as f:
                    queue_data = json.load(f)
            else:
                # Create new queue
                queue_data = {
                    'pending': [],
                    'in_progress': {},
                    'completed': [],
                    'failed': {}
                }
            
            # Add files that aren't already in queue and don't have embeddings
            current_files = set(queue_data['pending']) | set(queue_data['completed']) | set(queue_data['in_progress'].keys())
            new_files = [f for f in file_paths if f not in current_files and 
                         not os.path.exists(os.path.join(DEST_EMBED_FOLDER, f"{robust_basename(f)}.npy"))]
            
            if new_files:
                queue_data['pending'].extend(new_files)
                print(f"Added {len(new_files)} new files to the task queue")
            
            # Save updated queue
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
    """Claim the next batch of files from the task queue"""
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
            
            # Take up to batch_size files
            to_claim = queue_data['pending'][:batch_size]
            queue_data['pending'] = queue_data['pending'][batch_size:]
            
            # Update in_progress dictionary
            claim_time = int(time.time())
            for file_path in to_claim:
                queue_data['in_progress'][file_path] = {
                    'machine_id': MACHINE_ID,
                    'hostname': HOSTNAME,
                    'gpu_id': gpu_id,
                    'claimed_at': claim_time
                }
                claimed_files.append(file_path)
            
            # Save updated queue
            with open(TASK_QUEUE_FILE, 'w') as f:
                json.dump(queue_data, f)
        finally:
            queue_lock.release()
    
    return claimed_files

def mark_file_completed(file_path, success, error_msg=None):
    """Mark a file as completed or failed in the task queue"""
    lock_path = f"{TASK_QUEUE_FILE}.lock"
    queue_lock = SimpleLock(lock_path)
    
    if queue_lock.acquire():
        try:
            if not os.path.exists(TASK_QUEUE_FILE):
                return
            
            with open(TASK_QUEUE_FILE, 'r') as f:
                queue_data = json.load(f)
            
            # Remove from in_progress
            if file_path in queue_data['in_progress']:
                del queue_data['in_progress'][file_path]
            
            # Add to completed or failed
            if success:
                queue_data['completed'].append(file_path)
            else:
                queue_data['failed'][file_path] = {
                    'error': error_msg,
                    'machine_id': MACHINE_ID,
                    'hostname': HOSTNAME,
                    'time': int(time.time())
                }
            
            # Save updated queue
            with open(TASK_QUEUE_FILE, 'w') as f:
                json.dump(queue_data, f)
        finally:
            queue_lock.release()

def get_queue_status():
    """Get the current status of the task queue"""
    status = {
        'pending': 0,
        'in_progress': 0,
        'completed': 0,
        'failed': 0
    }
    
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

###############################
# Enhanced GPU management
###############################
def acquire_gpu_lock(gpu_id):
    """Acquire a lock for a GPU with enhanced stale lock detection"""
    os.makedirs(GPU_LOCK_FOLDER, exist_ok=True)
    lock_file = os.path.join(GPU_LOCK_FOLDER, f"gpu_{gpu_id}.lock")
    
    if os.path.exists(lock_file):
        # Check if the lock is stale
        try:
            with open(lock_file, "r") as f:
                lock_info = f.read().strip()
            
            # Check if the process still exists
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
                else:
                    print(f"GPU {gpu_id} is already locked. Skipping processing on this GPU.")
                    return None
            elif lock_info.isdigit():
                # Legacy format with just PID
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

def pid_exists(pid):
    """Check whether pid exists in the current process table"""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # ESRCH == No such process
            return False
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            return True
        else:
            raise
    else:
        return True

def release_gpu_lock(lock_file):
    """Release the GPU lock by deleting the lock file"""
    if lock_file and os.path.exists(lock_file):
        os.remove(lock_file)

def set_process_affinity(gpu_id):
    """Set CPU affinity for the current process to reduce context switching"""
    try:
        import psutil
        process = psutil.Process()
        num_cores = psutil.cpu_count(logical=False)
        num_gpus = AVAILABLE_GPUS
        cores_per_gpu = max(1, num_cores // num_gpus)
        start_core = gpu_id * cores_per_gpu
        end_core = start_core + cores_per_gpu
        cpu_list = list(range(start_core, min(end_core, num_cores)))
        if cpu_list:
            process.cpu_affinity(cpu_list)
            print(f"GPU {gpu_id} worker using CPU cores: {cpu_list}")
    except Exception as e:
        print(f"Failed to set CPU affinity: {e}")

###############################
# DataPrefetcher for efficient data loading
###############################
class DataPrefetcher:
    """Prefetch data files to reduce I/O waiting time"""
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
        """Get next prefetched data"""
        return self.queue.get()

    def stop(self):
        """Stop the prefetcher"""
        self.stop_event.set()
        if self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=1)

###############################
# Enhanced workload processing
###############################
def process_embeddings_gpu(file_path, df, queries, model, progress_file):
    """Process embeddings using prefetched data on GPU with distributed coordination"""
    file_stem = robust_basename(file_path)
    embeddings_path = os.path.join(DEST_EMBED_FOLDER, f"{file_stem}.npy")
    
    # Skip if embeddings already exist
    if os.path.exists(embeddings_path):
        print(f"Embeddings for {file_stem} already exist. Skipping.")
        mark_file_completed(file_path, True)
        return True

    # Try to acquire a processing lock for this file
    lock = acquire_file_lock(file_stem)
    if lock is None:
        print(f"File {file_stem} is currently locked by another process. Skipping.")
        return False

    try:
        start_time = time.time()
        print(f"Processing {file_stem} with {len(queries)} entries")
        if not queries or len(queries) == 0:
            print(f"No valid queries found in {file_path}, skipping.")
            mark_file_completed(file_path, False, "No valid queries found")
            return False
        
        query_embeddings = model.encode(
            queries,
            normalize_embeddings=True,
            precision='ubinary',
            batch_size=BATCH_SIZE,
            show_progress_bar=True
        )
        
        np.save(embeddings_path, query_embeddings)
        processing_time = time.time() - start_time
        items_per_second = len(queries) / processing_time
        print(f"Saved {len(queries)} embeddings to {embeddings_path}")
        print(f"Processing time: {processing_time:.2f}s ({items_per_second:.2f} items/sec)")
        
        with open(progress_file, 'a') as f:
            f.write(f"{file_stem},{len(queries)},{processing_time:.2f},{items_per_second:.2f}\n")
        
        mark_file_completed(file_path, True)
        del query_embeddings
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing file {file_path}: {error_msg}")
        mark_file_completed(file_path, False, error_msg)
        return False
    finally:
        release_file_lock(lock)

def process_embeddings_worker(gpu_id, batch_size=1):
    """Worker function that dynamically claims and processes files from the shared queue"""
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
            print(f"Worker for GPU {gpu_id} is using device: {gpu_name} with {gpu_mem:.1f} GB memory")
        else:
            print(f"Warning: CUDA not available in worker for GPU {gpu_id}")
        
        torch.backends.cudnn.benchmark = True
        print(f"[GPU {gpu_id}] Starting worker initialization")
        start_time = time.time()
        print(f"[GPU {gpu_id}] Loading model...")
        
        model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
        model.to(device)
        
        try:
            import psutil
            num_cores = psutil.cpu_count(logical=False)
            cores_per_gpu = max(1, num_cores // AVAILABLE_GPUS)
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
            # Claim a batch of files to process
            files_to_process = claim_next_batch(batch_size, gpu_id)
            
            if not files_to_process:
                print(f"[GPU {gpu_id}] No more files to process, worker finishing")
                break
            
            print(f"[GPU {gpu_id}] Claimed {len(files_to_process)} files for processing")
            
            # Process each file with prefetching
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
                            try:
                                del df, queries
                            except Exception:
                                pass
                            gc.collect()
                            torch.cuda.empty_cache()
                            
                            # Print status update periodically
                            status = get_queue_status()
                            print(f"[GPU {gpu_id}] Progress: completed {processed_count} files. "
                                  f"Queue status: {status['pending']} pending, {status['in_progress']} in progress, "
                                  f"{status['completed']} completed, {status['failed']} failed")
                except Exception as e:
                    print(f"[GPU {gpu_id}] Error processing file {file_path}: {e}")
                    mark_file_completed(file_path, False, str(e))
                finally:
                    # Clean up for better garbage collection
                    df = None
                    queries = None
            
            # Clean up prefetcher for this batch
            prefetcher.stop()
                
        if processed_count > 0 and total_time > 0:
            avg_time = total_time / processed_count
            items_per_sec = total_entries / total_time
            print(f"[GPU {gpu_id}] Completed {processed_count} files in {total_time:.2f}s")
            print(f"[GPU {gpu_id}] Average: {avg_time:.2f}s per file, {items_per_sec:.2f} items/sec")
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
        print(f"[GPU {gpu_id}] Worker finished and resources released")

###############################
# Enhanced monitoring
###############################
def monitor_progress(stop_event):
    """Enhanced progress monitor that provides cluster-wide status"""
    try:
        while not stop_event.is_set():
            time.sleep(10)
            
            # Get queue status
            queue_status = get_queue_status()
            
            # Get active machines
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
            
            # Clean up stale locks if needed
            if queue_status['in_progress'] > 0:
                cleanup_stale_locks()
    except Exception as e:
        print(f"Error in progress monitor: {e}")

###############################
# Main Entry Point with distributed coordination
###############################
def main_parallel_embeddings(input_directory):
    """Process embeddings in parallel with multi-workstation coordination"""
    # Ensure all required directories exist
    ensure_dirs()
    
    # Register this machine in the coordination system
    machine_file, heartbeat_thread = register_machine()
    
    # Find all parquet files to process
    file_paths = sorted(glob.glob(os.path.join(input_directory, '*.parquet')))
    if not file_paths:
        print("No Parquet files found in the directory.")
        return
    
    # Initialize or update the task queue
    num_pending = initialize_task_queue(file_paths)
    print(f"Task queue initialized with {num_pending} pending files")
    
    # Check GPU availability
    num_gpus_to_use = AVAILABLE_GPUS
    if num_gpus_to_use == 0:
        print("No GPUs available. Switching to CPU mode.")
        main_cpu_embeddings(input_directory)
        return
    
    print(f"Using {num_gpus_to_use} available GPUs on {HOSTNAME}")
    for i in range(num_gpus_to_use):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"GPU {i}: {gpu_name}, Memory: {gpu_mem:.1f} GB")
    
    # Start progress monitor
    progress_watcher_stop = threading.Event()
    progress_watcher = threading.Thread(
        target=monitor_progress,
        args=(progress_watcher_stop,)
    )
    progress_watcher.daemon = True
    progress_watcher.start()
    
    # Start GPU workers - each worker will claim tasks from the queue
    start_time = time.time()
    processes = []
    
    for gpu_id in range(num_gpus_to_use):
        p = multiprocessing.Process(
            target=process_embeddings_worker,
            args=(gpu_id, 1)  # Claim 1 file at a time
        )
        p.start()
        processes.append(p)
    
    # Wait for all workers to finish
    for p in processes:
        p.join()
    
    # Stop the progress monitor
    progress_watcher_stop.set()
    if progress_watcher.is_alive():
        progress_watcher.join(timeout=1)
    
    # Calculate and display statistics
    total_time = time.time() - start_time
    queue_status = get_queue_status()
    total_processed = queue_status['completed']
    
    print(f"\nAll done! Processed {total_processed} files in {total_time:.2f}s")
    
    # Update machine status to completed
    with open(machine_file, 'r') as f:
        machine_info = json.load(f)
    
    machine_info['status'] = 'completed'
    machine_info['end_time'] = int(time.time())
    
    with open(machine_file, 'w') as f:
        json.dump(machine_info, f)

# CPU version with distributed coordination
def process_embeddings_cpu(file, model):
    """Process embeddings on CPU with distributed coordination"""
    file_stem = robust_basename(file)
    embeddings_path = os.path.join(DEST_EMBED_FOLDER, f"{file_stem}.npy")
    
    if os.path.exists(embeddings_path):
        print(f"Embeddings for {file_stem} already exist. Skipping.")
        mark_file_completed(file, True)
        return
    
    lock = acquire_file_lock(file_stem)
    if lock is None:
        print(f"File {file_stem} is currently locked by another process. Skipping.")
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
        print(f"Saved embeddings at {embeddings_path} in {processing_time:.2f}s")
        
        mark_file_completed(file, True)
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing file {file}: {error_msg}")
        mark_file_completed(file, False, error_msg)
    finally:
        release_file_lock(lock)

def main_cpu_embeddings(input_directory):
    """CPU processing with distributed coordination"""
    # Ensure all required directories exist
    ensure_dirs()
    
    # Register this machine in the coordination system
    machine_file, heartbeat_thread = register_machine()
    
    # Initialize task queue
    file_paths = sorted(glob.glob(os.path.join(input_directory, '*.parquet')))
    if not file_paths:
        print("No parquet files found in the directory.")
        return
    
    num_pending = initialize_task_queue(file_paths)
    print(f"Task queue initialized with {num_pending} pending files")
    
    # Start progress monitor
    progress_watcher_stop = threading.Event()
    progress_watcher = threading.Thread(
        target=monitor_progress,
        args=(progress_watcher_stop,)
    )
    progress_watcher.daemon = True
    progress_watcher.start()
    
    # Load model
    os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
    print("Loading model on CPU...")
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")
    
    start_time = time.time()
    processed_count = 0
    batch_size = 5  # Number of files to claim at once
    
    while True:
        # Claim a batch of files to process
        files_to_process = claim_next_batch(batch_size, -1)  # Use -1 to indicate CPU
        
        if not files_to_process:
            print("No more files to process, finishing")
            break
        
        print(f"Claimed {len(files_to_process)} files for CPU processing")
        
        for file in tqdm(files_to_process, desc="CPU embeddings processing"):
            process_embeddings_cpu(file, model)
            processed_count += 1
            
            # Print status update periodically
            if processed_count % SAVE_INTERVAL == 0:
                status = get_queue_status()
                print(f"Progress: completed {processed_count} files. "
                      f"Queue status: {status['pending']} pending, {status['in_progress']} in progress, "
                      f"{status['completed']} completed, {status['failed']} failed")
    
    # Stop the progress monitor
    progress_watcher_stop.set()
    if progress_watcher.is_alive():
        progress_watcher.join(timeout=1)
    
    # Calculate and display statistics
    total_time = time.time() - start_time
    queue_status = get_queue_status()
    total_processed = queue_status['completed']
    
    print(f"CPU embedding generation completed.")
    print(f"Processed {total_processed} files in {total_time:.2f}s")

def main(embedding_mode):
    """Main function with enhanced distributed coordination"""
    if embedding_mode.lower() == "gpu":
        print("Starting GPU embeddings processing with distributed coordination...")
        main_parallel_embeddings(DEST_DF_FOLDER)
    else:
        print("Starting CPU embeddings processing with distributed coordination...")
        main_cpu_embeddings(DEST_DF_FOLDER)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    # Clean up any stale GPU locks from previous runs on this machine
    if os.path.exists(GPU_LOCK_FOLDER):
        for lock_file in glob.glob(os.path.join(GPU_LOCK_FOLDER, "*.lock")):
            try:
                os.remove(lock_file)
                print(f"Removed stale GPU lock: {lock_file}")
            except:
                pass
    
    mode = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Running in {mode.upper()} mode")
    
    overall_start = time.time()
    main(mode)
    print(f"Total execution time: {time.time() - overall_start:.2f} seconds")
