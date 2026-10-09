import os
import glob
import time
import json
import gc
import errno
import socket
import uuid
import threading
import multiprocessing
import queue

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# Paths and global config
# ------------------------------------------------------------
# NOTE: On Vast.ai, ensure these paths exist or are mapped to your storage
DEST_DF_FOLDER = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/pubmed26_parquet_files/' 
DEST_EMBED_FOLDER = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/pubmed26_update_embed/'
COORDINATION_DIR = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/pubmed26_update_coordination/'

TASK_QUEUE_FILE = os.path.join(COORDINATION_DIR, 'task_queue.json')
MACHINE_STATUS_DIR = os.path.join(COORDINATION_DIR, 'machine_status/')
GPU_LOCK_FOLDER = f"gpu_locks/{socket.gethostname()}/"

HOSTNAME = socket.gethostname()
MACHINE_ID = str(uuid.uuid4())[:8]

# Tunables
# Snowflake is larger than MxBai. 256 is safe for 24GB+ VRAM. 
# The code handles OOM automatically by halving this if it crashes.
BATCH_SIZE = 256  
PREFETCH_SIZE = 2
SAVE_INTERVAL = 1
LOCK_TIMEOUT = 60 * 25  # seconds
INPROGRESS_MAX_AGE = 30 * 60  # seconds

# Snowflake v2 output is 768 dimensions. 
# 768 bits / 8 bits per byte = 96 bytes.
UBINARY_DIM = 96 
MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def robust_basename(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0]

try:
    AVAILABLE_GPUS = torch.cuda.device_count()
    print(f"Detected {AVAILABLE_GPUS} available GPUs on {HOSTNAME}")
except Exception:
    AVAILABLE_GPUS = 0
    print(f"No GPUs detected on {HOSTNAME}")


def ensure_dirs():
    for d in [DEST_DF_FOLDER, DEST_EMBED_FOLDER, COORDINATION_DIR, GPU_LOCK_FOLDER, MACHINE_STATUS_DIR]:
        os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------
# Quantization Logic (From Strategy Code)
# ------------------------------------------------------------
def quantize_and_pack(embeddings):
    """
    Converts Float embeddings to Packed Binary (uint8).
    1. Threshold: x > 0 -> 1, x <= 0 -> 0
    2. Pack: 8 bits -> 1 uint8 byte
    Output shape: (N, 768) -> (N, 96)
    """
    # 1. Binarize (True/False)
    bits = (embeddings > 0)
    # 2. Pack bits into uint8
    packed = np.packbits(bits, axis=1)
    return packed

# ------------------------------------------------------------
# Machine registry & heartbeats
# ------------------------------------------------------------

def register_machine():
    machine_info = {
        'hostname': HOSTNAME,
        'machine_id': MACHINE_ID,
        'available_gpus': AVAILABLE_GPUS,
        'start_time': int(time.time()),
        'last_heartbeat': int(time.time()),
        'status': 'active'
    }
    fn = os.path.join(MACHINE_STATUS_DIR, f"{HOSTNAME}_{MACHINE_ID}.json")
    with open(fn, 'w') as f:
        json.dump(machine_info, f)

    def beat():
        while True:
            try:
                with open(fn, 'r') as rf:
                    info = json.load(rf)
                info['last_heartbeat'] = int(time.time())
                with open(fn, 'w') as wf:
                    json.dump(info, wf)
            except Exception as e:
                print(f"Heartbeat update failed: {e}")
            time.sleep(30)

    t = threading.Thread(target=beat, daemon=True)
    t.start()
    return fn, t


def get_active_machines():
    active = []
    now = int(time.time())
    cutoff = now - 90
    for fn in glob.glob(os.path.join(MACHINE_STATUS_DIR, '*.json')):
        try:
            with open(fn, 'r') as f:
                info = json.load(f)
            if info.get('last_heartbeat', 0) > cutoff:
                active.append(info)
            else:
                info['status'] = 'inactive'
                with open(fn, 'w') as wf:
                    json.dump(info, wf)
        except Exception:
            pass
    return active


# ------------------------------------------------------------
# Simple file lock for per-file processing
# ------------------------------------------------------------

class SimpleLock:
    def __init__(self, lock_path):
        self.lock_path = lock_path
        self.fd = None

    def acquire(self, timeout=LOCK_TIMEOUT):
        start = time.time()
        while True:
            try:
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self.fd, f"{MACHINE_ID}:{HOSTNAME}:{os.getpid()}:{int(time.time())}".encode())
                return True
            except FileExistsError:
                try:
                    with open(self.lock_path, 'r') as f:
                        parts = f.read().split(':')
                    if len(parts) >= 4 and (time.time() - int(parts[3]) > 300):
                        print(f"Removing stale lock: {self.lock_path}")
                        os.remove(self.lock_path)
                        continue
                except Exception:
                    pass
                if time.time() - start > timeout:
                    return False
                time.sleep(0.5)

    def release(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            if os.path.exists(self.lock_path):
                os.remove(self.lock_path)
        except FileNotFoundError:
            pass


def acquire_file_lock(file_stem):
    os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
    lock = SimpleLock(os.path.join(DEST_EMBED_FOLDER, f"{file_stem}.lock"))
    return lock if lock.acquire() else None


def release_file_lock(lock):
    if lock:
        lock.release()


# ------------------------------------------------------------
# GPU locking
# ------------------------------------------------------------

def pid_exists(pid: int) -> bool:
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
    os.makedirs(GPU_LOCK_FOLDER, exist_ok=True)
    lf = os.path.join(GPU_LOCK_FOLDER, f"gpu_{gpu_id}.lock")
    if os.path.exists(lf):
        try:
            with open(lf, 'r') as f:
                s = f.read().strip()
            parts = s.split(':')
            if parts and parts[0].isdigit():
                pid = int(parts[0])
                if not pid_exists(pid):
                    print(f"Removing stale GPU lock for GPU {gpu_id} (PID {pid} not found)")
                    os.remove(lf)
                else:
                    print(f"GPU {gpu_id} is locked by active process {pid}. Skipping.")
                    return None
            else:
                return None
        except Exception:
            return None
    with open(lf, 'w') as f:
        f.write(f"{os.getpid()}:{MACHINE_ID}:{int(time.time())}")
    return lf


def release_gpu_lock(lock_file):
    if lock_file and os.path.exists(lock_file):
        os.remove(lock_file)


def set_process_affinity(gpu_id):
    try:
        import psutil
        p = psutil.Process()
        n_phys = psutil.cpu_count(logical=False) or psutil.cpu_count()
        n_gpu = max(1, AVAILABLE_GPUS)
        per = max(1, n_phys // n_gpu)
        start = gpu_id * per
        cpu_list = list(range(start, min(start + per, n_phys)))
        if cpu_list:
            p.cpu_affinity(cpu_list)
            print(f"GPU {gpu_id} worker using CPU cores: {cpu_list}")
    except Exception as e:
        print(f"Failed to set CPU affinity: {e}")


# ------------------------------------------------------------
# Queue (JSON) helpers
# ------------------------------------------------------------

QUEUE_LOCK = TASK_QUEUE_FILE + '.lock'


def _with_queue_locked(fn):
    lock = SimpleLock(QUEUE_LOCK)
    if not lock.acquire(timeout=30):
        raise RuntimeError('Could not acquire task queue lock')
    try:
        if os.path.exists(TASK_QUEUE_FILE):
            with open(TASK_QUEUE_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {'pending': [], 'in_progress': {}, 'completed': [], 'failed': {}}
        res = fn(data)
        with open(TASK_QUEUE_FILE, 'w') as f:
            json.dump(data, f)
        return res
    finally:
        lock.release()


def audit_parquet_vs_npy(file_paths):
    npys = set(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(DEST_EMBED_FOLDER, '*.npy')))
    stms = [robust_basename(p) for p in file_paths]
    missing = [p for p, s in zip(file_paths, stms) if s not in npys]
    orphans = [n for n in npys if os.path.join(DEST_DF_FOLDER, n + '.parquet') not in file_paths]
    return missing, orphans


def reconcile_queue_with_disk(file_paths, requeue_age=INPROGRESS_MAX_AGE):
    def _reconcile(q):
        q['pending'] = list(dict.fromkeys(q.get('pending', [])))
        q['completed'] = list(dict.fromkeys(q.get('completed', [])))

        completed = list(q['completed'])
        restored = []
        for f in completed:
            stem = robust_basename(f)
            npy = os.path.join(DEST_EMBED_FOLDER, stem + '.npy')
            if not os.path.exists(npy):
                if f not in q['pending']:
                    q['pending'].append(f)
                if f in q['completed']:
                    q['completed'].remove(f)
                restored.append(f)
        if restored:
            print(f"Re-queued {len(restored)} file(s) that were incorrectly marked completed but missing outputs.")

        now = int(time.time())
        stale = []
        for f, meta in list(q['in_progress'].items()):
            claimed = int(meta.get('claimed_at', now))
            if now - claimed > requeue_age:
                stem = robust_basename(f)
                lock_path = os.path.join(DEST_EMBED_FOLDER, stem + '.lock')
                if not os.path.exists(lock_path):
                    stale.append(f)
        for f in stale:
            q['in_progress'].pop(f, None)
            if f not in q['pending']:
                q['pending'].append(f)
        if stale:
            print(f"Re-queued {len(stale)} stale in-progress file(s).")

        missing, _ = audit_parquet_vs_npy(file_paths)
        added = 0
        for f in missing:
            if f not in q['pending'] and f not in q['in_progress']:
                q['pending'].append(f)
                added += 1
        if added:
            print(f"Added {added} missing file(s) to pending based on disk audit.")
    _with_queue_locked(_reconcile)


def initialize_task_queue(file_paths):
    def _init(q):
        current = set(q.get('pending', [])) | set(q.get('completed', [])) | set(q.get('in_progress', {}).keys())
        existing_npy_files = glob.glob(os.path.join(DEST_EMBED_FOLDER, '*.npy'))
        print(f"Found {len(existing_npy_files)} existing .npy files.")

        new_files = [f for f in file_paths if (f not in current) and (not os.path.exists(os.path.join(DEST_EMBED_FOLDER, robust_basename(f) + '.npy')))]
        print(f"Found {len(new_files)} new files to process.")
        if new_files:
            q['pending'].extend(new_files)
        return len(q['pending'])
    pending = _with_queue_locked(_init)

    reconcile_queue_with_disk(file_paths)

    def _count(q):
        return len(q.get('pending', []))
    return _with_queue_locked(_count)


def claim_next_batch(batch_size, gpu_id):
    def _claim(q):
        claimed = []
        if not q.get('pending'):
            return claimed
        to_claim = q['pending'][:batch_size]
        q['pending'] = q['pending'][batch_size:]
        ts = int(time.time())
        for f in to_claim:
            q['in_progress'][f] = {'machine_id': MACHINE_ID, 'hostname': HOSTNAME, 'gpu_id': gpu_id, 'claimed_at': ts}
            claimed.append(f)
        return claimed
    return _with_queue_locked(_claim)


def mark_file_completed(file_path, success, error_msg=None):
    def _mark(q):
        q['in_progress'].pop(file_path, None)
        if success:
            q['completed'].append(file_path)
        else:
            q['failed'][file_path] = {'error': error_msg, 'machine_id': MACHINE_ID, 'hostname': HOSTNAME, 'time': int(time.time())}
    _with_queue_locked(_mark)


def get_queue_status():
    def _status(q):
        return {
            'pending': len(q.get('pending', [])),
            'in_progress': len(q.get('in_progress', {})),
            'completed': len(q.get('completed', [])),
            'failed': len(q.get('failed', {})),
        }
    return _with_queue_locked(_status)


# ------------------------------------------------------------
# Prefetcher
# ------------------------------------------------------------

class DataPrefetcher:
    def __init__(self, file_paths, max_prefetch=2):
        self.file_paths = file_paths
        self.q = queue.Queue(maxsize=max_prefetch)
        self.stop_event = threading.Event()
        # FIXED: Removed 'self.t = t' which caused the crash
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        for fp in self.file_paths:
            if self.stop_event.is_set():
                break
            try:
                df = pd.read_parquet(fp, columns=['title', 'abstract'])
                # UPDATE: Strategy format "Title. Abstract"
                texts = (df['title'].fillna('') + ". " + df['abstract'].fillna('')).tolist()
                self.q.put((fp, df, texts))
            except Exception as e:
                print(f"Error prefetching {fp}: {e}")
                self.q.put((fp, None, None))
        self.q.put((None, None, None))

    def next(self):
        return self.q.get()

    def stop(self):
        self.stop_event.set()
        try:
            if self.t.is_alive():
                self.t.join(timeout=1)
        except Exception:
            pass


# ------------------------------------------------------------
# Embedding processing (GPU)
# ------------------------------------------------------------

@torch.inference_mode()
def process_embeddings_gpu(file_path, df, texts, model, progress_file):
    stem = robust_basename(file_path)
    out_path = os.path.join(DEST_EMBED_FOLDER, f"{stem}.npy")

    if os.path.exists(out_path):
        print(f"Embeddings for {stem} already exist. Skipping.")
        mark_file_completed(file_path, True)
        return True

    lock = acquire_file_lock(stem)
    if lock is None:
        print(f"File {stem} is currently locked by another process. Skipping.")
        return False

    try:
        start = time.time()
        if not texts:
            # Create an empty ubinary array
            empty = np.empty((0, UBINARY_DIM), dtype=np.uint8)
            np.save(out_path, empty)
            print(f"{stem}: no valid texts; wrote empty ubinary array (0,{UBINARY_DIM}).")
            mark_file_completed(file_path, True)
            return True

        # Dynamic batch with OOM backoff
        bs = BATCH_SIZE
        while True:
            try:
                # UPDATE: Using standard float16 autocast encoding, then manual packing
                # This ensures parity with the 'Single Workstation' code strategy
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    embs_float = model.encode(
                        texts,
                        batch_size=bs,
                        show_progress_bar=True,
                        normalize_embeddings=True,
                        convert_to_numpy=True
                    )
                
                # Check format
                if not isinstance(embs_float, np.ndarray):
                    embs_float = np.array(embs_float)

                # APPLY STRATEGY: Quantize and Pack manually
                embs_packed = quantize_and_pack(embs_float)

                # Sanity check
                if embs_packed.shape[1] != UBINARY_DIM:
                    print(f"Warning: expected dim {UBINARY_DIM}, got {embs_packed.shape[1]}.")

                embs_packed = np.ascontiguousarray(embs_packed, dtype=np.uint8)
                break

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e) and bs > 4:
                    bs = max(4, bs // 2)
                    print(f"OOM at batch_size={bs*2}. Reducing to {bs} and retrying…")
                    torch.cuda.empty_cache()
                    continue
                raise

        np.save(out_path, embs_packed)
        dur = time.time() - start
        ips = len(texts) / dur if dur > 0 else 0
        print(f"Saved {len(texts)} ubinary uint8 embeddings -> {out_path} in {dur:.2f}s ({ips:.2f} items/s)")
        with open(progress_file, 'a') as f:
            f.write(f"{stem},{len(texts)},{dur:.2f},{ips:.2f}\n")
        mark_file_completed(file_path, True)
        del embs_float
        del embs_packed
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        mark_file_completed(file_path, False, str(e))
        return False
    finally:
        release_file_lock(lock)


def process_embeddings_worker(gpu_id, batch_size=1):
    lock_file = acquire_gpu_lock(gpu_id)
    if lock_file is None:
        return

    set_process_affinity(gpu_id)

    try:
        os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
        prog_file = os.path.join(DEST_EMBED_FOLDER, f"progress_{HOSTNAME}_gpu_{gpu_id}.csv")
        with open(prog_file, 'w') as f:
            f.write("file,num_entries,processing_time,items_per_second\n")

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = torch.device('cuda:0')
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"Worker for GPU {gpu_id} is using device: {name} with {mem:.1f} GB memory")
        else:
            print(f"Warning: CUDA not available for GPU worker {gpu_id}")

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        print(f"[GPU {gpu_id}] Loading Snowflake model: {MODEL_ID} ...")
        t0 = time.time()
        
        # UPDATE: Load Snowflake v2 with specific settings
        model = SentenceTransformer(
            MODEL_ID, 
            trust_remote_code=True,
            model_kwargs={
                "attn_implementation": "sdpa",
                "dtype": torch.float16 # FIXED: Changed 'torch_dtype' to 'dtype'
            }
        )
        #model.max_seq_length = 512
        model.to(device)

        try:
            import psutil
            n_phys = psutil.cpu_count(logical=False) or psutil.cpu_count()
            per = max(1, n_phys // max(1, AVAILABLE_GPUS))
            torch.set_num_threads(per)
            print(f"[GPU {gpu_id}] Set torch thread count to {per}")
        except Exception:
            pass

        print(f"[GPU {gpu_id}] Model loaded in {time.time() - t0:.2f}s")

        processed_count = 0
        total_time = 0
        total_entries = 0
        prefetcher = None

        while True:
            files = claim_next_batch(batch_size, gpu_id)
            if not files:
                print(f"[GPU {gpu_id}] No more files to process, worker finishing")
                break

            print(f"[GPU {gpu_id}] Claimed {len(files)} file(s)")
            if prefetcher:
                prefetcher.stop()
            prefetcher = DataPrefetcher(files, max_prefetch=PREFETCH_SIZE)

            while True:
                t = time.time()
                fp, df, texts = prefetcher.next()
                if fp is None:
                    break
                try:
                    t = time.time()
                    ok = process_embeddings_gpu(fp, df, texts, model, prog_file)
                    dt = time.time() - t
                    if ok:
                        processed_count += 1
                        total_time += dt
                        total_entries += len(texts) if texts else 0
                        if processed_count % SAVE_INTERVAL == 0:
                            df = None
                            texts = None
                            gc.collect()
                            torch.cuda.empty_cache()
                            status = get_queue_status()
                            print(f"[GPU {gpu_id}] Progress: completed {processed_count}. Queue: {status}")
                except Exception as e:
                    print(f"[GPU {gpu_id}] Error processing {fp}: {e}")
                    mark_file_completed(fp, False, str(e))
                finally:
                    df = None
                    texts = None

            prefetcher.stop()

        if processed_count and total_time:
            print(f"[GPU {gpu_id}] Completed {processed_count} files in {total_time:.2f}s; avg {total_time/processed_count:.2f}s/file; {(total_entries/total_time):.2f} items/s")
    except Exception as e:
        print(f"[GPU {gpu_id}] Worker error: {e}")
    finally:
        try:
            if 'prefetcher' in locals() and prefetcher:
                prefetcher.stop()
        except Exception:
            pass
        if 'model' in locals():
            del model
        gc.collect()
        torch.cuda.empty_cache()
        release_gpu_lock(lock_file)
        print(f"[GPU {gpu_id}] Worker finished and resources released")


def monitor_progress(stop_event, file_paths):
    try:
        while not stop_event.is_set():
            time.sleep(10)
            status = get_queue_status()
            active = get_active_machines()
            missing, orphans = audit_parquet_vs_npy(file_paths)
            print("\n=== Cluster Status ===")
            print(f"Active machines: {len(active)}")
            for m in active:
                print(f"  {m['hostname']} (ID {m['machine_id']}): {m['available_gpus']} GPUs")
            print("=== Queue Status ===")
            print(status)
            print(f"Audit: missing outputs={len(missing)}, orphan npy={len(orphans)}")
            reconcile_queue_with_disk(file_paths)
    except Exception as e:
        print(f"Error in progress monitor: {e}")


def main_parallel_embeddings(input_directory):
    ensure_dirs()
    machine_file, _ = register_machine()

    file_paths = sorted(glob.glob(os.path.join(input_directory, '*.parquet')))
    print(f"Found {len(file_paths)} Parquet files.")
    if not file_paths:
        print("No Parquet files found in the directory.")
        return

    pending = initialize_task_queue(file_paths)
    print(f"Task queue initialized with {pending} pending files")

    if AVAILABLE_GPUS == 0:
        print("No GPUs available. This script is optimized for GPU workers.")
        return 

    print(f"Using {AVAILABLE_GPUS} available GPUs on {HOSTNAME}")
    for i in range(AVAILABLE_GPUS):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}, Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.1f} GB")

    stop_evt = threading.Event()
    mon = threading.Thread(target=monitor_progress, args=(stop_evt, file_paths), daemon=True)
    mon.start()

    start = time.time()
    procs = []
    # Each process handles 1 GPU and claims 1 file at a time
    for gid in range(AVAILABLE_GPUS):
        p = multiprocessing.Process(target=process_embeddings_worker, args=(gid, 1))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    stop_evt.set()
    if mon.is_alive():
        mon.join(timeout=1)

    total_time = time.time() - start
    status = get_queue_status()
    print(f"\nAll done! Processed {status['completed']} files in {total_time:.2f}s")

    try:
        with open(os.path.join(MACHINE_STATUS_DIR, f"{HOSTNAME}_{MACHINE_ID}.json"), 'r') as f:
            info = json.load(f)
        info['status'] = 'completed'
        info['end_time'] = int(time.time())
        with open(os.path.join(MACHINE_STATUS_DIR, f"{HOSTNAME}_{MACHINE_ID}.json"), 'w') as f:
            json.dump(info, f)
    except Exception:
        pass


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    if os.path.exists(GPU_LOCK_FOLDER):
        for lf in glob.glob(os.path.join(GPU_LOCK_FOLDER, '*.lock')):
            try:
                os.remove(lf)
                print(f"Removed stale GPU lock: {lf}")
            except Exception:
                pass

    t0 = time.time()
    print("Starting Distributed GPU embeddings processing (Snowflake v2 Binary Strategy)...")
    main_parallel_embeddings(DEST_DF_FOLDER)
    print(f"Total execution time: {time.time() - t0:.2f} seconds")