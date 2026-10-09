import os
import glob
import time
import json
import gc
import socket
import uuid
import threading
import multiprocessing
import queue
import random
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# PATHS & CONFIG
# ------------------------------------------------------------
DEST_DF_FOLDER = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/pubmed26_parquet_files/' 
DEST_EMBED_FOLDER = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/pubmed26_update_embed/'
COORDINATION_DIR = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/pubmed26_update_coordination/'

TASK_QUEUE_FILE = os.path.join(COORDINATION_DIR, 'task_queue.json')
MACHINE_STATUS_DIR = os.path.join(COORDINATION_DIR, 'machine_status/')
GPU_LOCK_FOLDER = f"gpu_locks/{socket.gethostname()}/"

HOSTNAME = socket.gethostname()
MACHINE_ID = str(uuid.uuid4())[:8]

# --- TUNABLES ---
BATCH_SIZE = 1024       
PREFETCH_SIZE = 4
SAVE_INTERVAL = 1
LOCK_TIMEOUT = 60 * 25 
INPROGRESS_MAX_AGE = 30 * 60

# Reporting: Print progress every X docs
REPORT_CHUNK_SIZE = 5000 

# BGE Specifics
MODEL_ID = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
OUTPUT_BYTES = 48 # 384 bits / 8

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def robust_basename(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0]

def ensure_dirs():
    for d in [DEST_DF_FOLDER, DEST_EMBED_FOLDER, COORDINATION_DIR, GPU_LOCK_FOLDER, MACHINE_STATUS_DIR]:
        os.makedirs(d, exist_ok=True)

try:
    AVAILABLE_GPUS = torch.cuda.device_count()
    print(f"Detected {AVAILABLE_GPUS} available GPUs on {HOSTNAME}")
except Exception:
    AVAILABLE_GPUS = 0
    print(f"No GPUs detected on {HOSTNAME}")

# ------------------------------------------------------------
# QUANTIZATION & INTEGRITY LOGIC
# ------------------------------------------------------------
def quantize_and_pack_binary(embeddings):
    """Float (Normalized) -> Binary Packed (uint8)"""
    # Ensure numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().float().numpy()
    
    # Threshold at 0 (Binary Quantization)
    bits = (embeddings > 0)
    
    # Pack into bytes
    return np.packbits(bits, axis=1)

def calculate_hamming_similarity(bits_a, bits_b):
    """Calculates percentage of matching bits between two packed uint8 arrays."""
    xor_diff = np.bitwise_xor(bits_a, bits_b)
    diff_bits = np.unpackbits(xor_diff).sum()
    total_bits = len(bits_a) * 8
    matching_bits = total_bits - diff_bits
    return matching_bits / total_bits

def check_file_integrity(npy_path, df, model, samples=5, threshold=0.96):
    """
    Verifies if a saved .npy file matches the current model output.
    Returns True if valid (SKIP processing), False if invalid (REPROCESS).
    """
    try:
        # 1. Load existing embeddings
        saved_emb = np.load(npy_path)
        
        # 2. Check dimensions
        if len(saved_emb) != len(df):
            print(f"Integrity Fail: Count mismatch (Saved: {len(saved_emb)} vs DF: {len(df)})")
            return False

        if len(df) == 0:
            return True

        # 3. Select random samples
        indices = random.sample(range(len(df)), min(samples, len(df)))
        subset_df = df.iloc[indices]
        
        # 4. Reconstruct Text
        texts = (QUERY_PREFIX + subset_df['title'].fillna('') + ". " + subset_df['abstract'].fillna('')).tolist()
        
        # 5. Generate fresh embeddings (CPU or GPU depending on model location)
        # Note: We use the passed model which is already on GPU
        with torch.no_grad():
             embeddings_float = model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
        
        fresh_packed = quantize_and_pack_binary(embeddings_float)
        saved_subset = saved_emb[indices]

        # 6. Compare
        for i in range(len(indices)):
            sim = calculate_hamming_similarity(fresh_packed[i], saved_subset[i])
            if sim < threshold:
                print(f"Integrity Fail: Low similarity {sim:.2%} at index {indices[i]}")
                return False # Too different, recompute file
                
        return True # File is good

    except Exception as e:
        print(f"Integrity Check Error: {e}")
        return False

# ------------------------------------------------------------
# MACHINE REGISTRY
# ------------------------------------------------------------
def register_machine():
    machine_info = {
        'hostname': HOSTNAME, 'machine_id': MACHINE_ID,
        'available_gpus': AVAILABLE_GPUS, 'start_time': int(time.time()),
        'last_heartbeat': int(time.time()), 'status': 'active'
    }
    fn = os.path.join(MACHINE_STATUS_DIR, f"{HOSTNAME}_{MACHINE_ID}.json")
    with open(fn, 'w') as f: json.dump(machine_info, f)

    def beat():
        while True:
            try:
                with open(fn, 'r') as rf: info = json.load(rf)
                info['last_heartbeat'] = int(time.time())
                with open(fn, 'w') as wf: json.dump(info, wf)
            except: pass
            time.sleep(30)
    t = threading.Thread(target=beat, daemon=True)
    t.start()
    return fn, t

# ------------------------------------------------------------
# LOCKING & QUEUE
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
                if time.time() - start > timeout: return False
                time.sleep(0.5)
    def release(self):
        if self.fd: os.close(self.fd); self.fd = None
        if os.path.exists(self.lock_path): os.remove(self.lock_path)

def acquire_file_lock(file_stem):
    os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
    lock = SimpleLock(os.path.join(DEST_EMBED_FOLDER, f"{file_stem}.lock"))
    return lock if lock.acquire() else None

def release_file_lock(lock):
    if lock: lock.release()

def acquire_gpu_lock(gpu_id):
    os.makedirs(GPU_LOCK_FOLDER, exist_ok=True)
    lf = os.path.join(GPU_LOCK_FOLDER, f"gpu_{gpu_id}.lock")
    if not os.path.exists(lf):
        with open(lf, 'w') as f: f.write(f"{os.getpid()}")
        return lf
    return None

def release_gpu_lock(lock_file):
    if lock_file and os.path.exists(lock_file): os.remove(lock_file)

QUEUE_LOCK = TASK_QUEUE_FILE + '.lock'

def _with_queue_locked(fn):
    lock = SimpleLock(QUEUE_LOCK)
    if not lock.acquire(timeout=30): raise RuntimeError('Queue lock timeout')
    try:
        if os.path.exists(TASK_QUEUE_FILE):
            with open(TASK_QUEUE_FILE, 'r') as f: 
                try: data = json.load(f)
                except: data = {'pending': [], 'in_progress': {}, 'completed': [], 'failed': {}}
        else: data = {'pending': [], 'in_progress': {}, 'completed': [], 'failed': {}}
        
        for k in ['pending', 'in_progress', 'completed', 'failed']:
            if k not in data: data[k] = [] if k != 'in_progress' and k != 'failed' else {}
            
        res = fn(data)
        with open(TASK_QUEUE_FILE, 'w') as f: json.dump(data, f)
        return res
    finally: lock.release()

def reconcile_queue(all_file_paths):
    def _recon(q):
        print("DEBUG: Reconciling queue...")
        # Note: We do NOT remove completed items blindly here anymore
        # because the worker will verify them on the fly.
        
        known = set(q['pending']) | set(q['completed']) | set(q['in_progress'].keys())
        added = 0
        for f in all_file_paths:
            if f not in known:
                q['pending'].append(f)
                added += 1
        print(f"DEBUG: Added {added} new files. Pending: {len(q['pending'])}")
    _with_queue_locked(_recon)

def claim_next_batch(batch_size, gpu_id):
    def _claim(q):
        claimed = []
        if not q['pending']: return claimed
        to_claim = q['pending'][:batch_size]
        q['pending'] = q['pending'][batch_size:]
        ts = int(time.time())
        for f in to_claim:
            q['in_progress'][f] = {'machine_id': MACHINE_ID, 'gpu_id': gpu_id, 'claimed_at': ts}
            claimed.append(f)
        return claimed
    return _with_queue_locked(_claim)

def mark_file_completed(file_path, success, error_msg=None):
    def _mark(q):
        q['in_progress'].pop(file_path, None)
        if success: 
            if file_path not in q['completed']:
                q['completed'].append(file_path)
        else: 
            q['failed'][file_path] = {'error': error_msg}
    _with_queue_locked(_mark)

def get_queue_status():
    def _status(q):
        return {k: len(v) for k, v in q.items()}
    return _with_queue_locked(_status)

# ------------------------------------------------------------
# PREFETCHER
# ------------------------------------------------------------
class DataPrefetcher:
    def __init__(self, file_paths, max_prefetch=2):
        self.file_paths = file_paths
        self.q = queue.Queue(maxsize=max_prefetch)
        self.stop_event = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        for fp in self.file_paths:
            if self.stop_event.is_set(): break
            try:
                df = pd.read_parquet(fp, columns=['title', 'abstract'])
                # Pre-construct texts here to save GPU time later
                texts = (QUERY_PREFIX + df['title'].fillna('') + ". " + df['abstract'].fillna('')).tolist()
                self.q.put((fp, df, texts))
            except Exception as e:
                print(f"Error prefetching {fp}: {e}")
                self.q.put((fp, None, None))
        self.q.put((None, None, None))

    def next(self): return self.q.get()
    def stop(self):
        self.stop_event.set()
        if self.t.is_alive(): self.t.join(timeout=1)

# ------------------------------------------------------------
# PROCESSOR (GPU)
# ------------------------------------------------------------
@torch.inference_mode()
def process_embeddings_gpu(gpu_id, file_path, df, texts, model, progress_file):
    stem = robust_basename(file_path)
    out_path = os.path.join(DEST_EMBED_FOLDER, f"{stem}.npy")

    # --- INTEGRITY CHECK START ---
    if os.path.exists(out_path):
        # File exists, check if it is valid before skipping
        if check_file_integrity(out_path, df, model, samples=5):
            print(f"[GPU {gpu_id}] SKIPPING {stem}: Already exists and passed integrity check.")
            mark_file_completed(file_path, True)
            return True
        else:
             print(f"[GPU {gpu_id}] RECALCULATING {stem}: Exists but failed integrity check.")
    # --- INTEGRITY CHECK END ---

    lock = acquire_file_lock(stem)
    if not lock: return False

    try:
        start = time.time()
        if not texts:
            empty = np.empty((0, OUTPUT_BYTES), dtype=np.uint8)
            np.save(out_path, empty)
            mark_file_completed(file_path, True)
            return True

        total_docs = len(texts)
        results = []
        
        # CHUNKED PROCESSING LOOP
        for i in range(0, total_docs, REPORT_CHUNK_SIZE):
            chunk = texts[i : i + REPORT_CHUNK_SIZE]
            
            # Dynamic OOM fallback
            bs = BATCH_SIZE
            while True:
                try:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        embs_tensor = model.encode(
                            chunk,
                            batch_size=bs,
                            show_progress_bar=False,
                            normalize_embeddings=True,
                            convert_to_tensor=True
                        )
                    chunk_packed = quantize_and_pack_binary(embs_tensor)
                    results.append(chunk_packed)
                    break 
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and bs > 4:
                        bs = bs // 2
                        print(f"[GPU {gpu_id}] OOM. Reduced batch to {bs}...")
                        torch.cuda.empty_cache()
                        continue
                    raise

            # Print Progress
            progress_pct = min(100.0, ((i + len(chunk)) / total_docs) * 100)
            if i > 0:
                print(f"[GPU {gpu_id}] {stem}: {progress_pct:.1f}% ({i + len(chunk)}/{total_docs})")

        # Combine
        final_packed = np.vstack(results)
        np.save(out_path, final_packed)
        
        dur = time.time() - start
        ips = len(texts) / dur if dur > 0 else 0
        
        print(f"[GPU {gpu_id}] DONE {stem}: {len(texts)} docs in {dur:.2f}s ({ips:.0f} docs/s)")
        
        with open(progress_file, 'a') as f:
            f.write(f"{stem},{len(texts)},{dur:.2f},{ips:.2f}\n")
            
        mark_file_completed(file_path, True)
        return True

    except Exception as e:
        print(f"[GPU {gpu_id}] Error processing {file_path}: {e}")
        mark_file_completed(file_path, False, str(e))
        return False
    finally:
        release_file_lock(lock)

def process_embeddings_worker(gpu_id, batch_size=1):
    lock_file = acquire_gpu_lock(gpu_id)
    if not lock_file: return

    try:
        os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
        prog_file = os.path.join(DEST_EMBED_FOLDER, f"progress_{HOSTNAME}_gpu_{gpu_id}.csv")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        print(f"[GPU {gpu_id}] Loading BGE Model...")
        model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
        model.to("cuda")
        
        # Optimization: Compile model if possible
        try:
            model = torch.compile(model)
        except: pass

        prefetcher = None
        while True:
            # Claim files
            files = claim_next_batch(10, gpu_id) 
            if not files: break
            
            print(f"[GPU {gpu_id}] Claimed batch of {len(files)} files")
            prefetcher = DataPrefetcher(files, max_prefetch=PREFETCH_SIZE)
            
            while True:
                item = prefetcher.next()
                if item is None: break
                fp, df, texts = item
                if fp is None: break
                
                # Process (includes integrity check now)
                process_embeddings_gpu(gpu_id, fp, df, texts, model, prog_file)
                
                df = None
                texts = None
                
            prefetcher.stop()
            gc.collect()

    finally:
        release_gpu_lock(lock_file)
        print(f"[GPU {gpu_id}] Worker finished.")

def monitor_progress(stop_event, file_paths):
    while not stop_event.is_set():
        time.sleep(10)
        try:
            status = get_queue_status()
            print(f"\n--- Queue Status: {status} ---")
            reconcile_queue(file_paths)
        except: pass

def main_parallel_embeddings(input_directory):
    ensure_dirs()
    machine_file, _ = register_machine()

    file_paths = sorted(glob.glob(os.path.join(input_directory, '*.parquet')))
    print(f"Found {len(file_paths)} Parquet files.")
    
    reconcile_queue(file_paths)

    if AVAILABLE_GPUS == 0:
        print("No GPUs available.")
        return 

    stop_evt = threading.Event()
    mon = threading.Thread(target=monitor_progress, args=(stop_evt, file_paths), daemon=True)
    mon.start()

    start = time.time()
    procs = []
    for gid in range(AVAILABLE_GPUS):
        p = multiprocessing.Process(target=process_embeddings_worker, args=(gid, 1))
        p.start()
        procs.append(p)

    for p in procs: p.join()

    stop_evt.set()
    mon.join()
    print(f"Total time: {time.time() - start:.2f}s")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    # Clean stale locks
    if os.path.exists(GPU_LOCK_FOLDER):
        for lf in glob.glob(os.path.join(GPU_LOCK_FOLDER, '*.lock')):
            try: os.remove(lf)
            except: pass
            
    main_parallel_embeddings(DEST_DF_FOLDER)