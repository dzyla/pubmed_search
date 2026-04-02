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
import subprocess
import sys
import re
import argparse
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from unidecode import unidecode

# ------------------------------------------------------------
# PATHS & CONFIG
# ------------------------------------------------------------
DEST_DF_FOLDER    = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/arxiv_df/'
DEST_EMBED_FOLDER = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/arxiv_embed/'
COORDINATION_DIR  = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/arxiv_coordination/'

TASK_QUEUE_FILE  = os.path.join(COORDINATION_DIR, 'task_queue.json')
MACHINE_STATUS_DIR = os.path.join(COORDINATION_DIR, 'machine_status/')
GPU_LOCK_FOLDER  = f"gpu_locks/{socket.gethostname()}/"

KAGGLE_DATASET_DIR = '/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/arxiv_temp/kaggle_arxiv'
KAGGLE_JSON_FILE   = os.path.join(KAGGLE_DATASET_DIR, 'arxiv-metadata-oai-snapshot.json')

HOSTNAME   = socket.gethostname()
MACHINE_ID = str(uuid.uuid4())[:8]

# --- TUNABLES ---
BATCH_SIZE        = 1024
PREFETCH_SIZE     = 4
SAVE_INTERVAL     = 1
LOCK_TIMEOUT      = 60 * 25
INPROGRESS_MAX_AGE = 30 * 60

REPORT_CHUNK_SIZE = 5000
CHUNK_SIZE_ROWS   = 25000   # rows per output parquet file

# BGE — matches BioRxiv/MedRxiv convention (no query prefix on documents)
MODEL_ID     = "BAAI/bge-small-en-v1.5"
OUTPUT_BYTES = 48            # 384-dim / 8 = 48 bytes packed binary

try:
    AVAILABLE_GPUS = torch.cuda.device_count()
    print(f"Detected {AVAILABLE_GPUS} available GPUs on {HOSTNAME}")
except Exception:
    AVAILABLE_GPUS = 0
    print(f"No GPUs detected on {HOSTNAME}")

# ------------------------------------------------------------
# ARGUMENT PARSING
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="arXiv pipeline: download → parse → embed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
─────
  update     Full pipeline: download snapshot → parse new docs → embed new/missing  [default]
  reprocess  JSON already present → (re)parse into parquets → embed all
             Use --force to wipe existing parquets and rebuild from scratch.
  embed      Parquets already present → (re)embed all from scratch.
             Use --force to bypass integrity checks and always recompute.
  health     Spot-check random parquet+embedding pairs; no writes.

Examples
────────
  python arxiv_download_embed_update.py                          # update (default)
  python arxiv_download_embed_update.py --mode reprocess         # parse from existing JSON, embed new
  python arxiv_download_embed_update.py --mode reprocess --force # wipe parquets, full reparse + embed
  python arxiv_download_embed_update.py --mode embed             # embed only (skip bad ones)
  python arxiv_download_embed_update.py --mode embed --force     # force-recompute all embeddings
  python arxiv_download_embed_update.py --mode health            # quality check, no writes
  python arxiv_download_embed_update.py --mode health --samples 50 --rows-per-file 10
        """,
    )
    p.add_argument(
        "--mode",
        choices=["update", "reprocess", "embed", "health"],
        default="update",
        help="Pipeline stage to run (default: update)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "reprocess: wipe parquets before reparsing. "
            "embed: bypass integrity checks and recompute every file."
        ),
    )
    p.add_argument(
        "--samples",
        type=int,
        default=20,
        help="health: number of files to spot-check (default: 20)",
    )
    p.add_argument(
        "--rows-per-file",
        type=int,
        default=5,
        dest="rows_per_file",
        help="health: rows sampled per file for similarity test (default: 5)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.96,
        help="Hamming similarity threshold for integrity checks (default: 0.96)",
    )
    return p.parse_args()


# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def robust_basename(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0]

def ensure_dirs():
    for d in [DEST_DF_FOLDER, DEST_EMBED_FOLDER, COORDINATION_DIR,
              GPU_LOCK_FOLDER, MACHINE_STATUS_DIR, KAGGLE_DATASET_DIR]:
        os.makedirs(d, exist_ok=True)

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = unidecode(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_texts(df):
    """Construct embedding input: title + abstract, no query prefix (BGE doc convention)."""
    return (df['title'].fillna('') + ". " + df['abstract'].fillna('')).tolist()


# ------------------------------------------------------------
# QUANTIZATION
# ------------------------------------------------------------
def quantize_and_pack_binary(embeddings):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().float().numpy()
    return np.packbits(embeddings > 0, axis=1)

def calculate_hamming_similarity(bits_a, bits_b):
    xor_diff = np.bitwise_xor(bits_a, bits_b)
    diff_bits = np.unpackbits(xor_diff).sum()
    total_bits = len(bits_a) * 8
    return (total_bits - diff_bits) / total_bits


# ------------------------------------------------------------
# INTEGRITY CHECKS
# ------------------------------------------------------------
def check_file_integrity(npy_path, df, model, samples=5, threshold=0.96):
    """Returns True if the .npy file is valid and matches the current model."""
    try:
        saved_emb = np.load(npy_path)

        if len(saved_emb) != len(df):
            print(f"Integrity Fail: Count mismatch (saved={len(saved_emb)}, df={len(df)})")
            return False

        if len(df) == 0:
            return True

        if saved_emb.ndim != 2 or saved_emb.shape[1] != OUTPUT_BYTES:
            print(
                f"Integrity Fail: Width mismatch "
                f"(saved={saved_emb.shape[1] if saved_emb.ndim == 2 else '?'} bytes, "
                f"expected={OUTPUT_BYTES} bytes). Will recompute."
            )
            return False

        indices = random.sample(range(len(df)), min(samples, len(df)))
        texts = build_texts(df.iloc[indices])
        with torch.no_grad():
            fresh = model.encode(texts, batch_size=len(texts), show_progress_bar=False,
                                 normalize_embeddings=True, convert_to_numpy=True)
        fresh_packed = quantize_and_pack_binary(fresh)
        saved_subset = saved_emb[indices]

        for i in range(len(indices)):
            sim = calculate_hamming_similarity(fresh_packed[i], saved_subset[i])
            if sim < threshold:
                print(f"Integrity Fail: Low similarity {sim:.2%} at index {indices[i]}")
                return False

        return True

    except Exception as e:
        print(f"Integrity Check Error: {e}")
        return False


def check_file_integrity_detailed(npy_path, df, model, samples=5, threshold=0.96):
    """
    Like check_file_integrity but returns (passed: bool, avg_similarity: float | str).
    Used by health check to report similarity distributions.
    """
    try:
        saved_emb = np.load(npy_path)

        if len(saved_emb) != len(df):
            return False, f"count mismatch (saved={len(saved_emb)}, df={len(df)})"

        if len(df) == 0:
            return True, 1.0

        if saved_emb.ndim != 2 or saved_emb.shape[1] != OUTPUT_BYTES:
            return False, (
                f"width mismatch "
                f"(saved={saved_emb.shape[1] if saved_emb.ndim == 2 else '?'}b "
                f"expected={OUTPUT_BYTES}b)"
            )

        indices = random.sample(range(len(df)), min(samples, len(df)))
        texts = build_texts(df.iloc[indices])
        with torch.no_grad():
            fresh = model.encode(texts, batch_size=len(texts), show_progress_bar=False,
                                 normalize_embeddings=True, convert_to_numpy=True)
        fresh_packed = quantize_and_pack_binary(fresh)
        saved_subset = saved_emb[indices]

        sims = [calculate_hamming_similarity(fresh_packed[i], saved_subset[i])
                for i in range(len(indices))]
        avg_sim = float(np.mean(sims))
        passed = all(s >= threshold for s in sims)
        return passed, avg_sim

    except Exception as e:
        return False, f"error: {e}"


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------
def run_health_check(args):
    parquet_files = sorted(glob.glob(os.path.join(DEST_DF_FOLDER, '*.parquet')))
    npy_stems     = {robust_basename(f)
                     for f in glob.glob(os.path.join(DEST_EMBED_FOLDER, '*.npy'))}

    paired           = [f for f in parquet_files if robust_basename(f) in npy_stems]
    missing_embeddings = [f for f in parquet_files if robust_basename(f) not in npy_stems]

    print(f"\n{'='*50}")
    print(f"  Health Check — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")
    print(f"  Parquet files:       {len(parquet_files):>8,}")
    print(f"  Embedding files:     {len(npy_stems):>8,}")
    print(f"  Paired:              {len(paired):>8,}")
    print(f"  Missing embeddings:  {len(missing_embeddings):>8,}")
    if missing_embeddings:
        print(f"\n  Files missing embeddings (first 5):")
        for f in missing_embeddings[:5]:
            print(f"    {os.path.basename(f)}")
        if len(missing_embeddings) > 5:
            print(f"    … and {len(missing_embeddings) - 5} more")

    if not paired:
        print("\n  No paired files to check. Exiting.")
        return

    sample_n = min(args.samples, len(paired))
    sample_files = random.sample(paired, sample_n)

    print(f"\n  Spot-checking {sample_n} files, {args.rows_per_file} rows each "
          f"(threshold={args.threshold:.2f})")
    print(f"  Loading model: {MODEL_ID} ...")
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)

    passed_files, failed_files = [], []
    similarities = []

    for fp in tqdm(sample_files, desc="  Checking", ncols=80):
        stem    = robust_basename(fp)
        npy_path = os.path.join(DEST_EMBED_FOLDER, f"{stem}.npy")
        try:
            df = pd.read_parquet(fp)
            ok, sim = check_file_integrity_detailed(
                npy_path, df, model,
                samples=args.rows_per_file,
                threshold=args.threshold,
            )
            if ok:
                passed_files.append(stem)
                if isinstance(sim, float):
                    similarities.append(sim)
            else:
                failed_files.append((stem, sim))
        except Exception as e:
            failed_files.append((stem, f"load error: {e}"))

    print(f"\n{'='*50}")
    print(f"  Results")
    print(f"{'='*50}")
    print(f"  Passed: {len(passed_files)}/{sample_n} "
          f"({100 * len(passed_files) / sample_n:.1f}%)")
    if similarities:
        print(f"  Similarity — avg={np.mean(similarities):.4f}  "
              f"min={np.min(similarities):.4f}  max={np.max(similarities):.4f}")
    if failed_files:
        print(f"\n  Failed files ({len(failed_files)}):")
        for stem, reason in failed_files:
            print(f"    {stem}: {reason}")
        print(f"\n  Recommendation: run --mode embed to recompute failed files.")
    else:
        print(f"\n  All checked files passed.")


# ------------------------------------------------------------
# ARXIV DATA DOWNLOAD & PARSING
# ------------------------------------------------------------
def is_kaggle_installed():
    try:
        subprocess.run(['kaggle', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False

def download_kaggle_dataset():
    if not is_kaggle_installed():
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])
    print("Checking for updates from Kaggle...")
    command = (f"kaggle datasets download -d Cornell-University/arxiv "
               f"-p {KAGGLE_DATASET_DIR} --unzip --force")
    try:
        subprocess.run(command, shell=True, check=True)
        print("Kaggle download/update complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")

def get_existing_ids(df_folder):
    files = sorted(glob.glob(os.path.join(df_folder, '*.parquet')))
    existing_ids = set()
    if not files:
        return existing_ids
    print(f"Scanning {len(files)} parquet files for existing IDs...")
    for f in tqdm(files, desc="Indexing IDs"):
        try:
            df = pd.read_parquet(f, columns=['id'])
            existing_ids.update(df['id'].astype(str).values)
        except Exception as e:
            print(f"Warning: could not read {f}: {e}")
    print(f"Found {len(existing_ids):,} existing documents.")
    return existing_ids

def save_chunk(rows, folder, run_id, chunk_num):
    df = pd.DataFrame(rows)
    df['doi']     = 'https://arxiv.org/abs/' + df['id'].astype(str)
    df['journal'] = df['categories'].fillna('arXiv')

    compression_dict = {col: 'ZSTD' for col in
                        ['abstract', 'title', 'authors', 'id', 'categories', 'doi', 'journal']}
    valid_comp = {k: v for k, v in compression_dict.items() if k in df.columns}

    filename = f"arxiv_update_{run_id}_chunk_{chunk_num}.parquet"
    path = os.path.join(folder, filename)
    df.to_parquet(path, engine='pyarrow', compression=valid_comp, index=False)
    print(f"Saved: {filename}")

def incremental_process_json(json_file, output_folder, existing_ids):
    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        return

    print("Streaming JSON for new documents...")
    new_rows, chunks_created, new_docs = [], 0, 0
    run_id = int(time.time())

    with open(json_file, 'r') as f:
        for line in tqdm(f, desc="Scanning JSON"):
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = str(doc.get('id', ''))
            if doc_id in existing_ids:
                continue

            new_rows.append({
                'id':         doc_id,
                'title':      clean_text(doc.get('title', '')),
                'abstract':   clean_text(doc.get('abstract', '')),
                'authors':    doc.get('authors', ''),
                'date':       doc.get('update_date', ''),
                'categories': doc.get('categories', ''),
            })
            new_docs += 1

            if len(new_rows) >= CHUNK_SIZE_ROWS:
                save_chunk(new_rows, output_folder, run_id, chunks_created)
                chunks_created += 1
                new_rows = []
                gc.collect()

    if new_rows:
        save_chunk(new_rows, output_folder, run_id, chunks_created)
        chunks_created += 1

    print(f"Done. New docs: {new_docs:,}. New files: {chunks_created}.")

def wipe_parquets(folder):
    files = glob.glob(os.path.join(folder, '*.parquet'))
    print(f"Wiping {len(files)} existing parquet files from {folder} …")
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"  Could not delete {f}: {e}")
    print("Wipe complete.")


# ------------------------------------------------------------
# MACHINE REGISTRY
# ------------------------------------------------------------
def register_machine():
    info = {
        'hostname': HOSTNAME, 'machine_id': MACHINE_ID,
        'available_gpus': AVAILABLE_GPUS, 'start_time': int(time.time()),
        'last_heartbeat': int(time.time()), 'status': 'active',
    }
    fn = os.path.join(MACHINE_STATUS_DIR, f"{HOSTNAME}_{MACHINE_ID}.json")
    with open(fn, 'w') as f:
        json.dump(info, f)

    def beat():
        while True:
            try:
                with open(fn, 'r') as rf: d = json.load(rf)
                d['last_heartbeat'] = int(time.time())
                with open(fn, 'w') as wf: json.dump(d, wf)
            except Exception:
                pass
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
                if time.time() - start > timeout:
                    return False
                time.sleep(0.5)

    def release(self):
        if self.fd:
            os.close(self.fd)
            self.fd = None
        if os.path.exists(self.lock_path):
            os.remove(self.lock_path)

def acquire_file_lock(file_stem):
    os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
    lock = SimpleLock(os.path.join(DEST_EMBED_FOLDER, f"{file_stem}.lock"))
    return lock if lock.acquire() else None

def release_file_lock(lock):
    if lock:
        lock.release()

def acquire_gpu_lock(gpu_id):
    os.makedirs(GPU_LOCK_FOLDER, exist_ok=True)
    lf = os.path.join(GPU_LOCK_FOLDER, f"gpu_{gpu_id}.lock")
    if not os.path.exists(lf):
        with open(lf, 'w') as f:
            f.write(str(os.getpid()))
        return lf
    return None

def release_gpu_lock(lock_file):
    if lock_file and os.path.exists(lock_file):
        os.remove(lock_file)

QUEUE_LOCK = TASK_QUEUE_FILE + '.lock'

def _with_queue_locked(fn):
    lock = SimpleLock(QUEUE_LOCK)
    if not lock.acquire(timeout=30):
        raise RuntimeError('Queue lock timeout')
    try:
        if os.path.exists(TASK_QUEUE_FILE):
            with open(TASK_QUEUE_FILE, 'r') as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = {}
        else:
            data = {}
        for k, default in [('pending', []), ('in_progress', {}),
                            ('completed', []), ('failed', {})]:
            if k not in data:
                data[k] = default
        res = fn(data)
        with open(TASK_QUEUE_FILE, 'w') as f:
            json.dump(data, f)
        return res
    finally:
        lock.release()

def reconcile_queue(all_file_paths):
    def _recon(q):
        known = set(q['pending']) | set(q['completed']) | set(q['in_progress'].keys())
        added = 0
        for f in all_file_paths:
            if f not in known:
                q['pending'].append(f)
                added += 1
        print(f"Queue: +{added} new files  |  pending={len(q['pending'])}")
    _with_queue_locked(_recon)

def reset_queue(all_file_paths=None):
    """
    Clear completed/failed/in_progress so every file is re-queued.
    Optionally replace the pending list with all_file_paths.
    """
    def _reset(q):
        cleared = len(q.get('completed', [])) + len(q.get('failed', {}))
        q['completed']  = []
        q['failed']     = {}
        q['in_progress'] = {}
        if all_file_paths is not None:
            q['pending'] = list(all_file_paths)
        print(f"Queue reset: cleared {cleared} completed/failed entries. "
              f"pending={len(q['pending'])}")
    _with_queue_locked(_reset)

def claim_next_batch(batch_size, gpu_id):
    def _claim(q):
        to_claim = q['pending'][:batch_size]
        q['pending'] = q['pending'][batch_size:]
        ts = int(time.time())
        for f in to_claim:
            q['in_progress'][f] = {'machine_id': MACHINE_ID, 'gpu_id': gpu_id, 'claimed_at': ts}
        return to_claim
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
            if self.stop_event.is_set():
                break
            try:
                df    = pd.read_parquet(fp, columns=['title', 'abstract'])
                texts = build_texts(df)
                self.q.put((fp, df, texts))
            except Exception as e:
                print(f"Prefetch error {fp}: {e}")
                self.q.put((fp, None, None))
        self.q.put((None, None, None))

    def next(self):
        return self.q.get()

    def stop(self):
        self.stop_event.set()
        if self.t.is_alive():
            self.t.join(timeout=1)


# ------------------------------------------------------------
# GPU EMBEDDING
# ------------------------------------------------------------
@torch.inference_mode()
def process_embeddings_gpu(gpu_id, file_path, df, texts, model, progress_file, force=False):
    stem     = robust_basename(file_path)
    out_path = os.path.join(DEST_EMBED_FOLDER, f"{stem}.npy")

    if not force and os.path.exists(out_path):
        if check_file_integrity(out_path, df, model, samples=5):
            print(f"[GPU {gpu_id}] SKIP {stem} (passed integrity check)")
            mark_file_completed(file_path, True)
            return True
        else:
            print(f"[GPU {gpu_id}] RECOMPUTE {stem} (failed integrity check)")
    elif force and os.path.exists(out_path):
        print(f"[GPU {gpu_id}] FORCE RECOMPUTE {stem}")

    lock = acquire_file_lock(stem)
    if not lock:
        return False

    try:
        if not texts:
            np.save(out_path, np.empty((0, OUTPUT_BYTES), dtype=np.uint8))
            mark_file_completed(file_path, True)
            return True

        total = len(texts)
        results = []

        for i in range(0, total, REPORT_CHUNK_SIZE):
            chunk = texts[i: i + REPORT_CHUNK_SIZE]
            bs = BATCH_SIZE
            while True:
                try:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        embs = model.encode(chunk, batch_size=bs, show_progress_bar=False,
                                            normalize_embeddings=True, convert_to_tensor=True)
                    results.append(quantize_and_pack_binary(embs))
                    break
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and bs > 4:
                        bs //= 2
                        print(f"[GPU {gpu_id}] OOM → batch={bs}")
                        torch.cuda.empty_cache()
                    else:
                        raise

            pct = min(100.0, (i + len(chunk)) / total * 100)
            if i > 0:
                print(f"[GPU {gpu_id}] {stem}: {pct:.1f}% ({i + len(chunk)}/{total})")

        final = np.vstack(results)
        np.save(out_path, final)

        dur = time.time() - _proc_start.get(file_path, time.time())
        ips = total / max(dur, 1e-6)
        print(f"[GPU {gpu_id}] DONE {stem}: {total} docs  {ips:.0f} docs/s")
        with open(progress_file, 'a') as pf:
            pf.write(f"{stem},{total},{dur:.2f},{ips:.2f}\n")

        mark_file_completed(file_path, True)
        return True

    except Exception as e:
        print(f"[GPU {gpu_id}] Error {file_path}: {e}")
        mark_file_completed(file_path, False, str(e))
        return False
    finally:
        release_file_lock(lock)

_proc_start: dict = {}   # lightweight per-file timing without threading overhead


def process_embeddings_worker(gpu_id, batch_size=1, force=False):
    lock_file = acquire_gpu_lock(gpu_id)
    if not lock_file:
        return

    try:
        os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
        prog_file = os.path.join(DEST_EMBED_FOLDER, f"progress_{HOSTNAME}_gpu_{gpu_id}.csv")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        print(f"[GPU {gpu_id}] Loading {MODEL_ID} ...")
        model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
        model.to("cuda")
        try:
            model = torch.compile(model)
        except Exception:
            pass

        while True:
            files = claim_next_batch(10, gpu_id)
            if not files:
                break

            print(f"[GPU {gpu_id}] Claimed {len(files)} files")
            prefetcher = DataPrefetcher(files, max_prefetch=PREFETCH_SIZE)

            while True:
                item = prefetcher.next()
                if item is None:
                    break
                fp, df, texts = item
                if fp is None:
                    break
                _proc_start[fp] = time.time()
                process_embeddings_gpu(gpu_id, fp, df, texts, model, prog_file, force=force)
                df = texts = None

            prefetcher.stop()
            gc.collect()

    finally:
        release_gpu_lock(lock_file)
        print(f"[GPU {gpu_id}] Worker done.")


def monitor_progress(stop_event, file_paths):
    while not stop_event.is_set():
        time.sleep(10)
        try:
            status = get_queue_status()
            print(f"--- Queue: {status} ---")
            reconcile_queue(file_paths)
        except Exception:
            pass


# ------------------------------------------------------------
# CPU FALLBACK
# ------------------------------------------------------------
def _run_cpu_worker(file_paths, force=False):
    os.makedirs(DEST_EMBED_FOLDER, exist_ok=True)
    prog_file = os.path.join(DEST_EMBED_FOLDER, f"progress_{HOSTNAME}_cpu.csv")

    print(f"[CPU] Loading {MODEL_ID} ...")
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)

    for fp in tqdm(file_paths, desc="[CPU] Embedding"):
        stem     = robust_basename(fp)
        out_path = os.path.join(DEST_EMBED_FOLDER, f"{stem}.npy")
        try:
            df    = pd.read_parquet(fp, columns=['title', 'abstract'])
            texts = build_texts(df)

            if not force and os.path.exists(out_path):
                full_df = pd.read_parquet(fp)
                if check_file_integrity(out_path, full_df, model, samples=3):
                    print(f"[CPU] SKIP {stem}")
                    continue
                print(f"[CPU] RECOMPUTE {stem}")
            elif force and os.path.exists(out_path):
                print(f"[CPU] FORCE RECOMPUTE {stem}")

            if not texts:
                np.save(out_path, np.empty((0, OUTPUT_BYTES), dtype=np.uint8))
                continue

            start   = time.time()
            results = []
            for i in range(0, len(texts), BATCH_SIZE):
                chunk = texts[i: i + BATCH_SIZE]
                with torch.no_grad():
                    embs = model.encode(chunk, batch_size=BATCH_SIZE,
                                        normalize_embeddings=True,
                                        show_progress_bar=False,
                                        convert_to_numpy=True)
                results.append(quantize_and_pack_binary(embs))

            final = np.vstack(results)
            np.save(out_path, final)
            dur = time.time() - start
            ips = len(texts) / max(dur, 1e-6)
            print(f"[CPU] DONE {stem}: {len(texts)} docs  {ips:.0f} docs/s")
            with open(prog_file, 'a') as pf:
                pf.write(f"{stem},{len(texts)},{dur:.2f},{ips:.2f}\n")
        except Exception as e:
            print(f"[CPU] Error {fp}: {e}")

    print("[CPU] Embedding complete.")


# ------------------------------------------------------------
# SHARED EMBED DISPATCH (GPU or CPU)
# ------------------------------------------------------------
def run_embedding_pipeline(file_paths, force=False):
    """Run embedding on file_paths using GPU workers or CPU fallback."""
    if AVAILABLE_GPUS == 0:
        print("No GPUs — falling back to CPU.")
        _run_cpu_worker(file_paths, force=force)
        return

    stop_evt = threading.Event()
    mon = threading.Thread(target=monitor_progress, args=(stop_evt, file_paths), daemon=True)
    mon.start()

    procs = []
    for gid in range(AVAILABLE_GPUS):
        p = multiprocessing.Process(target=process_embeddings_worker, args=(gid, 1, force))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    stop_evt.set()
    mon.join()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    args = parse_args()
    ensure_dirs()
    register_machine()

    t0 = time.time()

    # ── Mode: health ────────────────────────────────────────
    if args.mode == 'health':
        run_health_check(args)
        return

    # ── Mode: update (default) ──────────────────────────────
    if args.mode == 'update':
        print("=== Mode: UPDATE (download → parse new → embed new/missing) ===")
        download_kaggle_dataset()
        existing_ids = get_existing_ids(DEST_DF_FOLDER)
        incremental_process_json(KAGGLE_JSON_FILE, DEST_DF_FOLDER, existing_ids)
        del existing_ids
        gc.collect()

        # Clean up the large raw JSON to free disk space
        if os.path.exists(KAGGLE_JSON_FILE):
            try:
                os.remove(KAGGLE_JSON_FILE)
                print("Cleaned up raw JSON.")
            except Exception:
                pass

        file_paths = sorted(glob.glob(os.path.join(DEST_DF_FOLDER, '*.parquet')))
        print(f"Total parquet files: {len(file_paths)}")
        reconcile_queue(file_paths)
        run_embedding_pipeline(file_paths, force=False)

    # ── Mode: reprocess ─────────────────────────────────────
    elif args.mode == 'reprocess':
        print("=== Mode: REPROCESS (parse JSON → embed all) ===")
        if not os.path.exists(KAGGLE_JSON_FILE):
            print(f"JSON not found at {KAGGLE_JSON_FILE}")
            print("Download it first: --mode update  (or copy it manually)")
            return

        if args.force:
            print("--force: wiping existing parquets before full reparse.")
            wipe_parquets(DEST_DF_FOLDER)
            existing_ids = set()
        else:
            existing_ids = get_existing_ids(DEST_DF_FOLDER)

        incremental_process_json(KAGGLE_JSON_FILE, DEST_DF_FOLDER, existing_ids)
        del existing_ids
        gc.collect()

        file_paths = sorted(glob.glob(os.path.join(DEST_DF_FOLDER, '*.parquet')))
        print(f"Total parquet files: {len(file_paths)}")
        # Reset queue so reprocessed files are re-embedded
        reset_queue(file_paths)
        run_embedding_pipeline(file_paths, force=args.force)

    # ── Mode: embed ─────────────────────────────────────────
    elif args.mode == 'embed':
        print("=== Mode: EMBED (embed all existing parquets) ===")
        if args.force:
            print("--force: all files will be recomputed (integrity checks skipped).")

        file_paths = sorted(glob.glob(os.path.join(DEST_DF_FOLDER, '*.parquet')))
        if not file_paths:
            print(f"No parquet files found in {DEST_DF_FOLDER}")
            print("Parse first: --mode reprocess")
            return

        print(f"Total parquet files: {len(file_paths)}")
        reset_queue(file_paths)
        run_embedding_pipeline(file_paths, force=args.force)

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    # Clean stale GPU locks from previous crashed runs
    if os.path.exists(GPU_LOCK_FOLDER):
        for lf in glob.glob(os.path.join(GPU_LOCK_FOLDER, '*.lock')):
            try:
                os.remove(lf)
            except Exception:
                pass

    main()
