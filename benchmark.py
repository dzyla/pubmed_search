import os
import json
import time
import yaml
import logging
import psutil
import numpy as np
import faiss
import gc
import concurrent.futures
from pathlib import Path

# Setup simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("Benchmark")

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------

def get_memory_usage_mb():
    """Returns current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_real_config(config_path="config_mss.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_random_packed_query(embedding_dim=768):
    """Generates a random packed uint8 query vector for testing IO/Search speed."""
    # 768 floats -> binary -> packed
    # For simulation, we just create random bytes of correct length
    n_bytes = embedding_dim // 8
    return np.random.randint(0, 255, size=(1, n_bytes), dtype=np.uint8)

# -----------------------------------------------------------------------------
# CORE LOGIC EXTRACTED (Modified for Benchmarking)
# -----------------------------------------------------------------------------

def copy_file_into_chunk(file_info, memmap_array, offset):
    """Helper for regeneration."""
    try:
        arr = np.load(file_info["path"], mmap_mode="r", allow_pickle=True)
        rows = file_info["rows"]
        memmap_array[offset : offset + rows] = arr[:rows]
        return {
            "source_stem": file_info["stem"],
            "chunk_local_start": offset,
            "chunk_local_end": offset + rows,
            "actual_rows": rows # Added for direct access
        }
    except Exception:
        return None

class BenchmarkEngine:
    def __init__(self, config_section, source_name):
        self.name = source_name
        self.config = config_section
        self.chunk_dir = self.config["chunk_dir"]
        self.metadata_path = self.config["metadata_path"]
        self.embeddings_dir = self.config["embeddings_directory"]
        self.npy_pattern = self.config["npy_files_pattern"]
        
        # Load initial metadata to get embedding dim (needed for regeneration)
        # If metadata doesn't exist, we assume a default or scan one file
        self.embedding_dim = self._detect_embedding_dim()

    def _detect_embedding_dim(self):
        # Try reading metadata first
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f).get("embedding_dim", 768 // 8) # Default bytes
            except:
                pass
        
        # Fallback: scan first file
        files = sorted(Path(self.embeddings_dir).glob(self.npy_pattern))
        if files:
            arr = np.load(files[0], mmap_mode="r", allow_pickle=True)
            if arr.ndim > 1:
                return arr.shape[1]
        return 96 # Default for 768-bit binary

    def regenerate_chunks(self, target_chunk_size_bytes):
        """Regenerates chunks with a specific size."""
        LOGGER.info(f"--- Regenerating {self.name} chunks with size {target_chunk_size_bytes/1024**3:.2f} GB ---")
        start_mem = get_memory_usage_mb()
        start_time = time.time()
        
        os.makedirs(self.chunk_dir, exist_ok=True)
        
        # 1. Scan Source
        source_files = sorted(Path(self.embeddings_dir).glob(self.npy_pattern))
        file_infos = []
        total_rows = 0
        
        for p in source_files:
            try:
                arr = np.load(p, mmap_mode="r", allow_pickle=True)
                rows = arr.shape[0]
                dims = arr.shape[1] if arr.ndim > 1 else 1
                if dims != self.embedding_dim: continue
                file_infos.append({"stem": p.stem, "path": str(p), "rows": rows})
                total_rows += rows
            except: pass

        if not file_infos:
            LOGGER.error("No source files found.")
            return False

        # 2. Group
        rows_per_chunk = target_chunk_size_bytes // (self.embedding_dim * 1)
        chunks_metadata = []
        current_chunk_files = []
        current_chunk_rows = 0
        chunk_idx = 0
        global_start = 0

        def flush_chunk(files, rows, idx, g_start):
            fname = f"chunk_{idx}.npy"
            fpath = os.path.join(self.chunk_dir, fname)
            
            # Create memmap
            mm = np.memmap(fpath, dtype=np.uint8, mode='w+', shape=(rows, self.embedding_dim))
            
            # Copy
            parts = []
            offset = 0
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for fi in files:
                    futures.append(executor.submit(copy_file_into_chunk, fi, mm, offset))
                    offset += fi["rows"]
                for fut in concurrent.futures.as_completed(futures):
                    if fut.result(): parts.append(fut.result())
            
            mm.flush()
            del mm
            gc.collect()
            
            parts.sort(key=lambda x: x["chunk_local_start"])
            return {
                "chunk_file": fname,
                "global_start": g_start,
                "actual_rows": rows,
                "parts": parts
            }

        for fi in file_infos:
            if current_chunk_rows + fi["rows"] > rows_per_chunk:
                if current_chunk_files:
                    meta = flush_chunk(current_chunk_files, current_chunk_rows, chunk_idx, global_start)
                    chunks_metadata.append(meta)
                    global_start += current_chunk_rows
                    chunk_idx += 1
                    current_chunk_files = []
                    current_chunk_rows = 0
            current_chunk_files.append(fi)
            current_chunk_rows += fi["rows"]

        if current_chunk_files:
            meta = flush_chunk(current_chunk_files, current_chunk_rows, chunk_idx, global_start)
            chunks_metadata.append(meta)

        # Save Metadata
        final_meta = {"total_rows": total_rows, "embedding_dim": self.embedding_dim, "chunks": chunks_metadata}
        with open(self.metadata_path, 'w') as f:
            json.dump(final_meta, f, indent=4)
            
        dur = time.time() - start_time
        LOGGER.info(f"Regeneration complete. Time: {dur:.2f}s. Peak RAM overhead: {get_memory_usage_mb() - start_mem:.2f} MB")
        return True

    def search_chunk_task(self, args):
        """Worker function for search."""
        chunk_info, query_packed, dim = args
        chunk_path = os.path.join(self.chunk_dir, chunk_info["chunk_file"])
        rows = chunk_info["actual_rows"]
        
        if not os.path.exists(chunk_path): return 0
        
        try:
            # mmap
            data = np.memmap(chunk_path, dtype=np.uint8, mode='r', shape=(rows, dim))
            
            # faiss
            d_bits = dim * 8
            index = faiss.IndexBinaryFlat(d_bits)
            index.add(data)
            
            # search
            # We search for top 50 per chunk as benchmark
            D, I = index.search(query_packed, 50)
            
            del data
            del index
            return rows # Return processed rows count
        except Exception as e:
            print(f"Error: {e}")
            return 0

    def run_search_benchmark(self, mode, workers):
        with open(self.metadata_path, 'r') as f:
            meta = json.load(f)
        
        query = generate_random_packed_query(self.embedding_dim * 8)
        chunks = meta["chunks"]
        
        start_time = time.time()
        start_mem = get_memory_usage_mb()
        
        total_rows_processed = 0
        
        if mode == "sequential":
            for chunk in chunks:
                total_rows_processed += self.search_chunk_task((chunk, query, self.embedding_dim))
                gc.collect()
        
        elif mode == "parallel":
            # Prepare args
            tasks = [(c, query, self.embedding_dim) for c in chunks]
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(self.search_chunk_task, tasks))
                total_rows_processed = sum(results)
                
        end_time = time.time()
        end_mem = get_memory_usage_mb()
        
        return {
            "time": end_time - start_time,
            "mem_peak_delta": end_mem - start_mem,
            "rows": total_rows_processed
        }

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Configuration
    config = load_real_config()
    
    # Select one database to test optimization on (e.g., PubMed is usually largest)
    # You can change this to "biorxiv_config" etc.
    TARGET_DB = "pubmed_config" 
    DB_NAME = "PubMed"
    
    if TARGET_DB not in config:
        print(f"Config for {TARGET_DB} not found.")
        exit()
        
    engine = BenchmarkEngine(config[TARGET_DB], DB_NAME)
    
    # 2. Define Test Parameters
    # Sizes to test: 256MB, 512MB, 1GB, 2GB
    CHUNK_SIZES = [
        128 * 1024 * 1024,
        256 * 1024 * 1024,
        512 * 1024 * 1024,
        1024 * 1024 * 1024, # 1GB (Current default)
        2048 * 1024 * 1024,
        4096 * 1024 * 1024
    ]
    
    CPU_COUNT = os.cpu_count() or 4
    WORKER_COUNTS = [1, 4] # Test Sequential vs Max CPU
    
    results_table = []

    print(f"\n{'='*60}")
    print(f"BENCHMARK START: {DB_NAME}")
    print(f"System CPUs: {CPU_COUNT}")
    print(f"Embedding Dim (Bytes): {engine.embedding_dim}")
    print(f"{'='*60}\n")

    for size in CHUNK_SIZES:
        size_gb = size / (1024**3)
        print(f"\n>>> TESTING CHUNK SIZE: {size_gb:.2f} GB")
        
        # A. Regenerate Chunks
        success = engine.regenerate_chunks(size)
        if not success:
            print("Regeneration failed. Skipping.")
            continue
            
        # B. Run Search Benchmarks
        for workers in WORKER_COUNTS:
            mode = "sequential" if workers == 1 else "parallel"
            mode_label = f"Seq (1 CPU)" if workers == 1 else f"Par ({workers} CPUs)"
            
            print(f"   Running {mode_label} search...")
            
            # Run multiple times for stability? Just once for now.
            # Force GC before run
            gc.collect()
            time.sleep(1) 
            
            res = engine.run_search_benchmark(mode, workers)
            
            print(f"      -> Time: {res['time']:.4f}s | Mem Delta: {res['mem_peak_delta']:.2f} MB")
            
            results_table.append({
                "Chunk Size": f"{size_gb:.2f} GB",
                "Mode": mode_label,
                "Time (s)": res['time'],
                "Mem Delta (MB)": res['mem_peak_delta'],
                "Speed (Rows/s)": int(res['rows'] / res['time']) if res['time'] > 0 else 0
            })

    # 3. Report
    print(f"\n\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Chunk Size':<15} | {'Mode':<15} | {'Time (s)':<10} | {'Mem (MB)':<10} | {'Speed (Rows/s)':<15}")
    print("-" * 75)
    
    for r in results_table:
        print(f"{r['Chunk Size']:<15} | {r['Mode']:<15} | {r['Time (s)']:<10.4f} | {r['Mem Delta (MB)']:<10.2f} | {r['Speed (Rows/s)']:<15,}")
    print("-" * 75)