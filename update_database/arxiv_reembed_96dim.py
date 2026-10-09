"""
arXiv re-embedding script — fixes 96-dim source files.

Run this on the GPU workstation that has access to both the embed files and the
parquet data.  It finds every .npy file with 96 bytes/vector in EMBED_DIR,
re-encodes the corresponding text with BGE-small (48 bytes/vector), and writes
a replacement .npy file in place.

After this script finishes and the new files are visible on the server (via the
shared mount or scp), the search service will detect the changed mtimes on next
startup / update-check and automatically rebuild the arXiv FAISS index.

Usage:
    python arxiv_reembed_96dim.py
    python arxiv_reembed_96dim.py --dry-run        # list files only, no writes
    python arxiv_reembed_96dim.py --force          # recompute even already-48dim files
"""
import os
import sys
import glob
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# PATHS  — adjust if your mount point differs
# ---------------------------------------------------------------------------
EMBED_DIR  = "/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/arxiv_embed/"
DATA_DIR   = "/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake/arxiv_df/"

# ---------------------------------------------------------------------------
# MODEL CONFIG  — must match the server's model_api.py
# ---------------------------------------------------------------------------
MODEL_ID     = "BAAI/bge-small-en-v1.5"
TARGET_BYTES = 48        # 384-dim floats → 48 packed uint8 bytes
BATCH_SIZE   = 1024
EXPECTED_WRONG_DIM = 96  # source files to reprocess


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Re-embed arXiv 96-dim files with BGE-small (48-dim)")
    p.add_argument("--dry-run", action="store_true", help="List files to fix, do not write")
    p.add_argument("--force",   action="store_true", help="Reprocess even already-48dim files")
    p.add_argument("--embed-dir", default=EMBED_DIR, help="Path to arxiv embed directory")
    p.add_argument("--data-dir",  default=DATA_DIR,  help="Path to arxiv parquet directory")
    return p.parse_args()


def build_texts(df: pd.DataFrame) -> list:
    return (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()


def quantize_and_pack(embeddings) -> np.ndarray:
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().float().numpy()
    return np.packbits(embeddings > 0, axis=1)


def embed_texts(model, texts: list) -> np.ndarray:
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        with torch.no_grad():
            embs = model.encode(
                batch,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        results.append(quantize_and_pack(embs))
        pct = min(100.0, (i + len(batch)) / len(texts) * 100)
        if i > 0 and i % (BATCH_SIZE * 5) == 0:
            print(f"    {pct:.1f}%  ({i + len(batch):,}/{len(texts):,})", flush=True)
    return np.vstack(results)


def find_wrong_dim_files(embed_dir: str, force: bool) -> list:
    """Return list of (npy_path, dim) for files that need reprocessing."""
    to_fix = []
    all_npy = sorted(glob.glob(os.path.join(embed_dir, "*.npy")))
    print(f"Scanning {len(all_npy)} .npy files in {embed_dir} …")
    for path in all_npy:
        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=True)
            dim = arr.shape[1] if arr.ndim == 2 else -1
            if force or dim == EXPECTED_WRONG_DIM:
                to_fix.append((path, dim))
        except Exception as e:
            print(f"  [WARN] Cannot read {os.path.basename(path)}: {e}")
    return to_fix


def main():
    args = parse_args()
    embed_dir = args.embed_dir.rstrip("/") + "/"
    data_dir  = args.data_dir.rstrip("/") + "/"

    wrong_files = find_wrong_dim_files(embed_dir, args.force)

    if not wrong_files:
        print("No files need reprocessing. All source files already have the correct dimension.")
        return

    print(f"\nFiles to reprocess: {len(wrong_files)}")
    missing_parquet = []
    to_process = []

    for npy_path, dim in wrong_files:
        stem = Path(npy_path).stem
        parquet_path = os.path.join(data_dir, f"{stem}.parquet")
        if not os.path.exists(parquet_path):
            missing_parquet.append((stem, npy_path))
        else:
            to_process.append((stem, npy_path, parquet_path, dim))

    if missing_parquet:
        print(f"\n[WARNING] {len(missing_parquet)} file(s) have no matching parquet — "
              "cannot recover text for re-embedding:")
        for stem, _ in missing_parquet[:10]:
            print(f"  {stem}.parquet not found in {data_dir}")
        if len(missing_parquet) > 10:
            print(f"  … and {len(missing_parquet) - 10} more")
        print("These papers will be excluded from arXiv search until their parquet data "
              "is available and the files are re-embedded.")

    print(f"\nFiles ready to re-embed: {len(to_process)}")
    total_rows = 0
    for stem, _, pq, dim in to_process:
        df = pd.read_parquet(pq, columns=["title"])
        total_rows += len(df)
        print(f"  {stem}.npy  ({dim}-dim → {TARGET_BYTES}-dim)  {len(df):,} rows")

    if args.dry_run:
        print(f"\nDry run — no files written. Total rows that would be re-embedded: {total_rows:,}")
        return

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {MODEL_ID} on {device} …")
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    model.to(device)
    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile.")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Re-embed
    # ------------------------------------------------------------------
    t0 = time.time()
    success, failed = 0, 0

    for stem, npy_path, parquet_path, old_dim in to_process:
        print(f"\n[{success + failed + 1}/{len(to_process)}] Re-embedding {stem} "
              f"({old_dim}-dim → {TARGET_BYTES}-dim) …", flush=True)
        try:
            df = pd.read_parquet(parquet_path)
            if "title" not in df.columns or "abstract" not in df.columns:
                print(f"  [SKIP] Missing title/abstract columns in {stem}.parquet")
                failed += 1
                continue

            texts = build_texts(df)
            if not texts:
                print(f"  [SKIP] No text rows in {stem}.parquet")
                failed += 1
                continue

            t1 = time.time()
            new_embeddings = embed_texts(model, texts)
            dur = time.time() - t1

            assert new_embeddings.shape == (len(df), TARGET_BYTES), (
                f"Shape mismatch: got {new_embeddings.shape}, expected ({len(df)}, {TARGET_BYTES})"
            )

            # Write atomically via temp file
            tmp_path = npy_path + ".tmp"
            np.save(tmp_path, new_embeddings)
            os.replace(tmp_path, npy_path)

            print(f"  Done: {len(df):,} rows in {dur:.1f}s  ({len(df)/dur:.0f} rows/s)")
            success += 1

        except Exception as e:
            import traceback
            print(f"  [ERROR] {stem}: {e}")
            traceback.print_exc()
            failed += 1

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Re-embedding complete in {total_time:.1f}s")
    print(f"  Success : {success}")
    print(f"  Failed  : {failed}")
    print(f"  Skipped (no parquet): {len(missing_parquet)}")
    print(f"\nNext step: restart the search services on the server so they")
    print(f"detect the updated mtime on the re-embedded .npy files and")
    print(f"rebuild the arXiv FAISS index automatically.")


if __name__ == "__main__":
    main()
