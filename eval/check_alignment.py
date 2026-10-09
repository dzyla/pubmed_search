"""
Row-alignment scan for single-file sources (bioRxiv / medRxiv): re-embeds every
parquet row ("title. abstract", no prefix) and compares with the stored code at
the same row. Reports misaligned row ranges and, for each, the offset at which
the stored code matches the parquet. Read-only unless --fix, which replaces the
codes of misaligned rows with fresh embeddings (atomic write).

    python eval/check_alignment.py <codes.npy> <metadata.parquet> [--fix]
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("npy")
ap.add_argument("parquet")
ap.add_argument("--fix", action="store_true")
ap.add_argument("--threshold", type=float, default=0.90)
args = ap.parse_args()

stored = np.load(args.npy)
df = pd.read_parquet(args.parquet, columns=["title", "abstract"])
assert len(df) == len(stored), (len(df), len(stored))
texts = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu",
                            model_kwargs={"dtype": torch.float16} if torch.cuda.is_available() else {})
fresh = np.packbits(model.encode(texts, batch_size=512, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False) > 0, axis=1)
agree = 1 - np.unpackbits(fresh ^ stored, axis=1).mean(axis=1)
bad = np.flatnonzero(agree < args.threshold)
print(f"{os.path.basename(args.npy)}: {len(stored):,} rows, median agreement {np.median(agree):.4f}, "
      f"misaligned rows: {len(bad):,}")
if len(bad):
    breaks = np.flatnonzero(np.diff(bad) > 1)
    starts, ends = np.r_[bad[0], bad[breaks + 1]], np.r_[bad[breaks], bad[-1]]
    for s, e in zip(starts, ends):
        r = (s + e) // 2
        best = max(range(-60, 61), key=lambda k: 1 - np.unpackbits(fresh[min(max(r + k, 0), len(fresh) - 1)]
                                                                     ^ stored[r]).mean())
        print(f"  rows {s:,}-{e:,} ({e - s + 1:,}): stored code matches parquet row r{best:+d}")
    if args.fix:
        stored[bad] = fresh[bad]
        tmp = args.npy + ".tmp.npy"
        np.save(tmp, stored)
        os.replace(tmp, args.npy)
        print(f"  fixed: re-embedded {len(bad):,} rows in place")
