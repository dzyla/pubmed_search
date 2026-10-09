"""
Removes non-English rows from existing parquet + .npy chunk pairs, keeping
them row-aligned (same mask on both; atomic rewrite, parquet first).

    python tools/filter_language.py <df_dir> <embed_dir> [--dry-run]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from textlang import is_english  # noqa: E402


def filter_pair(parquet: str, npy: str, dry_run: bool = False) -> int:
    df = pd.read_parquet(parquet)
    bits = np.load(npy)
    if len(df) != len(bits):
        raise SystemExit(f"{parquet}: {len(df)} rows but {len(bits)} embeddings — not touching it")
    keep = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).map(is_english).to_numpy()
    dropped = int((~keep).sum())
    if dropped and not dry_run:
        tmp = parquet + ".tmp"
        df[keep].reset_index(drop=True).to_parquet(tmp, index=False, row_group_size=2000, compression="zstd")
        os.replace(tmp, parquet)
        tmp = npy + ".tmp.npy"
        np.save(tmp, bits[keep])
        os.replace(tmp, npy)
    return dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("df_dir")
    ap.add_argument("embed_dir")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    total = 0
    for parquet in sorted(glob.glob(os.path.join(args.df_dir, "*.parquet"))):
        npy = os.path.join(args.embed_dir, os.path.basename(parquet)[:-8] + ".npy")
        if os.path.exists(npy):
            total += filter_pair(parquet, npy, args.dry_run)
    print(f"{'would drop' if args.dry_run else 'dropped'} {total:,} non-English rows")


if __name__ == "__main__":
    main()
