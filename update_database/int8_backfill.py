"""
Writes the int8 vectors (int8_codes.py) for every source's binary-code files.

For each <embed dir>/<stem>.npy whose <embed dir>_int8/<stem>.npy is missing or
older, the rows are (re-)embedded from their parquet text with the recipe the
update scripts use ("title. abstract"; PubMed with the BGE query prefix):
  - rows already in the old int8 file whose signs still match the stored binary
    code are kept (bioRxiv/medRxiv append to one file every night);
  - new or replaced rows are embedded on the GPU (fast_embed.py).
The fresh embeddings are also checked against the stored binary codes: a file whose
rows disagree means its .npy and parquet are out of step, which is reported.

Run before the nightly sync (cron 02:30), and once to fill everything (~6.5 h for
56M documents on the RTX 5080):
    python int8_backfill.py [--source PubMed ...] [--dry-run]
"""
import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import int8_codes as ic  # noqa: E402

ROOT = "/data/mss"
# Source -> (embeddings dir, data dir, combined parquet or None); same layout as build_aux_indexes.py.
# Small sources first so they are searchable with int8 early in a full run.
SOURCES = {
    "MedRxiv": ("medarxiv_embed_binary", "medarxiv_embed_binary", "medarxiv_metadata.parquet"),
    "BioRxiv": ("biorxiv_embed_binary", "biorxiv_embed_binary", "biorxiv_metadata.parquet"),
    "ClinicalTrials": ("clinicaltrials_embed", "clinicaltrials_df", None),
    "Grants": ("grants_embed", "grants_df", None),
    "Preprints": ("preprints_embed", "preprints_df", None),
    "arXiv": ("arxiv_embed", "arxiv_df", None),
    "OpenAlex": ("openalex_embed", "openalex_df", None),
    "PubMed": ("pubmed26_update_embed", "pubmed26_parquet_files", None),
}
SAME_ROW_AGREEMENT = 0.9    # share of equal sign bits: same document ≳ 0.97, a different one ≲ 0.8


def sign_agreement(packed_a: np.ndarray, packed_b: np.ndarray) -> np.ndarray:
    return 1.0 - np.unpackbits(packed_a ^ packed_b, axis=1).mean(axis=1)


def stale_files(names):
    """(source, npy path, parquet path) for every binary file whose int8 file is missing or older."""
    out = []
    for name in names:
        edir, ddir, combined = SOURCES[name]
        for npy in sorted(Path(ROOT, edir).glob("*.npy")):
            ipath = ic.int8_path(str(npy))
            if os.path.exists(ipath) and os.path.getmtime(ipath) >= npy.stat().st_mtime:
                continue
            parquet = Path(ROOT, ddir, combined) if combined else Path(ROOT, ddir, f"{npy.stem}.parquet")
            out.append((name, str(npy), str(parquet)))
    return out


def plan(name, npy, parquet):
    """Loads one file's inputs: stored codes, kept int8 rows, rows to embed and their texts."""
    codes = np.load(npy)
    n = len(codes)
    if not os.path.exists(parquet):
        return dict(error=f"no parquet {parquet}")
    rows = pq.ParquetFile(parquet).metadata.num_rows
    if rows != n:
        return dict(error=f"parquet has {rows:,} rows, .npy {n:,}")
    out = np.zeros((n, ic.DIM), dtype=np.int8)
    redo = np.ones(n, dtype=bool)
    ipath = ic.int8_path(npy)
    if os.path.exists(ipath):
        old = np.load(ipath)
        m = min(len(old), n)
        if m:
            lo, step = ic.load_calibration()
            old_signs = ic.binary((old[:m].astype(np.float32) + 128) * step + lo)
            keep = sign_agreement(old_signs, codes[:m]) >= SAME_ROW_AGREEMENT
            out[:m][keep] = old[:m][keep]
            redo[:m] = ~keep
    texts = []
    if redo.any():
        t = pq.read_table(parquet, columns=["title", "abstract"]).to_pandas()
        t = t[redo]
        prefix = ic_prefix(name)
        texts = (prefix + t["title"].fillna("").astype(str) + ". " + t["abstract"].fillna("").astype(str)).tolist()
    return dict(codes=codes, out=out, redo=redo, texts=texts)


def ic_prefix(name):
    from fast_embed import QUERY_PREFIX
    return QUERY_PREFIX if name == "PubMed" else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", choices=list(SOURCES))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    todo = stale_files(a.source or list(SOURCES))
    print(f"{time.strftime('%F %T')} int8 backfill: {len(todo)} file(s) to update", flush=True)
    if not todo or a.dry_run:
        for t in todo[:20]:
            print("  ", t[0], os.path.basename(t[1]))
        return

    from fast_embed import FastEmbedder
    fe = FastEmbedder()
    done_docs, t0, problems = 0, time.time(), []
    with ThreadPoolExecutor(2) as pool:             # read the next files while the GPU works
        futures = [pool.submit(plan, *t) for t in todo[:2]]
        for i, (name, npy, parquet) in enumerate(todo):
            p = futures[i].result()
            if i + 2 < len(todo):
                futures.append(pool.submit(plan, *todo[i + 2]))
            futures[i] = None
            stem = os.path.basename(npy)
            if "error" in p:
                print(f"  SKIP {name} {stem}: {p['error']}", flush=True)
                problems.append(f"{name} {stem}: {p['error']}")
                continue
            redo = p["redo"]
            if redo.any():
                emb = fe.encode(p["texts"])
                agree = sign_agreement(ic.binary(emb), p["codes"][redo])
                bad = int((agree < SAME_ROW_AGREEMENT).sum())
                if bad > max(5, 0.01 * len(agree)):
                    problems.append(f"{name} {stem}: {bad:,} of {len(agree):,} rows do not match their binary code")
                p["out"][redo] = ic.quantize(emb)
                done_docs += int(redo.sum())
            ipath = ic.int8_path(npy)
            os.makedirs(os.path.dirname(ipath), exist_ok=True)
            tmp = ipath + ".tmp"
            with open(tmp, "wb") as f:
                np.save(f, p["out"])
            os.replace(tmp, ipath)
            if redo.any() and (i % 25 == 0 or redo.sum() > 100_000 or i == len(todo) - 1):
                rate = done_docs / max(time.time() - t0, 1e-6)
                print(f"  [{i + 1}/{len(todo)}] {name} {stem}: {int(redo.sum()):,} embedded, "
                      f"median sign agreement {np.median(agree):.4f}; total {done_docs:,} docs, "
                      f"{rate:,.0f} docs/s", flush=True)
    print(f"{time.strftime('%F %T')} done: {done_docs:,} documents embedded in {(time.time() - t0) / 60:.1f} min",
          flush=True)
    for msg in problems:
        print("  PROBLEM:", msg, flush=True)


if __name__ == "__main__":
    main()
