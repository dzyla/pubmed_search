"""Step 1: draw a stratified sample of stored 384-bit codes + metadata (read-only on H:).

Output (WORK dir):
  sample_codes.npy   (N, 48) uint8  stored packed codes
  sample_meta.parquet  source, file, row, title, text (for re-embedding), year, journal,
                       mesh_terms / categories / category / conditions
"""
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake"
WORK = "/mnt/h/pubmed_semantic_search/umap_work"
os.makedirs(WORK, exist_ok=True)
RNG = np.random.default_rng(0)

TARGET = {"pubmed": 1_000_000, "arxiv": 250_000, "biorxiv": 120_000,
          "medrxiv": 50_000, "clinicaltrials": 100_000}


def pairs(source):
    if source == "pubmed":
        npys = sorted(glob.glob(f"{ROOT}/pubmed26_update_embed/*.npy"))
        return [(n, f"{ROOT}/pubmed26_parquet_files/{os.path.basename(n)[:-4]}.parquet") for n in npys]
    if source == "arxiv":
        npys = sorted(glob.glob(f"{ROOT}/arxiv_embed/*.npy"))
        return [(n, f"{ROOT}/arxiv_df/{os.path.basename(n)[:-4]}.parquet") for n in npys]
    if source == "clinicaltrials":
        npys = sorted(glob.glob(f"{ROOT}/clinicaltrials_embed/*.npy"))
        return [(n, f"{ROOT}/clinicaltrials_df/{os.path.basename(n)[:-4]}.parquet") for n in npys]
    if source == "biorxiv":
        return [(f"{ROOT}/biorxiv_embed_binary/biorxiv_binary_bge.npy",
                 f"{ROOT}/biorxiv_embed_binary/biorxiv_metadata.parquet")]
    if source == "medrxiv":
        return [(f"{ROOT}/medarxiv_embed_binary/medarxiv_binary_bge.npy",
                 f"{ROOT}/medarxiv_embed_binary/medarxiv_metadata.parquet")]


EXTRA = {"pubmed": ["journal", "mesh_terms"], "arxiv": ["categories"],
         "biorxiv": ["category"], "medrxiv": ["category"],
         "clinicaltrials": ["conditions"]}


def read_rows(source, npy, pqf, idx):
    codes = np.load(npy, mmap_mode="r")
    cols = ["title", "abstract", "date"] + EXTRA[source]
    t = pq.read_table(pqf, columns=cols).take(idx)
    df = t.to_pandas()
    df.insert(0, "row", idx)
    df.insert(0, "file", os.path.basename(npy))
    df.insert(0, "source", source)
    return np.asarray(codes[idx]), df


def main():
    t0 = time.time()
    all_codes, all_meta = [], []
    for source, n_target in TARGET.items():
        ps = pairs(source)
        rows = []
        good = []
        for n, p in ps:
            if not os.path.exists(p):
                continue
            nr = np.load(n, mmap_mode="r").shape[0]
            pr = pq.ParquetFile(p).metadata.num_rows
            if nr != pr:
                print(f"skip {n}: npy {nr} != parquet {pr}")
                continue
            good.append((n, p)); rows.append(nr)
        rows = np.array(rows)
        total = rows.sum()
        # global uniform sample without replacement, then split by file
        gidx = np.sort(RNG.choice(total, size=min(n_target, total), replace=False))
        bounds = np.concatenate([[0], np.cumsum(rows)])
        fid = np.searchsorted(bounds, gidx, side="right") - 1
        jobs = []
        for f in np.unique(fid):
            local = gidx[fid == f] - bounds[f]
            jobs.append((source, good[f][0], good[f][1], local))
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(lambda j: read_rows(*j), jobs))
        codes = np.concatenate([r[0] for r in res])
        meta = pd.concat([r[1] for r in res], ignore_index=True)
        print(f"{source}: total={total:,} sampled={len(meta):,} files={len(jobs)} t={time.time()-t0:.0f}s", flush=True)
        all_codes.append(codes); all_meta.append(meta)

    codes = np.concatenate(all_codes)
    meta = pd.concat(all_meta, ignore_index=True)
    for c in ["journal", "mesh_terms", "categories", "category", "conditions", "title", "abstract"]:
        if c in meta:
            meta[c] = meta[c].fillna("").astype(str)
    meta["year"] = pd.to_numeric(meta["date"].astype(str).str[:4], errors="coerce")
    meta["text"] = meta["title"] + ". " + meta["abstract"]
    meta = meta.drop(columns=["abstract", "date"])
    np.save(f"{WORK}/sample_codes.npy", codes)
    meta.to_parquet(f"{WORK}/sample_meta.parquet", index=False)
    print("saved", codes.shape, meta.shape, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
