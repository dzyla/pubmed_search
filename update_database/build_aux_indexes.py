"""
Builds the auxiliary search indexes on the pipeline desktop (weekly):

  1. Identifier index per source — for exact-term matching of gene symbols,
     variants, compound codes, trial ids (see identifiers.py). Postings are
     (file stem id, row) pairs, so the server maps them to its own row ids.
  2. PubMed superseded rows — PubMed's daily update files re-issue revised
     records under the same PMID; every copy except the newest is listed so
     the server can hide it (no re-embedding, no re-upload of the data).

Output (synced to the server by the nightly rsync):
    <base>/aux_index/<Source>/<version>/
        hash.npy      uint64, sorted unique token hashes
        offsets.npy   int64,  postings of token i are [offsets[i], offsets[i+1])
        stem.npy      uint16, posting -> index into stems.json
        row.npy       uint32, posting -> row in that file
        stems.json    file stems (same names as the .npy embedding files)
        superseded_stem.npy / superseded_row.npy   (PubMed only)
        zzz_complete.json   written last; the server only loads complete versions

    python build_aux_indexes.py                       # all sources
    python build_aux_indexes.py --sources arXiv ClinicalTrials
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identifiers import hash_tokens, identifier_tokens  # noqa: E402

DEFAULT_BASE = "/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake"
DF_CAP = 20_000          # tokens in more documents than this are common words, not identifiers
KEEP_VERSIONS = 2

# Source name (as used by the server) -> (embeddings dir, data dir, combined parquet or None, text columns)
SOURCES = {
    "PubMed": ("pubmed26_update_embed", "pubmed26_parquet_files", None, ["title", "abstract"]),
    "BioRxiv": ("biorxiv_embed_binary", "biorxiv_embed_binary", "biorxiv_metadata.parquet", ["title", "abstract"]),
    "MedRxiv": ("medarxiv_embed_binary", "medarxiv_embed_binary", "medarxiv_metadata.parquet", ["title", "abstract"]),
    "arXiv": ("arxiv_embed", "arxiv_df", None, ["title", "abstract"]),
    "ClinicalTrials": ("clinicaltrials_embed", "clinicaltrials_df", None,
                       ["title", "abstract", "nct_id", "conditions", "interventions"]),
    "Preprints": ("preprints_embed", "preprints_df", None, ["title", "abstract"]),
    "Grants": ("grants_embed", "grants_df", None, ["title", "abstract", "grant_id"]),
}


def _index_file(args):
    """Tokenizes one parquet file -> (hashes, rows, pmids or None, n_rows)."""
    stem, path, cols, want_pmid = args
    schema = pq.read_schema(path)
    use = [c for c in cols if c in schema.names] + (["pmid"] if want_pmid else [])
    table = pq.read_table(path, columns=use).to_pydict()
    n = len(table[use[0]]) if use else 0
    hashes, rows = [], []
    for i in range(n):
        text = " ".join(str(table[c][i] or "") for c in cols if c in table)
        toks = identifier_tokens(text)
        if toks:
            h = hash_tokens(toks)
            hashes.append(h)
            rows.append(np.full(len(h), i, dtype=np.uint32))
    pmids = None
    if want_pmid:
        pmids = np.array([int(p) if str(p).isdigit() else -1 for p in table["pmid"]], dtype=np.int64)
    return (stem,
            np.concatenate(hashes) if hashes else np.empty(0, np.uint64),
            np.concatenate(rows) if rows else np.empty(0, np.uint32),
            pmids, n)


def build_source(name: str, base: str, workers: int):
    emb_dir, data_dir, combined, cols = SOURCES[name]
    npys = sorted(glob.glob(os.path.join(base, emb_dir, "*.npy")))
    if not npys:
        print(f"[{name}] no embeddings in {emb_dir}; skipped")
        return
    jobs = []
    for npy in npys:
        stem = os.path.basename(npy)[:-4]
        path = os.path.join(base, data_dir, combined) if combined else os.path.join(base, data_dir, stem + ".parquet")
        if not os.path.exists(path):
            print(f"[{name}] missing parquet for {stem}; skipped")
            continue
        n_emb = np.load(npy, mmap_mode="r").shape[0]
        if pq.ParquetFile(path).metadata.num_rows != n_emb:
            print(f"[{name}] {stem}: parquet rows != embeddings ({n_emb}); skipped")
            continue
        jobs.append((stem, path, cols, name == "PubMed"))

    t0 = time.time()
    stems, H, S, R, pmid_parts = [], [], [], [], []
    with Pool(workers) as pool:
        for k, (stem, h, r, pmids, n) in enumerate(pool.imap(_index_file, jobs, chunksize=4)):
            sid = len(stems)
            stems.append(stem)
            H.append(h)
            R.append(r)
            S.append(np.full(len(h), sid, dtype=np.uint16))
            if pmids is not None:
                pmid_parts.append((sid, pmids))
            if (k + 1) % 100 == 0:
                print(f"[{name}] {k + 1}/{len(jobs)} files ({time.time() - t0:.0f}s)")
    H, S, R = np.concatenate(H), np.concatenate(S), np.concatenate(R)
    print(f"[{name}] {len(H):,} postings from {len(stems)} files; sorting …")

    order = np.argsort(H, kind="stable")
    H, S, R = H[order], S[order], R[order]
    uniq, start, counts = np.unique(H, return_index=True, return_counts=True)
    keep = counts <= DF_CAP
    keep_postings = np.repeat(keep, counts)
    H_u, counts = uniq[keep], counts[keep]
    S, R = S[keep_postings], R[keep_postings]
    offsets = np.zeros(len(H_u) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    out_root = os.path.join(base, "aux_index", name)
    version = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(out_root, version)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "hash.npy"), H_u)
    np.save(os.path.join(out, "offsets.npy"), offsets)
    np.save(os.path.join(out, "stem.npy"), S)
    np.save(os.path.join(out, "row.npy"), R)
    np.save(os.path.join(out, "common.npy"), uniq[~keep])     # dropped as common: ignored in every source
    with open(os.path.join(out, "stems.json"), "w") as f:
        json.dump(stems, f)

    meta = {"source": name, "version": version, "files": len(stems), "tokens": int(len(H_u)),
            "postings": int(len(S)), "dropped_common_tokens": int((~keep).sum()), "df_cap": DF_CAP}
    if pmid_parts:
        sup_stem, sup_row = superseded(pmid_parts)
        np.save(os.path.join(out, "superseded_stem.npy"), sup_stem)
        np.save(os.path.join(out, "superseded_row.npy"), sup_row)
        meta["superseded_rows"] = int(len(sup_row))
    meta["build_seconds"] = round(time.time() - t0)
    with open(os.path.join(out, "zzz_complete.json"), "w") as f:      # written last
        json.dump(meta, f, indent=1)
    for old in sorted(glob.glob(os.path.join(out_root, "*/")))[:-KEEP_VERSIONS]:
        shutil.rmtree(old, ignore_errors=True)
    print(f"[{name}] done: {json.dumps(meta)}")


def superseded(pmid_parts):
    """Every (file, row) whose PMID appears again in a later file/row."""
    sid = np.concatenate([np.full(len(p), s, dtype=np.uint16) for s, p in pmid_parts])
    row = np.concatenate([np.arange(len(p), dtype=np.uint32) for _, p in pmid_parts])
    pmid = np.concatenate([p for _, p in pmid_parts])
    valid = pmid >= 0
    sid, row, pmid = sid[valid], row[valid], pmid[valid]
    # Files are processed in sorted order (= chronological for pubmed26nNNNN), so the
    # last occurrence of each PMID in (file, row) order is the newest version.
    order = np.lexsort((row, sid, pmid))
    pmid_sorted = pmid[order]
    is_last = np.ones(len(order), dtype=bool)
    is_last[:-1] = pmid_sorted[:-1] != pmid_sorted[1:]
    old = order[~is_last]
    return sid[old], row[old]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--sources", nargs="*", default=list(SOURCES))
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 4))
    args = ap.parse_args()
    for name in args.sources:
        build_source(name, args.base, args.workers)


if __name__ == "__main__":
    main()
