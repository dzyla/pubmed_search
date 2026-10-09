"""
PMID -> PMCID lookup for "Free full text" links (PubMed Central).

Downloads NCBI's PMC-ids.csv.gz (all PMC articles, updated daily), keeps
articles that are readable now (Release Date "live" or already past), and
writes a compact sorted lookup into the aux-index tree, which the nightly
aux-index sync ships to the server:

    <base>/aux_index/PMC/<version>/pmid.npy     uint32, sorted
    <base>/aux_index/PMC/<version>/pmcid.npy    uint32, aligned with pmid.npy
    <base>/aux_index/PMC/<version>/zzz_complete.json   (written last)

    python pmc_links_update.py            # weekly (cron)
"""
import argparse
import glob
import json
import os
import shutil
import tempfile
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz"
DEFAULT_BASE = "/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake"
KEEP_VERSIONS = 2


def build(csv_path: str, out_root: str) -> dict:
    df = pd.read_csv(csv_path, usecols=["PMCID", "PMID", "Release Date"], dtype=str)
    pmid = pd.to_numeric(df["PMID"], errors="coerce")
    pmcid = pd.to_numeric(df["PMCID"].str.removeprefix("PMC"), errors="coerce")
    release = df["Release Date"].fillna("")
    # "live", or an embargo end date (YYYY-MM-DD) that has passed
    readable = (release == "live") | (pd.to_datetime(release, errors="coerce") <= pd.Timestamp(date.today()))
    ok = pmid.notna() & pmcid.notna() & readable
    pm = pmid[ok].astype(np.uint32).to_numpy()
    pc = pmcid[ok].astype(np.uint32).to_numpy()
    order = np.argsort(pm, kind="stable")
    pm, pc = pm[order], pc[order]
    keep = np.ones(len(pm), dtype=bool)
    keep[1:] = pm[1:] != pm[:-1]           # one PMC copy per PMID
    pm, pc = pm[keep], pc[keep]

    version = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(out_root, version)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "pmid.npy"), pm)
    np.save(os.path.join(out, "pmcid.npy"), pc)
    meta = {"version": version, "articles": int(len(pm)), "embargoed_or_unlinked": int((~ok).sum())}
    with open(os.path.join(out, "zzz_complete.json"), "w") as f:
        json.dump(meta, f)
    for old in sorted(glob.glob(os.path.join(out_root, "*/")))[:-KEEP_VERSIONS]:
        shutil.rmtree(old, ignore_errors=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--csv", help="use an already downloaded PMC-ids.csv(.gz)")
    args = ap.parse_args()
    out_root = os.path.join(args.base, "aux_index", "PMC")
    if args.csv:
        print(json.dumps(build(args.csv, out_root)))
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "PMC-ids.csv.gz")
        with requests.get(URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
        print(json.dumps(build(path, out_root)))


if __name__ == "__main__":
    main()
