"""
arXiv incremental update via arXiv's OAI-PMH feed (no Kaggle account needed).

Fetches every record whose metadata changed since --from (default: the date
saved by the last successful run, else the newest date in the parquet files
minus a safety margin), keeps papers whose arXiv id is not indexed yet, writes
them as new parquet chunks in exactly the format of
arxiv_download_embed_update.py, and embeds them with that script's GPU
pipeline. arXiv metadata is CC0.

    python arxiv_oai_update.py                       # since last run
    python arxiv_oai_update.py --from 2026-06-01     # explicit start date
    python arxiv_oai_update.py --no-embed            # fetch + parquet only

Be polite: one request at a time, 3 s apart, honouring Retry-After.
"""
import argparse
import glob
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arxiv_download_embed_update as pipe  # noqa: E402  (shared paths, chunking, embedding)

OAI_URL = "https://oaipmh.arxiv.org/oai"
USER_AGENT = "ManuscriptSearch/2.0 (https://manuscript-search.org; mailto:dawid.zyla@cuanschutz.edu)"
STATE_FILE = os.path.join(pipe.COORDINATION_DIR, "oai_state.json")
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "ax": "http://arxiv.org/OAI/arXiv/"}
REQUEST_GAP_S = 3.0


def _get(params: dict, attempts: int = 8) -> str:
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(OAI_URL, params=params, timeout=120, headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            wait = min(300, 10 * attempt)
            print(f"  request failed ({exc}); retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.text
        wait = int(r.headers.get("Retry-After", min(300, 10 * attempt)))
        print(f"  HTTP {r.status_code}; retry in {wait}s")
        time.sleep(wait)
    raise RuntimeError("arXiv OAI-PMH kept failing; try again later.")


def _text(node, path):
    found = node.find(path, NS)
    return (found.text or "").strip() if found is not None and found.text else ""


def parse_page(xml_text: str):
    """Returns (rows, resumption_token or None, complete_list_size or None)."""
    root = ET.fromstring(xml_text)
    error = root.find("oai:error", NS)
    if error is not None:
        if error.get("code") == "noRecordsMatch":
            return [], None, 0
        raise RuntimeError(f"OAI error {error.get('code')}: {error.text}")

    rows = []
    for rec in root.iterfind(".//oai:record", NS):
        header = rec.find("oai:header", NS)
        if header is not None and header.get("status") == "deleted":
            continue
        meta = rec.find("oai:metadata/ax:arXiv", NS)
        if meta is None:
            continue
        authors = []
        for a in meta.iterfind("ax:authors/ax:author", NS):
            name = " ".join(p for p in (_text(a, "ax:forenames"), _text(a, "ax:keyname"), _text(a, "ax:suffix")) if p)
            if name:
                authors.append(name)
        rows.append({
            "id": _text(meta, "ax:id"),
            "title": pipe.clean_text(_text(meta, "ax:title")),
            "abstract": pipe.clean_text(_text(meta, "ax:abstract")),
            "authors": ", ".join(authors),
            # Same meaning as the Kaggle snapshot's update_date: date of the latest version.
            "date": _text(meta, "ax:updated") or _text(meta, "ax:created"),
            "categories": _text(meta, "ax:categories"),
        })

    token_node = root.find(".//oai:resumptionToken", NS)
    token = token_node.text.strip() if token_node is not None and token_node.text else None
    size = token_node.get("completeListSize") if token_node is not None else None
    return rows, token, int(size) if size and size.isdigit() else None


def default_start() -> str:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)["last_until"]
    newest = None
    for path in glob.glob(os.path.join(pipe.DEST_DF_FOLDER, "*.parquet")):
        d = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"], errors="coerce").max()
        newest = d if newest is None or (pd.notna(d) and d > newest) else newest
    start = (newest.date() if newest is not None and pd.notna(newest) else date.today()) - timedelta(days=14)
    return start.isoformat()


def fetch_new(start: str, until: str, existing_ids: set):
    params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "from": start, "until": until}
    run_id, chunk_num, buffer, seen = int(time.time()), 0, [], set()
    pages = fetched = 0
    while True:
        rows, token, size = parse_page(_get(params))
        pages += 1
        fetched += len(rows)
        for row in rows:
            if row["id"] and row["id"] not in existing_ids and row["id"] not in seen and row["abstract"]:
                seen.add(row["id"])
                buffer.append(row)
        if len(buffer) >= pipe.CHUNK_SIZE_ROWS:
            pipe.save_chunk(buffer[:pipe.CHUNK_SIZE_ROWS], pipe.DEST_DF_FOLDER, run_id, chunk_num)
            buffer, chunk_num = buffer[pipe.CHUNK_SIZE_ROWS:], chunk_num + 1
        total = f" of ~{size:,}" if size else ""
        print(f"  page {pages}: {fetched:,}{total} records scanned, {len(seen):,} new papers")
        if not token:
            break
        params = {"verb": "ListRecords", "resumptionToken": token}
        time.sleep(REQUEST_GAP_S)
    if buffer:
        pipe.save_chunk(buffer, pipe.DEST_DF_FOLDER, run_id, chunk_num)
        chunk_num += 1
    return len(seen), chunk_num


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", help="YYYY-MM-DD (default: since last run)")
    ap.add_argument("--until", default=date.today().isoformat(), help="YYYY-MM-DD (default: today)")
    ap.add_argument("--no-embed", action="store_true", help="only fetch and write parquet chunks")
    args = ap.parse_args()

    pipe.ensure_dirs()
    start = args.start or default_start()
    datetime.strptime(start, "%Y-%m-%d")
    print(f"=== arXiv OAI-PMH update: {start} → {args.until} ===")

    existing = pipe.get_existing_ids(pipe.DEST_DF_FOLDER)
    new_papers, new_files = fetch_new(start, args.until, existing)
    del existing
    print(f"New papers: {new_papers:,} in {new_files} new parquet file(s).")

    with open(STATE_FILE, "w") as f:
        json.dump({"last_until": args.until, "run_at": datetime.now().isoformat(timespec="seconds"),
                   "new_papers": new_papers}, f)

    if args.no_embed:
        return
    pipe.register_machine()
    file_paths = sorted(glob.glob(os.path.join(pipe.DEST_DF_FOLDER, "*.parquet")))
    pipe.reconcile_queue(file_paths)          # queues only files without embeddings
    pipe.run_embedding_pipeline(file_paths, force=False)


if __name__ == "__main__":
    main()
