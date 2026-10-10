"""
Biology / medicine journal abstracts from OpenAlex that the other sources lack
(journals outside MEDLINE, meeting abstracts printed in journal supplements).

Selection (OpenAlex filter): Life + Health Sciences topics, English, with an
abstract, no PubMed id, not arXiv, not the noisy expansion corpus, journal
articles and reviews with a DOI in CWTS "core" journals, not retracted
(~5.7M works). Then cleaned here:

- abstract rebuilt from the inverted index, markup and a leading "Abstract"
  label removed; at least MIN_WORDS words; no landing-page junk; English
- not already indexed: DOI or normalised title found in PubMed, bioRxiv,
  medRxiv or the other preprint servers
- meeting abstracts (supplement issues, session codes as page numbers,
  numbered titles) are kept and labelled "Conference abstract"

Harvested month by month (cursor paging, 200 works per request) so a run can
stop when the daily API allowance is used up and the next run resumes; each
month is recorded in openalex_coordination/state.json once its rows are written.
Rows are embedded and written in chunks like the other sources:

    <base>/openalex_df/openalex_<n>.parquet + <base>/openalex_embed/openalex_<n>.npy

Needs OPENALEX_API_KEY (environment or ~/.config/mss/openalex.env).

    python openalex_update.py                 # continue the harvest (daily cron)
    python openalex_update.py --max-requests 50 --dry-run
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from textlang import is_english  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprints_update import embed  # noqa: E402  (same model and settings as the other sources)

API = "https://api.openalex.org/works"
DEFAULT_BASE = "/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake"
FILTER = ("primary_topic.domain.id:1|4,has_abstract:true,language:en,has_pmid:false,indexed_in:!arxiv,"
          "is_xpac:false,type:article|review,has_doi:true,primary_location.source.is_core:true,is_retracted:false")
SELECT = "id,doi,title,publication_date,type,biblio,primary_location,abstract_inverted_index,authorships,ids"
FIRST_YEAR = 1950
MIN_WORDS = 80
CHUNK_ROWS = 25_000
RESERVE_REQUESTS = 25          # leave a little of the daily allowance unused
COLUMNS = ["record_id", "doi", "title", "authors", "date", "abstract", "journal", "pub_type", "pmcid", "url"]

_JUNK = re.compile(r"(?i)you have access|get access|return to issue|advertisement|cookie|sign in to|"
                   r"purchase this|download pdf|full text available|this article is protected|"
                   r"no abstract available|all rights reserved|©\s*\d{4}")
_LEADING_LABEL = re.compile(r"^\s*(abstract|summary|background)\s*[:.\-–]?\s+", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
_SESSION_PAGE = re.compile(r"^(?!e\d)[A-Za-z]{1,5}[\-.]?\d")      # 'MP02-07', 'A123', 'P1-15' (not 'e1234')


def api_key() -> str:
    key = os.environ.get("OPENALEX_API_KEY", "")
    path = os.path.expanduser("~/.config/mss/openalex.env")
    if not key and os.path.exists(path):
        for line in open(path):
            if line.startswith("OPENALEX_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("OPENALEX_API_KEY is not set (environment or ~/.config/mss/openalex.env)")
    return key


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def rebuild_abstract(inverted) -> str:
    if not inverted:
        return ""
    words = {}
    for word, positions in inverted.items():
        for p in positions:
            words[p] = word
    text = " ".join(words[i] for i in sorted(words))
    text = _TAGS.sub(" ", text)
    text = _LEADING_LABEL.sub("", text)
    return " ".join(text.split())


def is_meeting_abstract(work: dict, title: str, abstract: str) -> bool:
    biblio = work.get("biblio") or {}
    issue, page = str(biblio.get("issue") or ""), str(biblio.get("first_page") or "")
    return bool(re.search(r"(?i)suppl|^s\d", issue) or _SESSION_PAGE.match(page)
                or re.match(r"^\d{1,4}\.\s", title) or "(this meeting)" in abstract.lower())


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(title or "").lower())


def h64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def to_row(work: dict):
    """A cleaned row, or (None, reason) if the work does not qualify."""
    title = " ".join(_TAGS.sub(" ", str(work.get("title") or "")).split())
    abstract = rebuild_abstract(work.get("abstract_inverted_index"))
    if len(title.split()) < 3:
        return None, "title"
    if len(abstract.split()) < MIN_WORDS:
        return None, "short"
    if _JUNK.search(abstract):
        return None, "junk"
    if not is_english(title + ". " + abstract):
        return None, "language"
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    authors = ", ".join(a.get("author", {}).get("display_name", "") for a in (work.get("authorships") or [])[:30]
                        if a.get("author"))
    pmcid = str((work.get("ids") or {}).get("pmcid") or "").rstrip("/").rsplit("/", 1)[-1]
    meeting = is_meeting_abstract(work, title, abstract)
    pub_type = "Conference abstract" if meeting else ("Review" if work.get("type") == "review" else "Journal Article")
    return {
        "record_id": str(work.get("id", "")).rsplit("/", 1)[-1],
        "doi": str(work.get("doi") or "").replace("https://doi.org/", ""),
        "title": title, "authors": authors, "date": work.get("publication_date") or "",
        "abstract": abstract, "journal": source.get("display_name") or "",
        "pub_type": pub_type, "pmcid": pmcid if pmcid.upper().startswith("PMC") else "",
        "url": loc.get("landing_page_url") or "",
    }, None


# ---------------------------------------------------------------------------
# What is already indexed (DOIs and titles of the other sources)
# ---------------------------------------------------------------------------

def known_keys(base: str, cache: str) -> tuple:
    """Sorted uint64 hashes of DOIs and normalised titles already in the index (cached daily)."""
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 86_400:
        z = np.load(cache)
        return z["doi"], z["title"]
    files = (glob.glob(os.path.join(base, "pubmed26_parquet_files", "*.parquet"))
             + glob.glob(os.path.join(base, "preprints_df", "*.parquet"))
             + [os.path.join(base, "biorxiv_embed_binary", "biorxiv_metadata.parquet"),
                os.path.join(base, "medarxiv_embed_binary", "medarxiv_metadata.parquet")])
    dois, titles = set(), set()
    for k, path in enumerate(files):
        if not os.path.exists(path):
            continue
        t = pq.read_table(path, columns=["doi", "title"])
        for d in pc.utf8_lower(t["doi"].cast("string")).to_pylist():
            if d:
                dois.add(h64(d.strip()))
        for x in t["title"].to_pylist():
            n = norm_title(x)
            if len(n) >= 20:                      # very short titles ('Editorial') say little
                titles.add(h64(n))
        if (k + 1) % 200 == 0:
            print(f"  known keys: {k + 1}/{len(files)} files")
    doi_arr = np.array(sorted(dois), dtype=np.uint64)
    title_arr = np.array(sorted(titles), dtype=np.uint64)
    np.savez(cache, doi=doi_arr, title=title_arr)
    return doi_arr, title_arr


def _contains(sorted_arr: np.ndarray, value: int) -> bool:
    i = int(np.searchsorted(sorted_arr, np.uint64(value)))
    return i < len(sorted_arr) and sorted_arr[i] == np.uint64(value)


# ---------------------------------------------------------------------------
# Harvest
# ---------------------------------------------------------------------------

class Budget(Exception):
    pass


def months(first_year: int):
    today = date.today()
    for y in range(first_year, today.year + 1):
        for m in range(1, 13):
            if (y, m) > (today.year, today.month):
                return
            end = date(y + (m == 12), m % 12 + 1, 1)
            yield f"{y:04d}-{m:02d}", date(y, m, 1).isoformat(), (end.fromordinal(end.toordinal() - 1)).isoformat()


def fetch_month(session, key: str, start: str, end: str, max_requests: int, used: list):
    """All works of one publication month; raises Budget when the allowance runs out."""
    cursor, works = "*", []
    while cursor:
        if used[0] >= max_requests:
            raise Budget()
        for attempt in range(6):
            r = session.get(API, params={"filter": f"{FILTER},from_publication_date:{start},to_publication_date:{end}",
                                         "per-page": 200, "cursor": cursor, "select": SELECT, "api_key": key},
                            timeout=120)
            used[0] += 1
            if r.status_code == 429:
                if int(r.headers.get("X-RateLimit-Remaining", "1") or 1) <= 0:
                    raise Budget()
                time.sleep(2 ** attempt)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        else:
            r.raise_for_status()
        remaining = r.headers.get("X-RateLimit-Remaining")
        if remaining is not None and int(remaining) < RESERVE_REQUESTS:
            max_requests = used[0]           # finish nothing more after this page
        data = r.json()
        works.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not data.get("results"):
            break
    return works


def write_chunk(rows: list, df_dir: str, emb_dir: str, n: int):
    df = pd.DataFrame(rows, columns=COLUMNS).reset_index(drop=True)
    bits = embed(df)
    stem = f"openalex_{n:04d}"
    tmp = os.path.join(df_dir, stem + ".parquet.tmp")
    df.to_parquet(tmp, index=False, row_group_size=2000, compression="zstd")
    os.replace(tmp, os.path.join(df_dir, stem + ".parquet"))             # metadata first
    tmp = os.path.join(emb_dir, stem + ".tmp.npy")
    np.save(tmp, bits)
    os.replace(tmp, os.path.join(emb_dir, stem + ".npy"))
    print(f"  wrote {stem}: {len(df):,} abstracts")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--max-requests", type=int, default=9_900, help="API requests for this run")
    ap.add_argument("--first-year", type=int, default=FIRST_YEAR)
    ap.add_argument("--dry-run", action="store_true", help="fetch and clean, write nothing")
    args = ap.parse_args()
    key = api_key()
    df_dir, emb_dir = os.path.join(args.base, "openalex_df"), os.path.join(args.base, "openalex_embed")
    coord = os.path.join(args.base, "openalex_coordination")
    for d in (df_dir, emb_dir, coord):
        os.makedirs(d, exist_ok=True)
    state_path = os.path.join(coord, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"done": [], "chunks": 0, "stats": {}}

    print("Loading DOIs and titles already indexed …")
    known_doi, known_title = known_keys(args.base, os.path.join(coord, "known_keys.npz"))
    print(f"  {len(known_doi):,} DOIs, {len(known_title):,} titles")

    session, used, buffer, pending_months = requests.Session(), [0], [], []
    stats = {k: int(v) for k, v in state.get("stats", {}).items()}
    seen = set()

    def flush(force=False):
        # Chunks are cut only between months, so the months marked done are
        # exactly the ones whose rows are on disk (a crash never duplicates rows).
        nonlocal buffer
        if buffer and (force or len(buffer) >= CHUNK_ROWS):
            if not args.dry_run:
                write_chunk(buffer, df_dir, emb_dir, state["chunks"])
            state["chunks"] += 1
            buffer = []
        if not buffer:
            state["done"].extend(pending_months)
            pending_months.clear()
            if not args.dry_run:
                state["stats"] = stats
                tmp = state_path + ".tmp"
                json.dump(state, open(tmp, "w"))
                os.replace(tmp, state_path)

    try:
        for month, start, end in months(args.first_year):
            if month in state["done"]:
                continue
            works = fetch_month(session, key, start, end, args.max_requests, used)
            for w in works:
                row, reason = to_row(w)
                if row is None:
                    stats[reason] = stats.get(reason, 0) + 1
                    continue
                if (row["record_id"] in seen or _contains(known_doi, h64(row["doi"].lower()))
                        or _contains(known_title, h64(norm_title(row["title"])))):
                    stats["already_indexed"] = stats.get("already_indexed", 0) + 1
                    continue
                seen.add(row["record_id"])
                stats["kept"] = stats.get("kept", 0) + 1
                buffer.append(row)
            pending_months.append(month)
            print(f"{month}: {len(works):,} works, {stats.get('kept', 0):,} kept so far ({used[0]:,} requests)")
            flush()
    except Budget:
        # the month cut short is not in pending_months: it is fetched again next run
        print(f"Daily API allowance reached after {used[0]:,} requests; the next run continues.")
    flush(force=True)
    print(json.dumps({"months_done": len(state["done"]), "chunks": state["chunks"], "requests": used[0], **stats}))


if __name__ == "__main__":
    main()
