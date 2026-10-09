"""
Preprints from servers not already indexed → Manuscript Search (source "Preprints").

Two feeds share one layout: parquet chunks in <base>/preprints_df/ and row-aligned
binary embeddings in <base>/preprints_embed/ (BAAI/bge-small-en-v1.5,
"title. abstract", no query prefix, np.packbits(emb > 0) -> 48 bytes).

  europepmc  Europe PMC REST search, SRC:PPR with an abstract, minus bioRxiv,
             medRxiv, arXiv (indexed already) and SSRN (Elsevier personal-use terms).
             Research Square, Preprints.org, PsyArXiv, Authorea, F1000-family, ...
  crossref   Crossref REST works, type:posted-content with an abstract, by DOI owner
             prefix: ChemRxiv, TechRxiv, EarthArXiv, SocArXiv, OSF Preprints and
             ESS Open Archive.

    python preprints_update.py --full            # first run (~0.8M records, ~3 h)
    python preprints_update.py                   # daily/weekly: changes since last run
    python preprints_update.py --base /tmp/x --limit 300 --full   # small test

Licence rule (a record is kept only if one holds):
  1. the record carries a Creative Commons licence (any CC variant, incl. CC0 and
     NC/ND) — Europe PMC's LICENSE field, Crossref's license URLs; or
  2. it has no licence field but comes from a server whose terms put every
     preprint under CC BY (CC_BY_SERVERS; stored as "cc by (server terms)").
  Everything else is dropped: e.g. Authorea (default licence "no reuse"),
  unlicensed OSF-hosted preprints, ESS Open Archive records without a CC URL.
Europe PMC's JSON does not return the licence, so it is harvested as one query
per licence value (LICENSE:"cc by", ...) plus one for the CC-BY-terms servers.

One row per preprint (not per version): the latest version's DOI/title/abstract,
`version` its number, `date` the posting date of the FIRST version (date filters
use it). Europe PMC groups versions with versionList (record_id = PPR id of v1);
Crossref versions share the DOI minus its version suffix (record_id = that base).
A work in both feeds is kept once, from Europe PMC. `published_doi` is the
journal version (Europe PMC "Preprint of" links are PMIDs, resolved to DOIs in
batches; Crossref "is-preprint-of").

Incremental runs fetch records updated since the last run (Europe PMC
UPDATE_DATE, Crossref from-index-date): new works go into a new chunk; known
ones are updated in place, and re-embedded only if their title/abstract changed.
Files are written to a temp name and renamed, parquet before .npy.
"""
import argparse
import glob
import html
import json
import os
import re
import time
from datetime import date, datetime

import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from textlang import is_english  # noqa: E402

EPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_POST_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/searchPOST"
CROSSREF_API = "https://api.crossref.org/works"
MAILTO = "dawid.zyla@cuanschutz.edu"
USER_AGENT = f"ManuscriptSearch/2.0 (https://manuscript-search.org; mailto:{MAILTO})"
DEFAULT_BASE = "/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake"
MODEL_ID = "BAAI/bge-small-en-v1.5"
CHUNK_ROWS = 25_000
PAGE_SIZE = 1000
REQUEST_GAP_S = 1.0
RESOLVE_BATCH = 1000       # PMIDs per Europe PMC lookup (searchPOST, ~5 s each)

# Already indexed elsewhere (bioRxiv, medRxiv, arXiv) or not redistributable (SSRN).
EXCLUDED_SERVERS = ("bioRxiv", "medRxiv", "arXiv", "SSRN")
# Values of Europe PMC's LICENSE search field for preprints (together they cover
# every licensed record: LICENSE:* minus these returns 0 hits, checked 2026-10-09).
EPMC_LICENSES = ("cc by", "cc0", "cc by-nc", "cc by-nc-nd", "cc by-sa", "cc by-nd", "cc by-nc-sa")
# Servers whose terms license every preprint CC BY 4.0, so a missing per-record
# licence field still means CC BY. (Authorea defaults to "no reuse" and
# OSF-hosted servers let authors choose "no licence": not listed.)
CC_BY_SERVERS = (
    "Research Square", "Preprints.org", "Qeios", "SciELO Preprints", "F1000Res",
    "Wellcome Open Res", "Gates Open Res", "Open Res Europe", "HRB Open Res",
    "NIHR Open Res", "Open Res Africa", "MedEdPublish", "Access Microbiology",
    "ScienceOpen Preprints", "ARPHA Preprints", "Beilstein Archives",
)
SERVER_TERMS_LICENSE = "cc by (server terms)"
# Display names for abbreviated server names (Europe PMC publisher / Crossref group-title).
SERVER_NAMES = {
    "F1000Res": "F1000Research", "Wellcome Open Res": "Wellcome Open Research",
    "Gates Open Res": "Gates Open Research", "Open Res Europe": "Open Research Europe",
    "HRB Open Res": "HRB Open Research", "NIHR Open Res": "NIHR Open Research",
    "Open Res Africa": "Open Research Africa", "Authorea Preprints": "Authorea",
    "Law Archive": "LawArXiv",
}
# Crossref owner prefixes to harvest. NB: Crossref's `prefix:` filter matches the
# DOI *owner* prefix, which need not be the DOI's own prefix (the Center for Open
# Science deposits PsyArXiv/LawArXiv/... DOIs under several owner prefixes, and
# CDL's EarthArXiv prefix owns old OSF-era DOIs), so the server name ("journal")
# is derived from the DOI itself (server_for_doi).
CROSSREF_PREFIXES = {
    "10.26434": "ChemRxiv",
    "10.36227": "TechRxiv",
    "10.31223": "EarthArXiv",
    "10.31235": "SocArXiv",
    "10.31219": "OSF Preprints",
    "10.1002": "ESS Open Archive",     # Wiley: ESSOAr before its move to Authorea (10.1002/essoar.*)
    "10.22541": "ESS Open Archive",    # Authorea: ESSOAr since 2023 (10.22541/essoar.*)
}
# Owner prefixes that also hold other content: keep only DOIs starting with these.
_ESSOAR = ("10.1002/essoar.", "10.22541/essoar.")
CROSSREF_DOI_FILTER = {"10.1002": _ESSOAR, "10.22541": _ESSOAR}
# DOI prefix -> server, for naming records (checked against Crossref group-title,
# 2026-10-09). Center for Open Science servers share 10.312xx; for those, an
# OSF group-title (e.g. "LawArXiv") is used when present. EarthArXiv's
# group-title is a subject area, so it is never used as a server name.
DOI_PREFIX_SERVERS = {
    "10.26434": "ChemRxiv", "10.36227": "TechRxiv", "10.31223": "EarthArXiv",
    "10.31235": "SocArXiv", "10.31219": "OSF Preprints", "10.31234": "PsyArXiv",
    "10.31222": "MetaArXiv", "10.31228": "LawArXiv", "10.31227": "INA-Rxiv",
    "10.31221": "Arabixiv", "10.31230": "MarXiv", "10.31231": "MindRxiv",
    "10.31237": "Thesis Commons", "10.32942": "EcoEvoRxiv", "10.35542": "EdArXiv",
    "10.22541": "Authorea", "10.1002": "ESS Open Archive",
}


def _server_name(name: str) -> str:
    name = str(name or "").strip()
    return SERVER_NAMES.get(name, name) or "Preprint"


def server_for_doi(doi: str, group_title: str = "") -> str:
    doi = str(doi or "").lower()
    if doi.startswith(_ESSOAR):
        return "ESS Open Archive"
    prefix = doi.split("/", 1)[0]
    group_title = str(group_title or "").strip()
    if prefix.startswith("10.312") and prefix != "10.31223" and group_title \
            and group_title != "Open Science Framework":
        return _server_name(group_title)
    if prefix in DOI_PREFIX_SERVERS:
        return DOI_PREFIX_SERVERS[prefix]
    return "OSF Preprints" if prefix.startswith("10.312") else "Preprint"


COLUMNS = ["record_id", "doi", "title", "abstract", "authors", "journal", "date", "license",
           "published_doi", "published_pmid", "version", "source_feed"]
TEXT_COLUMNS = ("title", "abstract")
MIN_ABSTRACT_CHARS = 50


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _get(url: str, params: dict, attempts: int = 8, post: bool = False) -> dict:
    """GET (or form POST) returning JSON, with retries; honours Retry-After on 429/503."""
    for attempt in range(1, attempts + 1):
        try:
            if post:
                r = requests.post(url, data=params, timeout=180, headers={"User-Agent": USER_AGENT})
            else:
                r = requests.get(url, params=params, timeout=180, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return r.json()
            if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                raise RuntimeError(f"HTTP {r.status_code} from {url}: {r.text[:300]}")
            wait = r.headers.get("Retry-After", "")
            wait = int(wait) if wait.isdigit() else min(300, 15 * attempt)
            print(f"  HTTP {r.status_code}; retry in {wait}s")
        except (requests.RequestException, ValueError) as exc:
            wait = min(300, 15 * attempt)
            print(f"  request failed ({exc}); retry in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"{url} kept failing; try again later.")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_HEADING = re.compile(r"<(?:jats:)?title>\s*(?:abstract|summary|graphical abstract)\s*[:.]?\s*</(?:jats:)?title>",
                      re.IGNORECASE)
_BLOCK_TAG = re.compile(r"</?(?:jats:)?(?:p|div|br|sec|title|li|list-item|list|pre|h\d)\b[^<>]{0,200}/?>", re.IGNORECASE)
_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9:_-]*(?:\s[^<>]{0,300})?/?>")


def clean_text(text) -> str:
    """Strips JATS/HTML tags (also HTML-escaped ones) and entities, collapses whitespace.
    A bare '<' as in 'p < 0.05' is kept."""
    text = str(text or "")
    for _ in range(2):              # some abstracts carry escaped markup (&lt;div&gt;)
        text = _HEADING.sub(" ", text)
        text = _BLOCK_TAG.sub(" ", text)
        text = _TAG.sub("", text)
        text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^(?:abstract|summary)\s*[:.]\s*", "", text, flags=re.IGNORECASE)


def _full_date(value: str) -> str:
    value = (value or "").strip()[:10]
    if re.fullmatch(r"\d{4}", value):
        return value + "-01-01"
    if re.fullmatch(r"\d{4}-\d{2}", value):
        return value + "-01"
    return value


def _date_parts(obj) -> str:
    try:
        parts = (obj or {}).get("date-parts", [[]])[0]
        parts = [int(p) for p in parts if p is not None]
    except (TypeError, ValueError, IndexError):
        return ""
    if not parts:
        return ""
    y, m, d = (parts + [1, 1])[:3]
    return f"{y:04d}-{m:02d}-{d:02d}"


_VERSION_SUFFIX = re.compile(r"[._/-]v(\d+)$", re.IGNORECASE)
_ESSOAR_V = re.compile(r"^(10\.1002/essoar\.\d+)\.(\d+)$")


def doi_base(doi: str) -> tuple:
    """'10.26434/chemrxiv.123.v2' -> ('10.26434/chemrxiv.123', 2). Lower-cased."""
    doi = str(doi or "").strip().lower()
    m = _ESSOAR_V.match(doi)
    if m:
        return m.group(1), int(m.group(2))
    m = _VERSION_SUFFIX.search(doi)
    if m:
        return doi[:m.start()], int(m.group(1))
    return doi, 1


def _license_label(url: str) -> str:
    """CC licence URL -> 'cc by-nc-nd' / 'cc0'; '' for anything that is not CC."""
    url = str(url or "").lower()
    if "creativecommons.org" not in url:
        return ""
    if "publicdomain" in url:
        return "cc0"
    m = re.search(r"/licenses/([a-z-]+)", url)
    return "cc " + m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Europe PMC
# ---------------------------------------------------------------------------

def epmc_base_query() -> str:
    excluded = " ".join(f'NOT PUBLISHER:"{s}"' for s in EXCLUDED_SERVERS)
    return f"SRC:PPR AND HAS_ABSTRACT:y {excluded}"


def epmc_queries(since: str = None) -> list:
    """[(query, licence label)]: one per licence value, then the CC-BY-terms servers."""
    base = epmc_base_query()
    tail = f" AND UPDATE_DATE:[{since} TO *]" if since else ""
    queries = [(f'{base} AND LICENSE:"{lic}"{tail}', lic) for lic in EPMC_LICENSES]
    servers = " OR ".join(f'PUBLISHER:"{s}"' for s in CC_BY_SERVERS)
    queries.append((f"{base} NOT LICENSE:* AND ({servers}){tail}", SERVER_TERMS_LICENSE))
    return queries


def epmc_record_to_row(rec: dict, license_label: str) -> dict:
    versions = (rec.get("versionList") or {}).get("version") or []
    first = min(versions, key=lambda v: v.get("versionNumber") or 0) if versions else {}
    published_pmid = ""
    for cc in (rec.get("commentCorrectionList") or {}).get("commentCorrection", []):
        if cc.get("type") == "Preprint of" and cc.get("source") == "MED" and str(cc.get("id", "")).isdigit():
            published_pmid = str(cc["id"])
            break
    posted = [v.get("firstPublishDate") for v in versions if v.get("firstPublishDate")]
    return {
        # Every version is its own PPR record; v1's id identifies the work.
        "record_id": first.get("id") or rec.get("id", ""),
        "doi": str(rec.get("doi") or "").strip(),
        "title": clean_text(rec.get("title")).rstrip("."),
        "abstract": clean_text(rec.get("abstractText")),
        "authors": str(rec.get("authorString") or "").strip().rstrip("."),
        "journal": _server_name((rec.get("bookOrReportDetails") or {}).get("publisher", "")),
        # Posting date of the first version; used by date filters.
        "date": _full_date(min(posted) if posted else rec.get("firstPublicationDate", "")),
        "license": license_label,
        "published_doi": "",
        "published_pmid": published_pmid,
        "version": str(rec.get("versionNumber") or doi_base(rec.get("doi"))[1]),
        "source_feed": "europepmc",
    }


def fetch_europepmc(since: str = None, limit: int = None):
    """Yields rows for Europe PMC preprints (all, or updated since `since`).
    `limit` caps the rows taken from each licence query (testing)."""
    for query, lic in epmc_queries(since):
        params = {"query": query, "format": "json", "resultType": "core",
                  "pageSize": PAGE_SIZE, "cursorMark": "*"}
        seen, page = 0, 0
        while True:
            data = _get(EPMC_API, params)
            page += 1
            results = (data.get("resultList") or {}).get("result", [])
            if page == 1:
                print(f"  Europe PMC [{lic}]: {data.get('hitCount', 0):,} records")
            for rec in results:
                row = epmc_record_to_row(rec, lic)
                if row["record_id"] and row["title"] and len(row["abstract"]) >= MIN_ABSTRACT_CHARS:
                    yield row
                    seen += 1
                    if limit and seen >= limit:
                        break
            if limit and seen >= limit:
                break
            if page % 50 == 0:
                print(f"    page {page}: {seen:,} kept")
            cursor = data.get("nextCursorMark")
            if not results or not cursor or cursor == params["cursorMark"]:
                break
            params["cursorMark"] = cursor
            time.sleep(REQUEST_GAP_S)


def resolve_pmid_dois(pmids) -> dict:
    """PMID -> DOI of the journal version, via batched Europe PMC lookups."""
    pmids = sorted({p for p in pmids if p})
    out = {}
    for i in range(0, len(pmids), RESOLVE_BATCH):
        batch = pmids[i:i + RESOLVE_BATCH]
        query = "SRC:MED AND (" + " OR ".join(f"EXT_ID:{p}" for p in batch) + ")"
        data = _get(EPMC_POST_API, {"query": query, "format": "json", "resultType": "lite",
                                    "pageSize": RESOLVE_BATCH}, post=True)
        for rec in (data.get("resultList") or {}).get("result", []):
            if rec.get("doi"):
                out[str(rec.get("id"))] = rec["doi"]
        time.sleep(REQUEST_GAP_S)
    return out


def fill_published_dois(rows: list) -> None:
    need = [r["published_pmid"] for r in rows if r.get("published_pmid") and not r.get("published_doi")]
    if not need:
        return
    found = resolve_pmid_dois(need)
    for r in rows:
        if r.get("published_pmid") and not r.get("published_doi"):
            r["published_doi"] = found.get(r["published_pmid"], "")


# ---------------------------------------------------------------------------
# Crossref
# ---------------------------------------------------------------------------

CROSSREF_SELECT = "DOI,title,abstract,author,posted,created,license,relation,group-title,prefix"


def crossref_item_to_row(item: dict):
    """Crossref work -> row, or None when it has no CC licence or no usable text."""
    lic = ""
    for entry in item.get("license") or []:
        lic = _license_label(entry.get("URL"))
        if lic:
            break
    doi = str(item.get("DOI") or "").strip()
    title = clean_text((item.get("title") or [""])[0]).rstrip(".")
    abstract = clean_text(item.get("abstract"))
    if not lic or not doi or not title or len(abstract) < MIN_ABSTRACT_CHARS:
        return None
    base, version = doi_base(doi)
    published = ""
    for rel in (item.get("relation") or {}).get("is-preprint-of", []):
        if rel.get("id-type") == "doi" and rel.get("id"):
            published = str(rel["id"]).strip()
            break
    authors = []
    for a in item.get("author") or []:
        if a.get("family"):
            initials = "".join(p[0] for p in re.split(r"[\s.-]+", a.get("given", "")) if p)
            authors.append(f"{a['family']} {initials}".strip())
        elif a.get("name"):
            authors.append(a["name"])
    return {
        "record_id": base,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "authors": ", ".join(authors),
        "journal": server_for_doi(doi, item.get("group-title") or ""),
        "date": _date_parts(item.get("posted")) or _date_parts(item.get("created")),
        "license": lic,
        "published_doi": published,
        "published_pmid": "",
        "version": str(version),
        "source_feed": "crossref",
    }


def fetch_crossref(prefix: str, since: str = None, limit: int = None):
    """Yields rows for one DOI prefix (all, or indexed since `since`)."""
    server = CROSSREF_PREFIXES[prefix]
    doi_filter = CROSSREF_DOI_FILTER.get(prefix)
    flt = f"type:posted-content,has-abstract:true,prefix:{prefix}"
    if since:
        flt += f",from-index-date:{since}"
    params = {"filter": flt, "rows": PAGE_SIZE, "cursor": "*", "select": CROSSREF_SELECT, "mailto": MAILTO}
    seen, page = 0, 0
    while True:
        msg = _get(CROSSREF_API, params).get("message", {})
        page += 1
        items = msg.get("items", [])
        if page == 1:
            print(f"  Crossref {prefix} ({server}): {msg.get('total-results', 0):,} records")
        for item in items:
            if doi_filter and not str(item.get("DOI", "")).lower().startswith(doi_filter):
                continue
            row = crossref_item_to_row(item)
            if row:
                yield row
                seen += 1
                if limit and seen >= limit:
                    return
        cursor = msg.get("next-cursor")
        if not items or not cursor:
            return
        params["cursor"] = cursor
        time.sleep(REQUEST_GAP_S)


# ---------------------------------------------------------------------------
# Merging versions / feeds
# ---------------------------------------------------------------------------

def _version(row) -> int:
    try:
        return int(row.get("version") or 1)
    except (TypeError, ValueError):
        return 1


def merge_rows(old: dict, new: dict):
    """Returns the row to keep for one work, or None to keep `old` unchanged.
    Europe PMC beats Crossref; within a feed a newer (or equal) version wins.
    The work's date stays the earliest posting date; a known published DOI is kept."""
    if old["source_feed"] == "europepmc" and new["source_feed"] == "crossref":
        return None
    if old["source_feed"] == new["source_feed"] and _version(new) < _version(old):
        return None
    merged = dict(new)
    dates = [d for d in (old.get("date"), new.get("date")) if d]
    merged["date"] = min(dates) if dates else ""
    for col in ("published_doi", "published_pmid"):
        merged[col] = new.get(col) or old.get(col) or ""
    return merged


def doi_key(row) -> str:
    return "doi:" + doi_base(row.get("doi"))[0]


# ---------------------------------------------------------------------------
# Embedding + files (same recipe as clinicaltrials_update.py)
# ---------------------------------------------------------------------------

_MODEL = None


def embed(df: pd.DataFrame) -> np.ndarray:
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"dtype": torch.float16} if device == "cuda" else {}
        _MODEL = SentenceTransformer(MODEL_ID, device=device, model_kwargs=kwargs)
        _MODEL.max_seq_length = 512
        print(f"Embedding model on {device}")
    texts = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()
    emb = _MODEL.encode(texts, batch_size=512, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=len(texts) > 5000)
    return np.packbits(emb > 0, axis=1)


def _atomic_parquet(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df[COLUMNS].to_parquet(tmp, index=False, row_group_size=2000, compression="zstd")
    os.replace(tmp, path)


def _atomic_npy(arr: np.ndarray, path: str):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def write_chunk(df: pd.DataFrame, df_dir: str, emb_dir: str, stem: str) -> str:
    # The embedding model is English-only; non-English abstracts (mostly SciELO,
    # some PsyArXiv/OSF) would only add noise to the index.
    english = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).map(is_english)
    if (~english).any():
        print(f"  skipping {int((~english).sum()):,} non-English preprint(s)")
    df = df[english].reset_index(drop=True)
    bits = embed(df)
    path = os.path.join(df_dir, stem + ".parquet")
    _atomic_parquet(df, path)                                         # metadata first
    _atomic_npy(bits, os.path.join(emb_dir, stem + ".npy"))
    print(f"  wrote {stem}: {len(df):,} preprints")
    return path


def existing_index(df_dir: str) -> dict:
    """record_id -> (parquet path, record_id) and 'doi:<base>' -> (path, record_id)."""
    index = {}
    for path in sorted(glob.glob(os.path.join(df_dir, "*.parquet"))):
        df = pd.read_parquet(path, columns=["record_id", "doi"])
        for rid, doi in zip(df["record_id"], df["doi"]):
            index[rid] = (path, rid)
            index[doi_key({"doi": doi})] = (path, rid)
    return index


def apply_updates(updates: dict, df_dir: str, emb_dir: str) -> tuple:
    """updates: stored record_id -> (path, new row). Rows are merged with
    merge_rows; only rows whose text changed are re-embedded.
    Returns (rows changed, rows re-embedded)."""
    by_file: dict = {}
    for rid, (path, row) in updates.items():
        by_file.setdefault(path, {})[rid] = row
    updated = reembedded = 0
    for path, rows in by_file.items():
        df = pd.read_parquet(path)
        pos = {rid: i for i, rid in enumerate(df["record_id"])}
        text_changed, changed = [], False
        for rid, row in rows.items():
            if rid not in pos:
                continue
            i = pos[rid]
            merged = merge_rows({c: df.at[i, c] for c in COLUMNS}, row)
            if merged is None or all(df.at[i, c] == merged[c] for c in COLUMNS):
                continue
            if any(df.at[i, c] != merged[c] for c in TEXT_COLUMNS):
                text_changed.append(i)
            for c in COLUMNS:
                df.at[i, c] = merged[c]
            changed = True
            updated += 1
        if not changed:
            continue
        _atomic_parquet(df, path)
        if text_changed:
            npy = os.path.join(emb_dir, os.path.basename(path)[:-8] + ".npy")
            bits = np.load(npy)
            bits[text_changed] = embed(df.iloc[text_changed])
            _atomic_npy(bits, npy)
            reembedded += len(text_changed)
    return updated, reembedded


class Collector:
    """Routes fetched rows: new works are buffered (versions merged in the buffer)
    and flushed as chunks; rows for works already on disk become updates."""

    def __init__(self, known: dict, df_dir: str, emb_dir: str, run_id: int):
        self.known, self.df_dir, self.emb_dir, self.run_id = known, df_dir, emb_dir, run_id
        self.buffer: dict = {}         # record_id -> row
        self.buffer_doi: dict = {}     # doi key -> record_id
        self.updates: dict = {}        # stored record_id -> (path, row)
        self.chunks = 0
        self.new_rows = 0

    def add(self, row: dict):
        rid, dkey = row["record_id"], doi_key(row)
        hit = self.known.get(rid) or self.known.get(dkey)
        if hit:
            path, stored = hit
            prev = self.updates.get(stored, (path, None))[1]
            merged = merge_rows(prev, row) if prev else row
            if merged is not None:
                self.updates[stored] = (path, merged)
            return
        key = rid if rid in self.buffer else self.buffer_doi.get(dkey)
        if key is not None:
            merged = merge_rows(self.buffer[key], row)
            if merged is not None:
                new_key = merged["record_id"]
                if new_key != key:
                    # Europe PMC replacing a Crossref row (rare): re-key the buffer.
                    del self.buffer[key]
                    for k, v in list(self.buffer_doi.items()):
                        if v == key:
                            self.buffer_doi[k] = new_key
                self.buffer[new_key] = merged
                self.buffer_doi[dkey] = new_key
            return
        self.buffer[rid] = row
        self.buffer_doi[dkey] = rid
        if len(self.buffer) >= CHUNK_ROWS:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        rows = list(self.buffer.values())
        fill_published_dois(rows)
        path = write_chunk(pd.DataFrame(rows, columns=COLUMNS), self.df_dir, self.emb_dir,
                           f"preprints_{self.run_id}_chunk_{self.chunks}")
        for r in rows:
            self.known[r["record_id"]] = (path, r["record_id"])
            self.known[doi_key(r)] = (path, r["record_id"])
        self.chunks += 1
        self.new_rows += len(rows)
        self.buffer, self.buffer_doi = {}, {}

    def finish(self) -> tuple:
        """Flushes new rows, applies updates; returns (rows updated, rows re-embedded)."""
        self.flush()
        rows = [row for _, row in self.updates.values()]
        fill_published_dois(rows)
        return apply_updates(self.updates, self.df_dir, self.emb_dir) if self.updates else (0, 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE, help="data root (default: the pipeline's snowflake dir)")
    ap.add_argument("--full", action="store_true", help="download everything (first run)")
    ap.add_argument("--since", help="YYYY-MM-DD; default: date of the last run")
    ap.add_argument("--limit", type=int,
                    help="testing: keep at most N records per Europe PMC licence query and per Crossref prefix")
    ap.add_argument("--feeds", default="europepmc,crossref", help="comma list: europepmc,crossref")
    ap.add_argument("--prefixes", default=",".join(CROSSREF_PREFIXES), help="Crossref DOI prefixes")
    args = ap.parse_args()

    df_dir = os.path.join(args.base, "preprints_df")
    emb_dir = os.path.join(args.base, "preprints_embed")
    state_path = os.path.join(args.base, "preprints_coordination", "state.json")
    for d in (df_dir, emb_dir, os.path.dirname(state_path)):
        os.makedirs(d, exist_ok=True)

    known = existing_index(df_dir)
    since = None
    if not args.full:
        if args.since:
            since = args.since
        elif os.path.exists(state_path):
            with open(state_path) as f:
                since = json.load(f)["last_run"]
        elif known:
            raise SystemExit("No state file; pass --since YYYY-MM-DD or --full.")
        else:
            args.full = True
    started = date.today().isoformat()
    n_known = sum(1 for k in known if not k.startswith("doi:"))
    print(f"=== Preprints {'full download' if args.full else f'updates since {since}'} "
          f"({n_known:,} preprints already indexed) ===")

    feeds = [f.strip() for f in args.feeds.split(",") if f.strip()]
    collector = Collector(known, df_dir, emb_dir, int(time.time()))
    if "europepmc" in feeds:          # first: Europe PMC wins duplicates
        for row in fetch_europepmc(since=since, limit=args.limit):
            collector.add(row)
        collector.flush()
    if "crossref" in feeds:
        for prefix in [p.strip() for p in args.prefixes.split(",") if p.strip()]:
            for row in fetch_crossref(prefix, since=since, limit=args.limit):
                collector.add(row)
    updated, reembedded = collector.finish()

    with open(state_path + ".tmp", "w") as f:
        json.dump({"last_run": started, "finished": datetime.now().isoformat(timespec="seconds")}, f)
    os.replace(state_path + ".tmp", state_path)
    total = sum(len(pd.read_parquet(p, columns=["record_id"])) for p in glob.glob(os.path.join(df_dir, "*.parquet")))
    print(f"Done: {collector.chunks} new chunk(s) with {collector.new_rows:,} preprints, "
          f"{updated:,} updated ({reembedded:,} re-embedded); {total:,} preprints indexed.")


if __name__ == "__main__":
    main()
