"""
NIH RePORTER project abstracts → Manuscript Search (source "Grants").

Writes the same layout as the other sources: parquet chunks in <base>/grants_df/
and row-aligned binary embeddings in <base>/grants_embed/ (BAAI/bge-small-en-v1.5,
"title. abstract", no query prefix, np.packbits(emb > 0) -> 48 bytes).

    python nih_grants_update.py --full              # first run: ~0.6M projects, ~25 min
    python nih_grants_update.py                     # weekly: applications added since last run
    python nih_grants_update.py --base /tmp/x --full --years 2025 --limit 2000   # small test

Bulk path: the ExPORTER yearly CSV zips (projects + abstracts, FY1985 onward,
~60 MB each, served in seconds) — far cheaper than the RePORTER API v2 (500
records per call, offset <= 14,999, ~1 call/s). The API is used only for fiscal
years that have no ExPORTER file yet (the current FY) and for incremental runs
(criteria.date_added; RePORTER loads new applications weekly, on Saturdays).
API queries that match more than 15,000 records are split by date_added range.

One row per core project number (e.g. R01GM123456): RePORTER lists one
application per fiscal year, supplement and renewal. Kept: the most recent
fiscal year's record with an abstract (parent award preferred over type-3
supplements), with start_date = earliest project start and end_date = latest
project end over all its years. Sub-projects of centre grants (SUBPROJECT_ID
set; "Core B", "Project 2", ...) are dropped: they share the parent's core
number. `date` (used by date filters) = award notice date of the kept, most
recent application (fallback: its budget start, project start, then the start
of its fiscal year), i.e. "when this project was last funded"; start_date keeps
the original project start.

Source: NIH RePORTER (US government data; includes AHRQ, CDC, FDA, VA awards).
Files are written to a temp name and renamed, parquet before .npy.
"""
import argparse
import glob
import io
import json
import os
import re
import time
import zipfile
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

EXPORTER_LIST = "https://reporter.nih.gov/services/exporter/allFilesInfo"
EXPORTER_FILE = "https://reporter.nih.gov/services/exporter/DownloadFromDocService"
API = "https://api.reporter.nih.gov/v2/projects/search"
USER_AGENT = "ManuscriptSearch/2.0 (https://manuscript-search.org; mailto:dawid.zyla@cuanschutz.edu)"
DEFAULT_BASE = "/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake"
MODEL_ID = "BAAI/bge-small-en-v1.5"
CHUNK_ROWS = 25_000
API_PAGE = 500
API_MAX_RESULTS = 15_000       # offset <= 14,999
REQUEST_GAP_S = 1.0            # RePORTER asks for at most ~1 request per second
MIN_ABSTRACT_CHARS = 50
API_FIELDS = ["ApplId", "SubprojectId", "FiscalYear", "ProjectNum", "CoreProjectNum", "ProjectTitle",
              "AbstractText", "PrincipalInvestigators", "Organization", "AgencyIcAdmin", "AwardNoticeDate",
              "BudgetStart", "ProjectStartDate", "ProjectEndDate", "AwardAmount", "ActivityCode",
              "AwardType", "DateAdded"]
# Administering IC code -> usual abbreviation (for "journal"); unknown codes stay as-is.
IC_ABBREV = {
    "AA": "NIAAA", "AG": "NIA", "AI": "NIAID", "AR": "NIAMS", "AT": "NCCIH", "CA": "NCI", "DA": "NIDA",
    "DC": "NIDCD", "DE": "NIDCR", "DK": "NIDDK", "EB": "NIBIB", "ES": "NIEHS", "EY": "NEI", "GM": "NIGMS",
    "HD": "NICHD", "HG": "NHGRI", "HL": "NHLBI", "LM": "NLM", "MD": "NIMHD", "MH": "NIMH", "NR": "NINR",
    "NS": "NINDS", "OD": "NIH OD", "RM": "NIH Common Fund", "RR": "NCRR", "TR": "NCATS", "TW": "FIC",
    "CL": "CC", "CIT": "CIT", "WH": "NIH OD", "VA": "VA", "FD": "FDA", "HS": "AHRQ", "OH": "NIOSH",
}

COLUMNS = ["grant_id", "title", "abstract", "authors", "journal", "date", "start_date", "end_date",
           "fiscal_year", "organization", "org_state", "org_country", "total_cost", "ic", "activity_code",
           "application_type", "project_num", "appl_id", "doi"]
TEXT_COLUMNS = ("title", "abstract")
# Application-level frame used before deduplication.
APP_COLUMNS = ["appl_id", "grant_id", "subproject_id", "fiscal_year", "project_num", "title", "abstract",
               "pis", "organization", "org_state", "org_country", "ic", "activity_code", "application_type",
               "notice_date", "budget_start", "start_date", "end_date", "total_cost"]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _request(method: str, url: str, attempts: int = 8, **kwargs) -> requests.Response:
    headers = {"User-Agent": USER_AGENT, **kwargs.pop("headers", {})}
    for attempt in range(1, attempts + 1):
        try:
            r = requests.request(method, url, timeout=600, headers=headers, **kwargs)
            if r.status_code == 200:
                return r
            if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                raise RuntimeError(f"HTTP {r.status_code} from {url}: {r.text[:300]}")
            wait = r.headers.get("Retry-After", "")
            wait = int(wait) if wait.isdigit() else min(300, 15 * attempt)
            print(f"  HTTP {r.status_code}; retry in {wait}s")
        except requests.RequestException as exc:
            wait = min(300, 15 * attempt)
            print(f"  request failed ({exc}); retry in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"{url} kept failing; try again later.")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_ABS_HEADER = re.compile(
    r"\s*(?P<h>(?:project\s+summary|abstract|description|summary|overall|project\s+description|"
    r"project\s+narrative|narrative)(?:\s*/\s*(?:abstract|summary))?"
    r"(?:\s*\((?:provided\s+by|see)[^)]{0,40}\))?)", re.IGNORECASE)
_MODIFIED_HEADER = re.compile(r"^\s*modified\s+project\s+summary\s*/\s*abstract\s+section\s*[:.\-]?\s*",
                              re.IGNORECASE)
_HEADER_END = re.compile(r"[ \t]*(?:[:.\-\u2013]+|\r?\n)\s*")


def clean_abstract(text) -> str:
    """Drops leading 'PROJECT SUMMARY/ABSTRACT', 'DESCRIPTION (provided by applicant):'
    headings and '[unreadable]' noise; collapses whitespace. A heading word is removed
    only when followed by ':', '.', '-', a line break, or when written in capitals
    (so 'Overall, this project ...' is kept)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text).replace("[unreadable]", " ")
    text = re.sub(r"^[^\w\[(]+", "", text)          # stray leading bytes/punctuation
    text = _MODIFIED_HEADER.sub("", text)
    for _ in range(3):
        m = _ABS_HEADER.match(text)
        if not m:
            break
        rest = text[m.end("h"):]
        end = _HEADER_END.match(rest)
        if end:
            text = rest[end.end():]
        elif m.group("h").isupper() and rest[:1].isspace():
            text = rest.lstrip()
        else:
            break
    return re.sub(r"\s+", " ", text).strip()


def _clean(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _iso(series: pd.Series) -> pd.Series:
    """Mixed ExPORTER/API date strings ('7/1/1985', '2000-09-27T00:00:00') -> 'YYYY-MM-DD' or ''."""
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed.dt.strftime("%Y-%m-%d").fillna("")


def format_pis(value) -> str:
    """'SMITH, JOHN A (contact); DOE, JANE;' -> 'John A Smith; Jane Doe'."""
    names = []
    for part in str(value or "").split(";"):
        part = re.sub(r"\(contact\)", "", part, flags=re.IGNORECASE).strip(" ,")
        if not part or part.lower() == "nan":
            continue
        if "," in part:
            last, first = [p.strip() for p in part.split(",", 1)]
            part = f"{first} {last}".strip()
        names.append(part.title() if part.isupper() else part)
    return "; ".join(names)


# ---------------------------------------------------------------------------
# ExPORTER (bulk CSV)
# ---------------------------------------------------------------------------

def exporter_years() -> list:
    """Fiscal years with both a project and an abstract ExPORTER file."""
    years = {}
    for group, kind in (("PROJECT", "EXPPRJ"), ("ABSTRACT", "EXPABS")):
        files = _request("POST", EXPORTER_LIST, json={"file_group": group}).json()
        years[kind] = {int(f["fy"]) for f in files
                       if f.get("doc_type_code") == kind and not f.get("to_fy")}
    return sorted(years["EXPPRJ"] & years["EXPABS"])


_BAD_BYTES = re.compile(r"[\udc80-\udcff]+")


def decode_mixed(raw: bytes) -> str:
    """ExPORTER CSVs are UTF-8, but some (e.g. FY2015 abstracts) contain stray
    CP1252 bytes; decode UTF-8 and re-read only the invalid byte runs as CP1252."""
    text = raw.decode("utf-8", errors="surrogateescape")
    return _BAD_BYTES.sub(lambda m: bytes(ord(c) - 0xDC00 for c in m.group())
                          .decode("cp1252", errors="replace"), text)


def _read_csv(content: bytes, **kwargs) -> pd.DataFrame:
    """Reads the single CSV inside an ExPORTER zip (all columns as strings)."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        raw = zf.read(zf.namelist()[0])
    return pd.read_csv(io.StringIO(decode_mixed(raw)), dtype=str, **kwargs)


def download_exporter(kind: str, fy: int) -> bytes:
    r = _request("GET", EXPORTER_FILE, params={"DocType": kind, "KeyId": fy})
    time.sleep(REQUEST_GAP_S)
    return r.content


_PRJ_COLUMNS = {
    "APPLICATION_ID": "appl_id", "CORE_PROJECT_NUM": "grant_id", "SUBPROJECT_ID": "subproject_id",
    "FY": "fiscal_year", "FULL_PROJECT_NUM": "project_num", "PROJECT_TITLE": "title", "PI_NAMEs": "pis",
    "ORG_NAME": "organization", "ORG_STATE": "org_state", "ORG_COUNTRY": "org_country",
    "ADMINISTERING_IC": "ic", "ACTIVITY": "activity_code", "APPLICATION_TYPE": "application_type",
    "AWARD_NOTICE_DATE": "notice_date", "BUDGET_START": "budget_start", "PROJECT_START": "start_date",
    "PROJECT_END": "end_date", "TOTAL_COST": "total_cost", "TOTAL_COST_SUB_PROJECT": "total_cost_sub",
}


def exporter_projects(prj_zip: bytes) -> pd.DataFrame:
    """ExPORTER project CSV -> application frame (APP_COLUMNS, abstract still empty)."""
    raw = _read_csv(prj_zip, usecols=lambda c: c in _PRJ_COLUMNS)
    df = raw.rename(columns=_PRJ_COLUMNS)
    for col in _PRJ_COLUMNS.values():
        if col not in df:
            df[col] = None
    df["total_cost"] = pd.to_numeric(df["total_cost"], errors="coerce")
    df["abstract"] = ""
    return standardize(df)


def exporter_abstracts(abs_zip: bytes, appl_ids=None) -> pd.Series:
    """appl_id -> cleaned abstract (only for `appl_ids` if given)."""
    df = _read_csv(abs_zip)
    df.columns = [c.strip().upper() for c in df.columns]
    df["APPLICATION_ID"] = df["APPLICATION_ID"].str.strip()
    if appl_ids is not None:
        df = df[df["APPLICATION_ID"].isin(set(appl_ids))]
    df = df.drop_duplicates("APPLICATION_ID")
    return pd.Series([clean_abstract(t) for t in df["ABSTRACT_TEXT"]], index=df["APPLICATION_ID"].values)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({c: df[c] if c in df else None for c in APP_COLUMNS})
    for col in ("appl_id", "grant_id", "subproject_id", "project_num", "ic", "activity_code",
                "application_type", "org_state", "org_country"):
        out[col] = out[col].map(_clean)
    out["title"] = out["title"].map(_clean)
    out["organization"] = out["organization"].map(_clean)
    out["pis"] = out["pis"].map(lambda v: format_pis(_clean(v)))
    out["abstract"] = out["abstract"].fillna("").astype(str)
    out["fiscal_year"] = pd.to_numeric(out["fiscal_year"], errors="coerce").fillna(0).astype(int)
    out["total_cost"] = pd.to_numeric(out["total_cost"], errors="coerce").astype(float)
    for col in ("notice_date", "budget_start", "start_date", "end_date"):
        out[col] = _iso(out[col])
    return out[out["grant_id"] != ""].reset_index(drop=True)


# ---------------------------------------------------------------------------
# RePORTER API v2
# ---------------------------------------------------------------------------

def api_record_to_app(rec: dict) -> dict:
    org = rec.get("organization") or {}
    pis = rec.get("principal_investigators") or []
    pis = sorted(pis, key=lambda p: not p.get("is_contact_pi"))
    names = []
    for p in pis:
        name = " ".join(x for x in (p.get("first_name"), p.get("middle_name"), p.get("last_name")) if x)
        name = re.sub(r"\s+", " ", name).strip()
        names.append(name.title() if name.isupper() else name)
    return {
        "appl_id": str(rec.get("appl_id") or ""),
        "grant_id": rec.get("core_project_num") or "",
        "subproject_id": str(rec.get("subproject_id") or ""),
        "fiscal_year": rec.get("fiscal_year"),
        "project_num": rec.get("project_num") or "",
        "title": rec.get("project_title") or "",
        "abstract": clean_abstract(rec.get("abstract_text")),
        "pis": "; ".join(names),
        "organization": org.get("org_name") or "",
        "org_state": org.get("org_state") or "",
        "org_country": org.get("org_country") or "",
        "ic": (rec.get("agency_ic_admin") or {}).get("code") or "",
        "activity_code": rec.get("activity_code") or "",
        "application_type": str(rec.get("award_type") or ""),
        "notice_date": rec.get("award_notice_date") or "",
        "budget_start": rec.get("budget_start") or "",
        "start_date": rec.get("project_start_date") or "",
        "end_date": rec.get("project_end_date") or "",
        "total_cost": rec.get("award_amount"),
    }


def _api_search(criteria: dict, offset: int, limit: int) -> dict:
    payload = {"criteria": criteria, "include_fields": API_FIELDS, "offset": offset, "limit": limit,
               "sort_field": "appl_id", "sort_order": "asc"}
    data = _request("POST", API, json=payload).json()
    time.sleep(REQUEST_GAP_S)
    return data


def fetch_api(criteria: dict, from_date: str, to_date: str, limit: int = None):
    """Yields application dicts matching `criteria` with date_added in [from, to];
    ranges matching more than 15,000 records are split in halves."""
    crit = dict(criteria, date_added={"from_date": from_date, "to_date": to_date})
    first = _api_search(crit, 0, API_PAGE)
    total = first.get("meta", {}).get("total", 0)
    d0, d1 = date.fromisoformat(from_date), date.fromisoformat(to_date)
    if total > API_MAX_RESULTS and d1 > d0:
        mid = d0 + (d1 - d0) // 2
        seen = 0
        for part in ((d0, mid), (mid + timedelta(days=1), d1)):
            for rec in fetch_api(criteria, part[0].isoformat(), part[1].isoformat(),
                                 None if limit is None else limit - seen):
                yield rec
                seen += 1
                if limit and seen >= limit:
                    return
        return
    if total > API_MAX_RESULTS:
        print(f"  warning: {total:,} records added on {from_date}; only the first {API_MAX_RESULTS:,} reachable")
    print(f"  API date_added {from_date}..{to_date}: {total:,} applications")
    seen, offset, data = 0, 0, first
    while True:
        results = data.get("results") or []
        for rec in results:
            yield api_record_to_app(rec)
            seen += 1
            if limit and seen >= limit:
                return
        offset += API_PAGE
        if not results or offset >= min(total, API_MAX_RESULTS):
            return
        data = _api_search(crit, offset, min(API_PAGE, API_MAX_RESULTS - offset))


def api_apps(criteria: dict, from_date: str, to_date: str, limit: int = None) -> pd.DataFrame:
    rows = list(fetch_api(criteria, from_date, to_date, limit))
    return standardize(pd.DataFrame(rows, columns=APP_COLUMNS))


# ---------------------------------------------------------------------------
# Deduplication to one row per core project
# ---------------------------------------------------------------------------

def _rank(df: pd.DataFrame) -> pd.DataFrame:
    """Sort key columns: best application last."""
    return df.assign(_has_abs=(df["abstract"].str.len() >= MIN_ABSTRACT_CHARS).astype(int),
                     _not_supp=(df["application_type"] != "3").astype(int))


_RANK_KEY = ["_has_abs", "fiscal_year", "_not_supp", "notice_date", "appl_id"]


def reduce_apps(apps: pd.DataFrame) -> pd.DataFrame:
    """Application frame -> one row per grant_id (parent awards only): the best
    application (abstract, newest FY, not a supplement, latest notice), with the
    earliest start and latest end over all applications."""
    apps = apps[apps["subproject_id"] == ""]
    if apps.empty:
        return _rank(apps).set_index("grant_id")
    ranked = _rank(apps).sort_values(_RANK_KEY)
    best = ranked.drop_duplicates("grant_id", keep="last").set_index("grant_id")
    starts = apps.loc[apps["start_date"] != ""].groupby("grant_id")["start_date"].min()
    ends = apps.loc[apps["end_date"] != ""].groupby("grant_id")["end_date"].max()
    best["start_date"] = starts.reindex(best.index).fillna(best["start_date"])
    best["end_date"] = ends.reindex(best.index).fillna(best["end_date"])
    return best


def combine(older: pd.DataFrame, newer: pd.DataFrame) -> pd.DataFrame:
    """Merges two reduced frames (indexed by grant_id) with the reduce_apps rules."""
    both = pd.concat([older.reset_index(), newer.reset_index()], ignore_index=True)
    starts = both.loc[both["start_date"] != ""].groupby("grant_id")["start_date"].min()
    ends = both.loc[both["end_date"] != ""].groupby("grant_id")["end_date"].max()
    best = _rank(both).sort_values(_RANK_KEY).drop_duplicates("grant_id", keep="last").set_index("grant_id")
    best["start_date"] = starts.reindex(best.index).fillna(best["start_date"])
    best["end_date"] = ends.reindex(best.index).fillna(best["end_date"])
    return best


def to_rows(best: pd.DataFrame) -> pd.DataFrame:
    """Reduced frame -> output COLUMNS (drops projects without an abstract)."""
    best = best[best["abstract"].str.len() >= MIN_ABSTRACT_CHARS].copy()
    fy_start = (best["fiscal_year"] - 1).astype(str) + "-10-01"
    when = best["notice_date"].where(best["notice_date"] != "", best["budget_start"])
    when = when.where(when != "", best["start_date"])
    when = when.where(when != "", fy_start)
    ic = best["ic"].fillna("")
    out = pd.DataFrame({
        "grant_id": best.index.astype(str),
        "title": best["title"].values,
        "abstract": best["abstract"].values,
        "authors": best["pis"].values,
        "journal": ("NIH RePORTER (" + ic.map(lambda c: IC_ABBREV.get(c, c)) + ")").str.replace(" ()", "", regex=False).values,
        "date": when.values,
        "start_date": best["start_date"].values,
        "end_date": best["end_date"].values,
        "fiscal_year": best["fiscal_year"].astype(int).values,
        "organization": best["organization"].values,
        "org_state": best["org_state"].values,
        "org_country": best["org_country"].values,
        "total_cost": best["total_cost"].astype(float).values,
        "ic": ic.values,
        "activity_code": best["activity_code"].values,
        "application_type": best["application_type"].values,
        "project_num": best["project_num"].values,
        "appl_id": best["appl_id"].values,
        "doi": "",
    })
    return out[out["title"] != ""].reset_index(drop=True)


def rows_to_reduced(rows: pd.DataFrame) -> pd.DataFrame:
    """Stored output rows -> reduced frame (inverse of to_rows, for incremental merges)."""
    return pd.DataFrame({
        "appl_id": rows["appl_id"].values, "subproject_id": "", "fiscal_year": rows["fiscal_year"].astype(int).values,
        "project_num": rows["project_num"].values, "title": rows["title"].values,
        "abstract": rows["abstract"].values, "pis": rows["authors"].values,
        "organization": rows["organization"].values, "org_state": rows["org_state"].values,
        "org_country": rows["org_country"].values, "ic": rows["ic"].values,
        "activity_code": rows["activity_code"].values, "application_type": rows["application_type"].values,
        "notice_date": rows["date"].values, "budget_start": "", "start_date": rows["start_date"].values,
        "end_date": rows["end_date"].values, "total_cost": rows["total_cost"].values,
    }, index=pd.Index(rows["grant_id"].values, name="grant_id"))


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


def write_chunks(df: pd.DataFrame, df_dir: str, emb_dir: str, run_id: int) -> int:
    n = 0
    for n, start in enumerate(range(0, len(df), CHUNK_ROWS)):
        part = df.iloc[start:start + CHUNK_ROWS].reset_index(drop=True)
        stem = f"grants_{run_id}_chunk_{n}"
        bits = embed(part)
        _atomic_parquet(part, os.path.join(df_dir, stem + ".parquet"))     # metadata first
        _atomic_npy(bits, os.path.join(emb_dir, stem + ".npy"))
        print(f"  wrote {stem}: {len(part):,} grants")
    return (n + 1) if len(df) else 0


def existing_index(df_dir: str) -> dict:
    """grant_id -> parquet path."""
    index = {}
    for path in sorted(glob.glob(os.path.join(df_dir, "*.parquet"))):
        for gid in pd.read_parquet(path, columns=["grant_id"])["grant_id"]:
            index[gid] = path
    return index


def apply_updates(reduced: pd.DataFrame, known: dict, df_dir: str, emb_dir: str) -> tuple:
    """Merges newly fetched applications (reduced, indexed by grant_id) into the
    stored rows of known grants; re-embeds only rows whose text changed.
    Returns (rows changed, rows re-embedded)."""
    by_file: dict = {}
    for gid in reduced.index:
        by_file.setdefault(known[gid], []).append(gid)
    updated = reembedded = 0
    for path, gids in by_file.items():
        df = pd.read_parquet(path)
        pos = {g: i for i, g in enumerate(df["grant_id"])}
        stored = df.iloc[[pos[g] for g in gids]]
        merged = to_rows(combine(rows_to_reduced(stored), reduced.loc[gids]))
        text_changed, changed = [], 0
        for _, row in merged.iterrows():
            i = pos[row["grant_id"]]
            if all(df.at[i, c] == row[c] or (pd.isna(df.at[i, c]) and pd.isna(row[c])) for c in COLUMNS):
                continue
            if any(df.at[i, c] != row[c] for c in TEXT_COLUMNS):
                text_changed.append(i)
            for c in COLUMNS:
                df.at[i, c] = row[c]
            changed += 1
        if not changed:
            continue
        updated += changed
        _atomic_parquet(df, path)
        if text_changed:
            npy = os.path.join(emb_dir, os.path.basename(path)[:-8] + ".npy")
            bits = np.load(npy)
            bits[text_changed] = embed(df.iloc[text_changed])
            _atomic_npy(bits, npy)
            reembedded += len(text_changed)
    return updated, reembedded


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def full_download(years=None, limit: int = None) -> pd.DataFrame:
    """All grants: API for fiscal years without an ExPORTER file, then ExPORTER,
    newest year first. Returns output rows."""
    file_years = exporter_years()
    print(f"ExPORTER yearly files: FY{file_years[0]}-FY{file_years[-1]}")
    current_fy = date.today().year + (1 if date.today().month >= 10 else 0)
    api_years = [y for y in range(file_years[-1] + 1, current_fy + 1) if not years or y in years]
    best = None
    for fy in reversed(api_years):
        print(f"FY{fy}: RePORTER API")
        apps = api_apps({"fiscal_years": [fy]}, "1985-01-01", date.today().isoformat(), limit)
        reduced = reduce_apps(apps)
        best = reduced if best is None else combine(best, reduced)
    for fy in reversed(file_years):
        if years and fy not in years:
            continue
        t0 = time.time()
        apps = exporter_projects(download_exporter("EXPPRJ", fy))
        apps = apps[apps["subproject_id"] == ""]
        # Abstracts only for applications that could still be kept.
        need = apps["appl_id"]
        if best is not None:
            have_text = best.index[best["abstract"].str.len() >= MIN_ABSTRACT_CHARS]
            need = apps.loc[~apps["grant_id"].isin(have_text), "appl_id"]
        abstracts = exporter_abstracts(download_exporter("EXPABS", fy), need)
        apps["abstract"] = apps["appl_id"].map(abstracts).fillna("")
        reduced = reduce_apps(apps)
        best = reduced if best is None else combine(best, reduced)
        print(f"FY{fy}: {len(apps):,} applications -> {len(best):,} projects so far ({time.time() - t0:.0f}s)")
    rows = to_rows(best) if best is not None else pd.DataFrame(columns=COLUMNS)
    rows = rows.sort_values(["date", "grant_id"], ascending=False).reset_index(drop=True)
    return rows.head(limit) if limit else rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE, help="data root (default: the pipeline's snowflake dir)")
    ap.add_argument("--full", action="store_true", help="rebuild from all ExPORTER files (first run)")
    ap.add_argument("--since", help="YYYY-MM-DD; default: date of the last run")
    ap.add_argument("--limit", type=int, help="stop after N grants / applications (testing)")
    ap.add_argument("--years", help="--full only: fiscal years to load, e.g. 2025 or 2020-2026 (testing)")
    args = ap.parse_args()

    df_dir = os.path.join(args.base, "grants_df")
    emb_dir = os.path.join(args.base, "grants_embed")
    state_path = os.path.join(args.base, "grants_coordination", "state.json")
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
    if args.full and known:
        raise SystemExit(f"{df_dir} already holds {len(known):,} grants; --full rebuilds from scratch, "
                         "so move the old grants_df/grants_embed away first.")
    years = None
    if args.years:
        lo, _, hi = args.years.partition("-")
        years = set(range(int(lo), int(hi or lo) + 1))
    started = date.today().isoformat()
    print(f"=== NIH RePORTER {'full download' if args.full else f'updates since {since}'} "
          f"({len(known):,} grants already indexed) ===")

    run_id = int(time.time())
    chunks = updated = reembedded = 0
    if args.full:
        rows = full_download(years, args.limit)
        chunks = write_chunks(rows, df_dir, emb_dir, run_id)
        new = len(rows)
    else:
        apps = api_apps({}, since, date.today().isoformat(), args.limit)
        reduced = reduce_apps(apps)
        old = reduced[reduced.index.isin(list(known))]
        fresh = to_rows(reduced[~reduced.index.isin(list(known))])
        chunks = write_chunks(fresh, df_dir, emb_dir, run_id)
        new = len(fresh)
        if len(old):
            updated, reembedded = apply_updates(old, known, df_dir, emb_dir)

    with open(state_path + ".tmp", "w") as f:
        json.dump({"last_run": started, "finished": datetime.now().isoformat(timespec="seconds")}, f)
    os.replace(state_path + ".tmp", state_path)
    total = sum(len(pd.read_parquet(p, columns=["grant_id"])) for p in glob.glob(os.path.join(df_dir, "*.parquet")))
    print(f"Done: {chunks} new chunk(s) with {new:,} grants, {updated:,} updated "
          f"({reembedded:,} re-embedded); {total:,} grants indexed.")


if __name__ == "__main__":
    main()
