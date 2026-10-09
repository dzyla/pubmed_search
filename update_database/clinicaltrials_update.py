"""
ClinicalTrials.gov → Manuscript Search (API v2, public US-government data).

Writes the same layout as the other sources: parquet chunks in
<base>/clinicaltrials_df/ and row-aligned binary embeddings in
<base>/clinicaltrials_embed/ (BAAI/bge-small-en-v1.5, "title. brief summary",
no query prefix, np.packbits(emb > 0) -> 48 bytes).

    python clinicaltrials_update.py --full            # first run: all ~600k trials
    python clinicaltrials_update.py                   # weekly: changes since last run
    python clinicaltrials_update.py --base /tmp/x --limit 2000 --full   # small test

Incremental runs ask the API for studies whose LastUpdatePostDate is on or
after the last run: new trials go into a new chunk; for known trials the
metadata (status, phase, dates, results flag, ...) is updated in place, and
only trials whose title or summary changed are re-embedded. Files are written
to a temp name and renamed, parquet before .npy, so a sync never ships an
embedding file without its metadata.

Source: ClinicalTrials.gov, U.S. National Library of Medicine (attribution).
"""
import argparse
import glob
import json
import os
import re
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests

API = "https://clinicaltrials.gov/api/v2/studies"
USER_AGENT = "ManuscriptSearch/2.0 (https://manuscript-search.org; mailto:dawid.zyla@cuanschutz.edu)"
DEFAULT_BASE = "/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake"
MODEL_ID = "BAAI/bge-small-en-v1.5"
CHUNK_ROWS = 25_000
PAGE_SIZE = 1000
REQUEST_GAP_S = 1.3          # ClinicalTrials.gov asks for ~50 requests/minute at most
FIELDS = ",".join([
    "NCTId", "BriefTitle", "OfficialTitle", "Acronym", "BriefSummary", "OverallStatus",
    "StudyFirstPostDate", "StartDate", "CompletionDate", "LastUpdatePostDate", "Condition",
    "StudyType", "Phase", "LeadSponsorName", "HasResults", "InterventionName", "ReferencePMID",
])
COLUMNS = ["nct_id", "title", "abstract", "authors", "journal", "date", "start_date",
           "completion_date", "last_update", "trial_status", "trial_phase", "study_type",
           "conditions", "interventions", "has_results", "pmids", "doi"]
TEXT_COLUMNS = ("title", "abstract")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _get(params: dict, attempts: int = 8) -> dict:
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(API, params=params, timeout=120, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return r.json()
            wait = int(r.headers.get("Retry-After", min(300, 15 * attempt)))
            print(f"  HTTP {r.status_code}; retry in {wait}s")
        except (requests.RequestException, ValueError) as exc:
            wait = min(300, 15 * attempt)
            print(f"  request failed ({exc}); retry in {wait}s")
        time.sleep(wait)
    raise RuntimeError("ClinicalTrials.gov API kept failing; try again later.")


def _full_date(value: str) -> str:
    """'2008-11' -> '2008-11-01', '2008' -> '2008-01-01', '' stays ''."""
    value = (value or "").strip()
    if re.fullmatch(r"\d{4}", value):
        return value + "-01-01"
    if re.fullmatch(r"\d{4}-\d{2}", value):
        return value + "-01"
    return value


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _pretty(code: str) -> str:
    """'ACTIVE_NOT_RECRUITING' -> 'Active, not recruiting'; 'PHASE2' -> 'Phase 2'."""
    code = str(code or "")
    if code.startswith("PHASE"):
        return "Phase " + code[5:]
    if code == "EARLY_PHASE1":
        return "Early phase 1"
    if code == "NA":
        return ""
    text = code.replace("_", " ").lower().capitalize()
    return text.replace("Active not recruiting", "Active, not recruiting")


def study_to_row(study: dict) -> dict:
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    design = p.get("designModule", {})
    conds = p.get("conditionsModule", {})
    title = ident.get("briefTitle") or ident.get("officialTitle") or ""
    if ident.get("acronym") and ident["acronym"] not in title:
        title = f"{title} ({ident['acronym']})"
    return {
        "nct_id": ident.get("nctId", ""),
        "title": _clean(title),
        "abstract": _clean(p.get("descriptionModule", {}).get("briefSummary", "")),
        "authors": _clean(p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", "")),
        "journal": "ClinicalTrials.gov",
        # Registration date: when the trial became public. Used by date filters.
        "date": _full_date(status.get("studyFirstPostDateStruct", {}).get("date", "")),
        "start_date": _full_date(status.get("startDateStruct", {}).get("date", "")),
        "completion_date": _full_date(status.get("completionDateStruct", {}).get("date", "")),
        "last_update": _full_date(status.get("lastUpdatePostDateStruct", {}).get("date", "")),
        "trial_status": _pretty(status.get("overallStatus", "")),
        "trial_phase": "/".join(filter(None, (_pretty(x) for x in design.get("phases", [])))),
        "study_type": _pretty(design.get("studyType", "")),
        "conditions": "; ".join(conds.get("conditions", [])),
        "interventions": "; ".join(i.get("name", "") for i in
                                   p.get("armsInterventionsModule", {}).get("interventions", [])),
        "has_results": bool(study.get("hasResults", False)),
        "pmids": "; ".join(r["pmid"] for r in p.get("referencesModule", {}).get("references", [])
                           if r.get("pmid")),
        "doi": "",
    }


def fetch(since: str = None, limit: int = None):
    """Yields rows for all studies (or those updated since `since`)."""
    params = {"pageSize": PAGE_SIZE, "fields": FIELDS, "countTotal": "true", "format": "json"}
    if since:
        params["filter.advanced"] = f"AREA[LastUpdatePostDate]RANGE[{since},MAX]"
    seen, page = 0, 0
    while True:
        data = _get(params)
        page += 1
        studies = data.get("studies", [])
        for s in studies:
            yield study_to_row(s)
            seen += 1
            if limit and seen >= limit:
                return
        total = data.get("totalCount")
        if page == 1 or page % 25 == 0:
            print(f"  page {page}: {seen:,}{f' of {total:,}' if total else ''} studies")
        token = data.get("nextPageToken")
        if not token:
            return
        params = {k: v for k, v in params.items() if k != "countTotal"}
        params["pageToken"] = token
        time.sleep(REQUEST_GAP_S)


# ---------------------------------------------------------------------------
# Embedding + files
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


def write_chunk(df: pd.DataFrame, df_dir: str, emb_dir: str, stem: str):
    df = df.reset_index(drop=True)
    bits = embed(df)
    _atomic_parquet(df, os.path.join(df_dir, stem + ".parquet"))     # metadata first
    _atomic_npy(bits, os.path.join(emb_dir, stem + ".npy"))
    print(f"  wrote {stem}: {len(df):,} trials")


def existing_index(df_dir: str) -> dict:
    """nct_id -> parquet path."""
    index = {}
    for path in sorted(glob.glob(os.path.join(df_dir, "*.parquet"))):
        for nct in pd.read_parquet(path, columns=["nct_id"])["nct_id"]:
            index[nct] = path
    return index


def apply_updates(updates: dict, df_dir: str, emb_dir: str) -> int:
    """Updates known trials in place; re-embeds only rows whose text changed."""
    by_file: dict = {}
    for nct, (path, row) in updates.items():
        by_file.setdefault(path, {})[nct] = row
    reembedded = 0
    for path, rows in by_file.items():
        df = pd.read_parquet(path)
        pos = {nct: i for i, nct in enumerate(df["nct_id"])}
        text_changed = []
        for nct, row in rows.items():
            i = pos[nct]
            if any(df.at[i, c] != row[c] for c in TEXT_COLUMNS):
                text_changed.append(i)
            for c in COLUMNS:
                df.at[i, c] = row[c]
        _atomic_parquet(df, path)
        if text_changed:
            npy = os.path.join(emb_dir, os.path.basename(path)[:-8] + ".npy")
            bits = np.load(npy)
            bits[text_changed] = embed(df.iloc[text_changed])
            _atomic_npy(bits, npy)
            reembedded += len(text_changed)
    return reembedded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE, help="data root (default: the pipeline's snowflake dir)")
    ap.add_argument("--full", action="store_true", help="download every trial (first run)")
    ap.add_argument("--since", help="YYYY-MM-DD; default: date of the last run")
    ap.add_argument("--limit", type=int, help="stop after N studies (testing)")
    args = ap.parse_args()

    df_dir = os.path.join(args.base, "clinicaltrials_df")
    emb_dir = os.path.join(args.base, "clinicaltrials_embed")
    state_path = os.path.join(args.base, "clinicaltrials_coordination", "state.json")
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
    print(f"=== ClinicalTrials.gov {'full download' if args.full else f'updates since {since}'} "
          f"({len(known):,} trials already indexed) ===")

    run_id = int(time.time())
    new_rows, updates, chunk = [], {}, 0
    for row in fetch(since=None if args.full else since, limit=args.limit):
        if not row["nct_id"] or not row["title"]:
            continue
        if row["nct_id"] in known:
            updates[row["nct_id"]] = (known[row["nct_id"]], row)
            continue
        known[row["nct_id"]] = None
        new_rows.append(row)
        if len(new_rows) >= CHUNK_ROWS:
            write_chunk(pd.DataFrame(new_rows), df_dir, emb_dir, f"ctgov_{run_id}_chunk_{chunk}")
            new_rows, chunk = [], chunk + 1
    if new_rows:
        write_chunk(pd.DataFrame(new_rows), df_dir, emb_dir, f"ctgov_{run_id}_chunk_{chunk}")
        chunk += 1

    updates = {k: v for k, v in updates.items() if v[0]}
    reembedded = apply_updates(updates, df_dir, emb_dir) if updates else 0
    with open(state_path, "w") as f:
        json.dump({"last_run": started, "finished": datetime.now().isoformat(timespec="seconds")}, f)
    total = sum(len(pd.read_parquet(p, columns=["nct_id"])) for p in glob.glob(os.path.join(df_dir, "*.parquet")))
    print(f"Done: {chunk} new chunk(s), {len(updates):,} updated trials "
          f"({reembedded:,} re-embedded); {total:,} trials indexed.")


if __name__ == "__main__":
    main()
