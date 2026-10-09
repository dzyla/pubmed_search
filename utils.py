import time
import uuid
import sqlite3
import logging
import re
import requests
import pandas as pd
import streamlit as st
import concurrent.futures
from contextlib import contextmanager
import doi
from crossref.restful import Works, Etiquette
import json
import glob
LOGGER = logging.getLogger(__name__)

@contextmanager
def log_time(task_name: str, status_placeholder=None):
    start = time.perf_counter()
    msg = f"Starting {task_name}..."
    LOGGER.info(msg)
    if status_placeholder:
        status_placeholder.info(msg)
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        completion_message = f"Completed {task_name} in {duration:.2f} seconds."
        LOGGER.info(completion_message)
        if status_placeholder:
            status_placeholder.info(completion_message)


def get_current_active_users(db_path: str = "sessions_history.db", timeout: int = 300) -> int:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    session_id = st.session_state.session_id
    current_time = int(time.time())
    expiration_time = current_time - timeout

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_history (
                    session_id TEXT PRIMARY KEY,
                    last_seen  INTEGER NOT NULL
                )
            """)
            # Upsert: one row per session, never one row per page rerun
            conn.execute("""
                INSERT INTO session_history (session_id, last_seen)
                VALUES (?, ?)
                ON CONFLICT(session_id) DO UPDATE SET last_seen = excluded.last_seen
            """, (session_id, current_time))
            # Prune expired sessions to keep the table from growing unbounded
            conn.execute("DELETE FROM session_history WHERE last_seen < ?", (expiration_time,))
            row = conn.execute(
                "SELECT COUNT(*) FROM session_history WHERE last_seen >= ?", (expiration_time,)
            ).fetchone()
        return row[0] if row else 1
    except Exception:
        return 1


def get_clean_doi(doi_str):
    if not isinstance(doi_str, str):
        return ""
    if 'arxiv.org' in doi_str:
        return doi_str
    try:
        doi_clean = doi.get_clean_doi(doi_str)
        return doi_clean
    except Exception:
        return doi_str


def report_dates_from_metadata(metadata_dict: dict) -> str:
    folder = metadata_dict.get("embeddings_directory", "")
    json_files = glob.glob(f"{folder}/*.json")
    if not json_files:
        logging.warning(f"No JSON files found in {folder}")
        return "N/A"
    with open(json_files[0], "r") as f:
        return json.load(f).get("last_fetch_date", "N/A")


my_etiquette = Etiquette('Manuscript Search', '1.0', 'https://www.zylalab.org', 'dawid.zyla@cuanschutz.edu')
works = Works(etiquette=my_etiquette)


def get_citation_count(doi_str):
    try:
        if not doi_str or "arxiv" in str(doi_str):
            return 0
        paper_data = works.doi(doi_str)
        if paper_data:
            return paper_data.get("is-referenced-by-count", 0)
        return 0
    except Exception:
        return 0


def get_full_text_link(row):
    source = str(row.get("source", "None")).lower()
    if source == "pubmed":
        doi_val = row.get("doi")
        if doi_val and "10." in str(doi_val):
            return f"https://doi.org/{doi_val}"
        return None
    else:
        doi_val = row.get("doi")
        if doi_val:
            if "arxiv.org" in str(doi_val):
                return doi_val
            else:
                return f"https://doi.org/{doi_val}"
        return None


def precalculate_full_text_links_parallel(df):
    """
    Computes full-text links for each result row using vectorised string ops.
    No thread pool — get_full_text_link is pure string logic with zero I/O.
    """
    if df.empty:
        df["full_text_link"] = None
        return df

    doi_col = df["doi"].astype(str)
    source_col = df["source"].str.lower()

    is_arxiv = doi_col.str.contains("arxiv.org", na=False)
    has_doi = doi_col.str.contains("10.", na=False)
    is_pubmed = source_col == "pubmed"

    link = pd.Series([None] * len(df), index=df.index, dtype=object)

    # arXiv rows: the doi column already is a full URL
    link[is_arxiv] = doi_col[is_arxiv]
    # Non-arXiv rows with a real DOI
    non_arxiv_doi = ~is_arxiv & has_doi
    link[non_arxiv_doi] = "https://doi.org/" + doi_col[non_arxiv_doi]
    # PubMed rows without a DOI get None (already set above via default)
    link[is_pubmed & ~has_doi] = None

    df["full_text_link"] = link
    return df
