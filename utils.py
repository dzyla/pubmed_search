import time
import uuid
import sqlite3
import logging
import re
import requests
import streamlit as st
import concurrent.futures
from contextlib import contextmanager
import doi
from crossref.restful import Works, Etiquette
import json
import glob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
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
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    session_id = st.session_state.session_id
    current_time = int(time.time())
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                last_seen INTEGER
            )
        """)
        conn.commit()
        cursor.execute("""
            INSERT INTO session_history (session_id, last_seen)
            VALUES (?, ?)
        """, (session_id, current_time))
        conn.commit()
        
        expiration_time = current_time - timeout
        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT session_id, MAX(last_seen) AS last_seen
                FROM session_history
                GROUP BY session_id
                HAVING last_seen >= ?
            ) AS active_sessions
        """, (expiration_time,))
        active_count = cursor.fetchone()[0]
        conn.close()
        return active_count
    except Exception as e:
        return 1

def get_clean_doi(doi_str):
    if not isinstance(doi_str, str):
        return ""
    if 'arxiv.org' in doi_str:
        return doi_str
    try:
        doi_clean = doi.get_clean_doi(doi_str)
        return doi_clean
    except Exception as e:
        # LOGGER.error(f"Error cleaning DOI {doi_str}: {e}")
        return doi_str

def report_dates_from_metadata(metadata_dict: dict) -> str:
    import json

    # search for json in embeddings_directory
    folder = metadata_dict.get("embeddings_directory", "")
    json_files = glob.glob(f"{folder}/*.json")
    if not json_files:
        logging.warning(f"No JSON files found in {folder}")
        return "N/A" 
    else:
        last_fetch_date = json.load(open(json_files[0], "r"))['last_fetch_date']
        return last_fetch_date


my_etiquette = Etiquette('Manuscript Search', '1.0', 'https://www.zylalab.org', 'dawid.zyla@cuanschutz.edu')
works = Works(etiquette=my_etiquette)

def get_citation_count(doi_str):
    try:
        if not doi_str or "arxiv" in str(doi_str):
            return 0
            
        # The 'works' object is now "polite" and less likely to get 429 errors
        paper_data = works.doi(doi_str)
        
        if paper_data:
            return paper_data.get("is-referenced-by-count", 0)
        return 0
    except Exception as e:
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
    if df.empty:
        df["full_text_link"] = None
        return df

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(get_full_text_link, [row for _, row in df.iterrows()]))
    df["full_text_link"] = results
    return df