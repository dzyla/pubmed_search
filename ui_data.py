"""
UI-side helpers for the Streamlit app: live visitor count and citation counts.
(Backend-neutral helpers live in utils.py.)
"""
import logging
import os
import sqlite3
import time
import uuid

import streamlit as st
from crossref.restful import Etiquette, Works

LOGGER = logging.getLogger(__name__)

_SESSIONS_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions_history.db")


def get_current_active_users(db_path: str = _SESSIONS_DB, timeout: int = 300) -> int:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    session_id = st.session_state.session_id
    current_time = int(time.time())
    expiration_time = current_time - timeout

    try:
        with sqlite3.connect(db_path) as conn:
            # Older deployments created the table with an autoincrement id and no
            # unique session_id, which makes the upsert below fail. The table only
            # holds the last few minutes of activity, so recreate it.
            pk = conn.execute(
                "SELECT pk FROM pragma_table_info('session_history') WHERE name = 'session_id'"
            ).fetchone()
            if pk is not None and pk[0] == 0:
                LOGGER.info("Migrating session_history table to one row per session.")
                conn.execute("DROP TABLE session_history")
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
    except Exception as exc:
        LOGGER.warning(f"Active-user count unavailable: {exc}")
        return 1


my_etiquette = Etiquette('Manuscript Search', '1.0', 'https://www.zylalab.org', 'dawid.zyla@cuanschutz.edu')
works = Works(etiquette=my_etiquette)


# DOI -> (fetched_at, count). Successful lookups only, so a Crossref outage is
# retried on the next search. Shared across sessions; plain dict ops are atomic
# under the GIL, which is enough for the worker threads that call this.
_CITATION_CACHE: dict = {}
_CITATION_TTL_S = 24 * 3600
_CITATION_CACHE_MAX = 50_000


def get_citation_count(doi_str):
    if not doi_str or "arxiv" in str(doi_str):
        return 0
    cached = _CITATION_CACHE.get(doi_str)
    if cached and time.time() - cached[0] < _CITATION_TTL_S:
        return cached[1]
    try:
        paper_data = works.doi(doi_str)
    except Exception:
        return 0
    count = paper_data.get("is-referenced-by-count", 0) if paper_data else 0
    if len(_CITATION_CACHE) >= _CITATION_CACHE_MAX:
        _CITATION_CACHE.clear()
    _CITATION_CACHE[doi_str] = (time.time(), count)
    return count
