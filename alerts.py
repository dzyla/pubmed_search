"""
Saved-search alerts: new papers that match a saved query, delivered as a private
RSS feed (the alert's random token is the only key to it; no account, no email).

How matching works
- A saved search keeps its query vector and a bar: the int8 score of (about) the N-th
  best document for that query across its databases at creation (N from the sensitivity,
  SENSITIVITY; search_logic.alert_calibration). A new paper that clears the bar would
  have ranked in that top N. Bars set by an older calibration (column `calibration` <
  CALIBRATION) are recomputed at backend start, and feed items below the new bar removed.
- After each data update the backend calls run(): per source it compares the rows of
  every embedding file with the previous run's snapshot, keeps new rows that are not
  superseded and were published within RECENT_DAYS (weekly full re-downloads such as
  ClinicalTrials.gov bring back old records as "new" rows), scores them against all
  alerts at once with their int8 vectors, and stores the hits with their metadata
  (row ids may shift when the index is rebuilt; stored items do not).
- New alerts are seeded with matching papers from the last SEED_DAYS days.

Abuse limits: creating an alert costs a calibration (seconds of index work) and a row,
so each visitor may keep MAX_ACTIVE_PER_CLIENT alerts and create MAX_NEW_PER_CLIENT_DAY a
day, and the whole site MAX_NEW_PER_DAY a day (check_quota). A visitor is a salted hash of
their address; the address itself is not stored.

Storage: SQLite at MSS_ALERTS_DB (default alerts.db next to this file; *.db files are
never deployed or committed). Alerts nobody read (feed or UI) for EXPIRE_DAYS are deleted.
"""
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from email.utils import format_datetime
from urllib.parse import quote_plus
from xml.sax.saxutils import escape

import numpy as np

import int8_codes
import paper_links

LOGGER = logging.getLogger(__name__)

DB_PATH = os.environ.get("MSS_ALERTS_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "alerts.db"))
# Sensitivity -> rank of the bar among all documents of the alert's databases. With
# ~15k new papers a day in ~51M, an average topic gets roughly rank × 0.002 a week.
SENSITIVITY = {"focused": 500, "balanced": 2000, "broad": 10000}
RECENT_DAYS = 120
SEED_DAYS = 30
MAX_ITEMS_PER_RUN = 30
MAX_ITEMS_KEPT = 300
EXPIRE_DAYS = 180
MAX_ALERTS = 20000
MAX_ACTIVE_PER_CLIENT = 25
MAX_NEW_PER_CLIENT_DAY = 10
MAX_NEW_PER_DAY = 500
# 1: bar = lowest int8 score among the binary top N (far too low: true rank 100k-1.4M for
#    N = 2,000); 2: rank-N int8 score in a re-scored pool of 4N (2026-10-16).
CALIBRATION = 2

_LOCK = threading.Lock()


class AlertLimit(RuntimeError):
    """A creation limit was reached (message is user-safe)."""


def _db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS alerts (
            token TEXT PRIMARY KEY, query TEXT NOT NULL, sources TEXT NOT NULL, vec BLOB NOT NULL,
            threshold REAL NOT NULL, sensitivity TEXT NOT NULL, created INTEGER NOT NULL,
            last_seen INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS items (
            token TEXT NOT NULL, key TEXT NOT NULL, found INTEGER NOT NULL, score REAL,
            source TEXT, title TEXT, authors TEXT, journal TEXT, date TEXT, doi TEXT, url TEXT,
            abstract TEXT, PRIMARY KEY (token, key));
        CREATE INDEX IF NOT EXISTS items_by_token ON items (token, found);
        CREATE TABLE IF NOT EXISTS state (name TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    if "calibration" not in {r[1] for r in conn.execute("PRAGMA table_info(alerts)")}:
        conn.execute("ALTER TABLE alerts ADD COLUMN calibration INTEGER NOT NULL DEFAULT 1")
        conn.commit()
    if "creator" not in {r[1] for r in conn.execute("PRAGMA table_info(alerts)")}:
        conn.execute("ALTER TABLE alerts ADD COLUMN creator TEXT NOT NULL DEFAULT ''")
        conn.execute("CREATE INDEX IF NOT EXISTS alerts_by_creator ON alerts (creator, created)")
        conn.commit()
    return conn


def _creator(conn, client: str) -> str:
    """Salted hash of a visitor's address ('' when unknown)."""
    if not client:
        return ""
    row = conn.execute("SELECT value FROM state WHERE name='creator_salt'").fetchone()
    if row is None:
        conn.execute("INSERT OR IGNORE INTO state VALUES ('creator_salt', ?)", (secrets.token_hex(16),))
        row = conn.execute("SELECT value FROM state WHERE name='creator_salt'").fetchone()
    return hashlib.sha256((row[0] + client).encode()).hexdigest()[:24]


def _check_quota(conn, creator: str):
    day_ago = int(time.time()) - 86400
    if conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] >= MAX_ALERTS:
        raise RuntimeError("The alert service is full; please try again later.")
    if conn.execute("SELECT COUNT(*) FROM alerts WHERE created >= ?", (day_ago,)).fetchone()[0] >= MAX_NEW_PER_DAY:
        raise AlertLimit("Too many alerts were created today; please try again tomorrow.")
    if not creator:
        return
    if conn.execute("SELECT COUNT(*) FROM alerts WHERE creator=?", (creator,)).fetchone()[0] >= MAX_ACTIVE_PER_CLIENT:
        raise AlertLimit(f"You have {MAX_ACTIVE_PER_CLIENT} alerts already; delete one under Alerts first.")
    if conn.execute("SELECT COUNT(*) FROM alerts WHERE creator=? AND created >= ?",
                    (creator, day_ago)).fetchone()[0] >= MAX_NEW_PER_CLIENT_DAY:
        raise AlertLimit(f"You can create {MAX_NEW_PER_CLIENT_DAY} alerts a day; please try again tomorrow.")


def check_quota(client: str = ""):
    """Raises AlertLimit (or RuntimeError when the service is full) if `client` may not create an alert now."""
    with _db() as conn:
        _check_quota(conn, _creator(conn, client))


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def create(query: str, sources: list, vec: np.ndarray, threshold: float, sensitivity: str,
           client: str = "") -> str:
    """Saves an alert for visitor `client` (their address); raises AlertLimit / RuntimeError over the limits."""
    token = secrets.token_urlsafe(18)
    now = int(time.time())
    with _LOCK, _db() as conn:
        creator = _creator(conn, client)
        _check_quota(conn, creator)
        conn.execute("INSERT INTO alerts (token, query, sources, vec, threshold, sensitivity, created, last_seen, "
                     "calibration, creator) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (token, query, json.dumps(list(sources)), np.asarray(vec, np.float32).tobytes(),
                      float(threshold), sensitivity, now, now, CALIBRATION, creator))
    return token


def stale_calibrations() -> list:
    """Alerts whose bar was set by an older calibration: [{token, sources, vec, rank}]."""
    with _db() as conn:
        rows = conn.execute("SELECT token, sources, vec, sensitivity FROM alerts WHERE calibration < ?",
                            (CALIBRATION,)).fetchall()
    return [{"token": r[0], "sources": json.loads(r[1]), "vec": np.frombuffer(r[2], np.float32),
             "rank": SENSITIVITY.get(r[3], SENSITIVITY["balanced"])} for r in rows]


def set_threshold(token: str, threshold: float) -> int:
    """Stores a recomputed bar and removes the alert's items below it. Returns how many were removed."""
    with _LOCK, _db() as conn:
        conn.execute("UPDATE alerts SET threshold=?, calibration=? WHERE token=?",
                     (float(threshold), CALIBRATION, token))
        return conn.execute("DELETE FROM items WHERE token=? AND score < ?", (token, float(threshold))).rowcount


def get(token: str, touch: bool = True):
    with _db() as conn:
        row = conn.execute("SELECT query, sources, threshold, sensitivity, created FROM alerts WHERE token=?",
                           (token,)).fetchone()
        if row is None:
            return None
        if touch:
            conn.execute("UPDATE alerts SET last_seen=? WHERE token=?", (int(time.time()), token))
    return {"query": row[0], "sources": json.loads(row[1]), "threshold": row[2], "sensitivity": row[3],
            "created": row[4]}


def delete(token: str) -> bool:
    with _db() as conn:
        n = conn.execute("DELETE FROM alerts WHERE token=?", (token,)).rowcount
        conn.execute("DELETE FROM items WHERE token=?", (token,))
    return n > 0


def items(token: str, limit: int = 50) -> list:
    cols = ("key", "found", "score", "source", "title", "authors", "journal", "date", "doi", "url", "abstract")
    with _db() as conn:
        rows = conn.execute(f"SELECT {', '.join(cols)} FROM items WHERE token=? ORDER BY found DESC, score DESC "
                            "LIMIT ?", (token, limit)).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def _item_key(row: dict) -> str:
    doi = paper_links.linkable_doi(row.get("doi"))
    if doi:
        return "doi:" + doi.lower()
    for k in ("nct_id", "grant_id", "registry_id", "pmid"):
        if row.get(k):
            return f"{k}:{row[k]}"
    return "t:" + re.sub(r"[^a-z0-9]+", " ", str(row.get("title") or "").lower()).strip()[:200]


def _clean(value, limit=None):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if s.lower() in ("", "nan", "none", "n/a"):
        return None
    return s[:limit] if limit else s


def add_items(token: str, rows: list, found: int = None):
    """rows: result dicts (title, authors, journal, date, doi, source, score, …). Duplicates are ignored."""
    found = int(time.time()) if found is None else found
    with _db() as conn:
        for row in rows:
            conn.execute(
                "INSERT OR IGNORE INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (token, _item_key(row), found, float(row.get("score") or 0), row.get("source"),
                 _clean(row.get("title"), 500), _clean(row.get("authors"), 500), _clean(row.get("journal"), 300),
                 _clean(row.get("date"), 10), paper_links.linkable_doi(row.get("doi")) or None,
                 _clean(row.get("url")) or paper_links.primary_link(row), _clean(row.get("abstract"), 1200)))
        # keep the newest MAX_ITEMS_KEPT per alert
        conn.execute("DELETE FROM items WHERE token=? AND rowid NOT IN (SELECT rowid FROM items WHERE token=? "
                     "ORDER BY found DESC, score DESC LIMIT ?)", (token, token, MAX_ITEMS_KEPT))


# ---------------------------------------------------------------------------
# RSS
# ---------------------------------------------------------------------------

def rss(token: str, feed_url: str, site_url: str) -> str:
    alert = get(token)
    if alert is None:
        return None
    entries = []
    for it in items(token, 100):
        bits = [b for b in (it["authors"], it["journal"], it["date"], it["source"]) if b]
        desc = " · ".join(bits) + ("\n\n" + it["abstract"] if it["abstract"] else "")
        link = it["url"] or (f"https://doi.org/{it['doi']}" if it["doi"] else site_url)
        entries.append(
            "<item>"
            f"<title>{escape(it['title'] or 'Untitled')}</title>"
            f"<link>{escape(link)}</link>"
            f"<guid isPermaLink=\"false\">{escape(token[:8] + ':' + it['key'])}</guid>"
            f"<pubDate>{format_datetime(datetime.fromtimestamp(it['found'], timezone.utc))}</pubDate>"
            f"<description>{escape(desc)}</description>"
            "</item>")
    # query first: chat integrations (Zulip, Slack) cut the feed title to ~60 characters for the topic
    title = f"{alert['query'][:80]} · Manuscript Search"
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0" '
            'xmlns:atom="http://www.w3.org/2005/Atom"><channel>'
            f"<title>{escape(title)}</title><link>{escape(site_url)}/?q={escape(quote_plus(alert['query'][:500]))}</link>"
            f'<atom:link href="{escape(feed_url)}" rel="self" type="application/rss+xml"/>'
            f"<description>{escape('New papers matching: ' + alert['query'][:500])}</description>"
            f"<ttl>720</ttl>{''.join(entries)}</channel></rss>")


# ---------------------------------------------------------------------------
# Matching new papers (called by the backend after data updates)
# ---------------------------------------------------------------------------

def _stem_rows(searcher) -> dict:
    rows: dict = {}
    for iv in searcher.intervals:
        n = iv["source_local_start"] + iv["global_end"] - iv["global_start"] + 1
        rows[iv["source_stem"]] = max(rows.get(iv["source_stem"], 0), n)
    return rows


def _new_gids(searcher, old: dict) -> np.ndarray:
    """Global row ids of rows not in the previous snapshot (new files, or rows appended to known ones)."""
    out = []
    for iv in searcher.intervals:
        seen = old.get(iv["source_stem"], 0)
        first_local = iv["source_local_start"]
        last_local = first_local + iv["global_end"] - iv["global_start"]
        start = max(seen, first_local)
        if start <= last_local:
            g0 = iv["global_start"] + (start - first_local)
            out.append(np.arange(g0, iv["global_end"] + 1, dtype=np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _load_alerts(conn):
    rows = conn.execute("SELECT token, sources, vec, threshold FROM alerts").fetchall()
    return ([r[0] for r in rows], [set(json.loads(r[1])) for r in rows],
            np.stack([np.frombuffer(r[2], np.float32) for r in rows]) if rows else np.zeros((0, int8_codes.DIM)),
            np.array([r[3] for r in rows], dtype=np.float32))


def run(searchers: dict, today_days: int = None) -> dict:
    """
    Matches papers added since the last run against all alerts. searchers: {source name:
    ChunkedSearcher}. Returns {source: new papers checked}. The first run only records
    the snapshot.
    """
    if today_days is None:
        today_days = (datetime.now(timezone.utc).date() - datetime(1970, 1, 1).date()).days
    with _LOCK:
        conn = _db()
        try:
            state_row = conn.execute("SELECT value FROM state WHERE name='snapshot'").fetchone()
            snapshot = json.loads(state_row[0]) if state_row else {}
            tokens, alert_sources, vecs, thresholds = _load_alerts(conn)
            lo, step = int8_codes.load_calibration()
            W = vecs * step                          # (n_alerts, 384)
            b = vecs @ (lo + 128.0 * step)            # (n_alerts,)
            checked, hits_by_alert = {}, {}
            for name, searcher in searchers.items():
                if searcher is None or searcher.data_problem:
                    continue
                old = snapshot.get(name)
                snapshot[name] = _stem_rows(searcher)
                if old is None or not tokens:
                    continue
                gids = _new_gids(searcher, old)
                if not len(gids):
                    continue
                gids = gids[~searcher.is_superseded(gids)]
                days = searcher.dates_for(gids)
                gids = gids[(days >= today_days - RECENT_DAYS) & (days <= today_days + 365)]
                checked[name] = int(len(gids))
                which = [j for j, s in enumerate(alert_sources) if name in s]
                if not len(gids) or not which:
                    continue
                for start in range(0, len(gids), 20000):
                    g = gids[start:start + 20000]
                    codes, ok = searcher.checked_int8_rows(g)
                    g, codes = g[ok], codes[ok]
                    if not len(g):
                        continue
                    scores = (1.0 + (codes.astype(np.float32) @ W[which].T + b[which])) / 2.0   # (n, k)
                    for col, j in enumerate(which):
                        above = np.flatnonzero(scores[:, col] >= thresholds[j])
                        for i in above:
                            hits_by_alert.setdefault(j, []).append((float(scores[i, col]), name, int(g[i])))
            conn.execute("INSERT OR REPLACE INTO state VALUES ('snapshot', ?)", (json.dumps(snapshot),))
            conn.execute("DELETE FROM alerts WHERE last_seen < ?", (int(time.time()) - EXPIRE_DAYS * 86400,))
            conn.execute("DELETE FROM items WHERE token NOT IN (SELECT token FROM alerts)")
            conn.commit()
        finally:
            conn.close()

        # metadata for the hits (best first, capped per alert), fetched per source in one go
        wanted = {}
        for j, hits in hits_by_alert.items():
            for score, name, gid in sorted(hits, reverse=True)[:MAX_ITEMS_PER_RUN]:
                wanted.setdefault((name, gid), []).append((j, score))
        rows_by_id = {}
        for name in {n for n, _ in wanted}:
            cands = [{"corpus_id": g, "score": 0.0} for n, g in wanted if n == name]
            try:
                df = searchers[name].fetch_rows(cands)
            except Exception as exc:
                LOGGER.error(f"Alert rows for {name} could not be read: {exc}")
                continue
            for row in df.to_dict("records"):
                row["source"] = name
                rows_by_id[(name, int(row["corpus_id"]))] = row
        per_alert = {}
        for key, lst in wanted.items():
            row = rows_by_id.get(key)
            if row is None:
                continue
            for j, score in lst:
                per_alert.setdefault(j, []).append(dict(row, score=score))
        for j, rows in per_alert.items():
            add_items(tokens[j], rows)
        if checked:
            LOGGER.info(f"Alerts: {sum(checked.values()):,} new papers checked against {len(tokens):,} alert(s); "
                        f"{sum(len(r) for r in per_alert.values())} new item(s)")
        return checked
