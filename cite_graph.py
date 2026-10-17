"""
Citation links of one paper, from OpenAlex: the papers that cite it and the papers
it cites. The backend matches them back to the index (identifiers.id_key), so indexed
papers show with their usual labels and links; the rest keep OpenAlex's metadata.

OpenAlex allows ~1,000 list requests a day without a key (MSS_OPENALEX_KEY raises
that); one lookup costs 2-5 requests, and results are cached for a day.
"""
import logging
import os
import threading
import time

import requests

from identifiers import id_key

LOGGER = logging.getLogger(__name__)

API = "https://api.openalex.org"
SELECT = ("id,doi,ids,display_name,publication_date,primary_location,authorships,"
          "cited_by_count,type")
MAX_RESULTS = 200
_TTL_S = 24 * 3600
_CACHE: dict = {}
_CACHE_MAX = 2_000
_LOCK = threading.Lock()


class CitationsUnavailable(RuntimeError):
    """OpenAlex could not be reached or its daily allowance is used up (message is user-safe)."""


def _get(path: str, params: dict = None) -> dict:
    params = dict(params or {})
    if os.environ.get("MSS_OPENALEX_KEY"):
        params["api_key"] = os.environ["MSS_OPENALEX_KEY"]
    if os.environ.get("MSS_CONTACT_EMAIL"):
        params["mailto"] = os.environ["MSS_CONTACT_EMAIL"]
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=15)
    except requests.RequestException as exc:
        LOGGER.warning(f"OpenAlex request failed: {exc}")
        raise CitationsUnavailable("The citation database could not be reached. Please try again later.") from None
    if r.status_code == 404:
        return {}
    if r.status_code == 429:
        raise CitationsUnavailable("Citation lookups are used up for today. Please try again tomorrow.")
    if not r.ok:
        LOGGER.warning(f"OpenAlex {path}: HTTP {r.status_code}")
        raise CitationsUnavailable("The citation database could not be reached. Please try again later.")
    return r.json()


def _cached(key, fn):
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < _TTL_S:
            return hit[1]
    value = fn()
    with _LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = (time.time(), value)
    return value


def find_work(doi: str = None, pmid: str = None) -> dict:
    """The OpenAlex work for a DOI or PMID ({} if OpenAlex does not know it)."""
    for kind, value in (("doi", doi), ("pmid", pmid)):
        key = id_key(kind, value)
        if not key:
            continue
        work = _cached(("work", key), lambda: _get(
            f"/works/{key}", {"select": "id,display_name,cited_by_count,referenced_works,publication_date"}))
        if work:
            return work
    return {}


def _short_id(openalex_url: str) -> str:
    return str(openalex_url or "").rsplit("/", 1)[-1]


def cited_by(work: dict, limit: int = 100) -> list:
    """Works citing `work`, most cited first."""
    wid = _short_id(work.get("id"))
    limit = max(1, min(int(limit), MAX_RESULTS))
    data = _cached(("cites", wid, limit), lambda: _get(
        "/works", {"filter": f"cites:{wid}", "sort": "cited_by_count:desc", "per_page": limit, "select": SELECT}))
    return data.get("results", []) if data else []


def references(work: dict, limit: int = 100) -> list:
    """Works that `work` cites, newest first."""
    wids = [_short_id(w) for w in (work.get("referenced_works") or [])][:MAX_RESULTS]
    out = []
    for i in range(0, len(wids), 50):
        batch = wids[i:i + 50]
        data = _cached(("works", tuple(batch)), lambda: _get(
            "/works", {"filter": "openalex_id:" + "|".join(batch), "per_page": 50, "select": SELECT}))
        out += data.get("results", []) if data else []
    out.sort(key=lambda w: w.get("publication_date") or "", reverse=True)
    return out[:max(1, int(limit))]


def work_keys(work: dict) -> list:
    """Paper-id keys of an OpenAlex work (DOI, PMID)."""
    ids = work.get("ids") or {}
    keys = [id_key("doi", ids.get("doi") or work.get("doi")),
            id_key("pmid", str(ids.get("pmid") or "").rstrip("/").rsplit("/", 1)[-1])]
    return [k for k in keys if k]


def work_row(work: dict) -> dict:
    """A result row for a work that is not in the index."""
    names = [((a.get("author") or {}).get("display_name") or "") for a in (work.get("authorships") or [])]
    names = [n for n in names if n]
    authors = ", ".join(names[:3]) + (", et al." if len(names) > 3 else "")
    source = ((work.get("primary_location") or {}).get("source") or {})
    ids = work.get("ids") or {}
    pmid = str(ids.get("pmid") or "").rstrip("/").rsplit("/", 1)[-1]
    return {
        "title": work.get("display_name") or "Untitled",
        "authors": authors,
        "journal": source.get("display_name") or "",
        "doi": str(work.get("doi") or ids.get("doi") or "").replace("https://doi.org/", ""),
        "date": work.get("publication_date") or "",
        "pmid": pmid if pmid.isdigit() else None,
        "source": "External",
        "abstract": "",
        "score": 0.0,
        "pub_type": "Preprint" if work.get("type") == "preprint" else "",
        "openalex_url": work.get("id"),
        "cited_by_count": work.get("cited_by_count"),
    }
