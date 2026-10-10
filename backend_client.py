"""
HTTP client the Streamlit UI uses to talk to the backend (search_api.py).
The UI holds no model and no index; every search goes through here.

Environment:
    MSS_BACKEND_URL   default http://127.0.0.1:8080
    MSS_INTERNAL_KEY  must match the backend's MSS_INTERNAL_KEY
"""
import os

import pandas as pd
import requests

BACKEND_URL = os.environ.get("MSS_BACKEND_URL", "http://127.0.0.1:8080").rstrip("/")
INTERNAL_KEY = os.environ.get("MSS_INTERNAL_KEY", "")
SEARCH_TIMEOUT_S = 90


class BackendError(RuntimeError):
    """A message that can be shown to the user as is."""


class BackendBusy(BackendError):
    pass


def _error_detail(response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return detail if isinstance(detail, str) else f"HTTP {response.status_code}"


def search(query: str, top_k: int = 10, start_date=None, end_date=None,
           high_quality_only: bool = True, sources=None):
    """Returns (results DataFrame, response metadata dict)."""
    body = {"query": query, "top_k": top_k, "high_quality_only": high_quality_only}
    if start_date:
        body["start_date"] = str(start_date)
    if end_date:
        body["end_date"] = str(end_date)
    if sources:
        body["sources"] = list(sources)
    try:
        r = requests.post(f"{BACKEND_URL}/v1/search", json=body, timeout=SEARCH_TIMEOUT_S,
                          headers={"X-API-Key": INTERNAL_KEY})
    except requests.exceptions.ConnectionError:
        raise BackendError("The search service is not reachable. It may be restarting; "
                           "try again in a minute.") from None
    except requests.exceptions.Timeout:
        raise BackendError("The search took too long. Try a shorter query or fewer results.") from None
    if r.status_code == 503 and r.headers.get("Retry-After"):
        raise BackendBusy(_error_detail(r))
    if r.status_code != 200:
        raise BackendError(f"Search failed: {_error_detail(r)}")
    payload = r.json()
    df = pd.DataFrame(payload.pop("results", []))
    return df, payload


def _post(path: str, body: dict):
    try:
        r = requests.post(f"{BACKEND_URL}{path}", json=body, timeout=SEARCH_TIMEOUT_S,
                          headers={"X-API-Key": INTERNAL_KEY})
    except requests.exceptions.ConnectionError:
        raise BackendError("The search service is not reachable. It may be restarting; "
                           "try again in a minute.") from None
    except requests.exceptions.Timeout:
        raise BackendError("The search took too long. Try again or ask for fewer results.") from None
    if r.status_code == 503 and r.headers.get("Retry-After"):
        raise BackendBusy(_error_detail(r))
    if r.status_code != 200:
        raise BackendError(_error_detail(r))
    payload = r.json()
    return pd.DataFrame(payload.pop("results", [])), payload


def similar(refs, top_k: int = 10, start_date=None, end_date=None,
            high_quality_only: bool = True, sources=None):
    """Papers similar to example papers (their 'ref's). Returns (DataFrame, metadata incl. seeds)."""
    body = {"refs": list(refs), "top_k": top_k, "high_quality_only": high_quality_only}
    if start_date:
        body["start_date"] = str(start_date)
    if end_date:
        body["end_date"] = str(end_date)
    if sources:
        body["sources"] = list(sources)
    return _post("/v1/similar", body)


def stats() -> dict:
    """Corpus sizes and update dates; {} if the backend is unreachable."""
    try:
        r = requests.get(f"{BACKEND_URL}/v1/stats", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except requests.exceptions.RequestException:
        return {}


def map_info() -> dict:
    """Paper map description (tile URL template, coordinate system, labels); {} if unavailable."""
    try:
        r = requests.get(f"{BACKEND_URL}/v1/map", timeout=10)
        return r.json() if r.status_code == 200 else {}
    except requests.exceptions.RequestException:
        return {}
