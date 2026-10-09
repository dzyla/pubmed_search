"""Tests for search_api, model_api and the session counter (no network, no model download)."""
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import EMB_BYTES


# ---------------------------------------------------------------------------
# search_api
# ---------------------------------------------------------------------------

@pytest.fixture
def api(corpus, monkeypatch):
    import search_api

    config, _ = corpus
    monkeypatch.setattr(search_api, "CONFIGS", [config, {}, {}, {}])
    monkeypatch.setattr(search_api, "VALID_KEYS", {"test-key"})
    monkeypatch.setattr(
        search_api, "get_query_embedding_packed",
        lambda query, server_url: np.zeros((1, EMB_BYTES), dtype=np.uint8),
    )
    # No `with` block: skip the lifespan (startup update check + warm-up).
    return TestClient(search_api.app), search_api


def _search(client, **body):
    return client.post("/search", json={"query": "binary embeddings", **body},
                       headers={"X-API-Key": "test-key"})


def test_search_returns_results(api):
    client, _ = api
    r = _search(client, top_k=5, start_date="2015-01-01")
    assert r.status_code == 200
    papers = r.json()["results"]
    assert len(papers) == 5
    assert all(p["year"] == 2015 for p in papers)


@pytest.mark.parametrize("body", [
    {"start_date": "2020-13-45"},
    {"end_date": "not-a-date"},
    {"start_date": "2024-01-01", "end_date": "2020-01-01"},
])
def test_bad_dates_are_rejected_with_422(api, body):
    client, _ = api
    assert _search(client, **body).status_code == 422


def test_internal_errors_are_not_leaked(api, monkeypatch):
    client, search_api = api

    def boom(*args, **kwargs):
        raise RuntimeError("/root/secret/path exploded")

    monkeypatch.setattr(search_api, "combined_search_orchestrator", boom)
    r = _search(client)
    assert r.status_code == 500
    assert "/root" not in r.text


def test_missing_key_is_rejected(api):
    client, _ = api
    assert client.post("/search", json={"query": "abc"}).status_code == 401


# ---------------------------------------------------------------------------
# model_api
# ---------------------------------------------------------------------------

def test_model_health_is_503_until_model_loaded():
    import model_api

    model_api.model_context.clear()
    client = TestClient(model_api.app)  # no lifespan → model never loads
    assert client.get("/health").status_code == 503


# ---------------------------------------------------------------------------
# Active-user counter
# ---------------------------------------------------------------------------

def test_active_users_migrates_legacy_table(tmp_path, monkeypatch):
    import utils

    db = tmp_path / "sessions.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE session_history "
                     "(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, last_seen INTEGER)")
        conn.executemany("INSERT INTO session_history (session_id, last_seen) VALUES (?, ?)",
                         [("old", 0)] * 3)

    state = {}

    class FakeSessionState(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    monkeypatch.setattr(utils.st, "session_state", FakeSessionState(state))

    assert utils.get_current_active_users(str(db)) == 1
    # A second session is counted, and the same session is not double-counted.
    monkeypatch.setattr(utils.st, "session_state", FakeSessionState())
    assert utils.get_current_active_users(str(db)) == 2
    assert utils.get_current_active_users(str(db)) == 2
