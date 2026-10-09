"""Tests for the backend (REST + MCP), the embedder and the session counter.
No network, no model download: the embedding model is stubbed."""
import asyncio
import hashlib
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import EMB_BYTES

PUBLIC, INTERNAL = {"X-API-Key": "pub-key"}, {"X-API-Key": "int-key"}


def fake_encode(text):
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    vec = np.random.default_rng(seed).standard_normal(EMB_BYTES * 8).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return np.packbits(vec > 0)[None, :], vec


@pytest.fixture(scope="module")
def backend(corpus_module, tmp_path_factory):
    import access
    import embedder
    import search_api

    config, _ = corpus_module
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(access, "DB_PATH", str(tmp_path_factory.mktemp("access") / "access.sqlite3"))
        access.create_key("free", label="test", key="pub-key")
        mp.setattr(search_api, "API_KEYS_FILE", "/nonexistent/api_keys.txt")
        mp.setattr(search_api, "CONFIGS", [config, {}, {}, {}])
        mp.setattr(search_api, "INTERNAL_KEY", "int-key")
        mp.setattr(embedder, "load_model", lambda: None)
        mp.setattr(embedder, "is_loaded", lambda: True)
        mp.setattr(embedder, "encode_query", fake_encode)
        with TestClient(search_api.app) as client:   # runs the real lifespan
            yield client, search_api


def _search(client, headers=PUBLIC, **body):
    return client.post("/search", json={"query": "binary embeddings", **body}, headers=headers)


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------

def test_health_and_stats(backend):
    client, _ = backend
    assert client.get("/health").json()["status"] == "ok"
    stats = client.get("/v1/stats").json()
    assert stats["sources"]["PubMed"]["papers"] == 600
    assert stats["total_papers"] == 600


def test_search_returns_rich_results(backend):
    client, _ = backend
    r = _search(client, top_k=5, start_date="2015-01-01")
    assert r.status_code == 200
    papers = r.json()["results"]
    assert len(papers) == 5
    assert all(p["year"] == 2015 and p["date"].startswith("2015") for p in papers)
    assert all(p["url"].startswith("https://doi.org/10.1234/") for p in papers)
    assert all(p["links"][0] == {"label": "DOI", "url": p["url"]} for p in papers)
    assert all(p["labels"] == [] and p["retracted"] is False for p in papers)


def test_v1_alias_and_sources_filter(backend):
    client, _ = backend
    r = client.post("/v1/search", json={"query": "abc def", "sources": ["PubMed"]}, headers=PUBLIC)
    assert r.status_code == 200 and r.json()["total_results"] == 10
    r = client.post("/v1/search", json={"query": "abc def", "sources": ["arXiv"]}, headers=PUBLIC)
    assert r.status_code == 200 and r.json()["total_results"] == 0
    assert client.post("/v1/search", json={"query": "abc", "sources": ["Scopus"]},
                       headers=PUBLIC).status_code == 422


def test_tier_limits(backend):
    client, _ = backend
    assert _search(client, top_k=20).status_code == 422
    r = _search(client, headers=INTERNAL, top_k=20)
    assert r.status_code == 200 and r.json()["total_results"] == 20
    long_query = "word " * 600
    assert client.post("/search", json={"query": long_query}, headers=PUBLIC).status_code == 422
    assert client.post("/search", json={"query": long_query}, headers=INTERNAL).status_code == 200


@pytest.mark.parametrize("body", [
    {"start_date": "2020-13-45"},
    {"end_date": "not-a-date"},
    {"start_date": "2024-01-01", "end_date": "2020-01-01"},
])
def test_bad_dates_are_rejected_with_422(backend, body):
    client, _ = backend
    assert _search(client, **body).status_code == 422


def _behind_local_proxy(monkeypatch):
    """The test client is not 127.0.0.1; pretend requests come through the local nginx."""
    import search_api
    monkeypatch.setattr(search_api, "_client_ip", lambda host, headers: headers.get("x-real-ip", host))


def test_keys_tiers_and_quota(backend, monkeypatch):
    import access

    client, _ = backend
    assert client.post("/search", json={"query": "abc"}, headers={"X-API-Key": "nope"}).status_code == 401
    r = _search(client, query="free tier query")
    assert r.status_code == 200 and int(r.headers["x-ratelimit-limit"]) == access.LIMITS["free"]
    assert "x-ratelimit-limit" not in _search(client, headers=INTERNAL, query="ui query").headers

    _behind_local_proxy(monkeypatch)
    monkeypatch.setitem(access.LIMITS, "anonymous", 2)
    anon = {"X-Real-IP": "203.0.113.7"}
    codes = [client.post("/search", json={"query": f"anonymous {i}"}, headers=anon).status_code for i in range(3)]
    assert codes == [200, 200, 429]
    r = client.post("/search", json={"query": "anonymous again"}, headers=anon)
    assert "/signup" in r.json()["detail"] and int(r.headers["retry-after"]) > 0


def test_internal_errors_are_not_leaked(backend, monkeypatch):
    client, search_api = backend

    def boom(*args, **kwargs):
        raise RuntimeError("/root/secret/path exploded")

    monkeypatch.setattr(search_api, "combined_search_orchestrator", boom)
    r = _search(client, query="something new")
    assert r.status_code == 500
    assert "/root" not in r.text


def test_busy_server_returns_503_with_retry_after(backend, monkeypatch):
    client, search_api = backend
    monkeypatch.setattr(search_api, "QUEUE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(search_api, "_SEARCH_SLOTS", asyncio.Semaphore(0))
    r = _search(client, query="queued query")
    assert r.status_code == 503
    assert r.headers["retry-after"] == "10"


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------

MCP_HEADERS = {**PUBLIC, "Accept": "application/json, text/event-stream",
               "Content-Type": "application/json"}


def _rpc(client, method, params=None, rid=1, headers=MCP_HEADERS):
    return client.post("/mcp/", headers=headers,
                       json={"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})


def test_mcp_anonymous_and_invalid_key(backend):
    client, _ = backend
    no_key = {k: v for k, v in MCP_HEADERS.items() if k != "X-API-Key"}
    assert _rpc(client, "tools/list", headers=no_key).status_code == 200
    assert _rpc(client, "tools/list", headers={**no_key, "X-API-Key": "nope"}).status_code == 401


def test_mcp_search_calls_count_against_quota(backend, monkeypatch):
    import access

    client, _ = backend
    _behind_local_proxy(monkeypatch)
    monkeypatch.setitem(access.LIMITS, "anonymous", 1)
    headers = {**{k: v for k, v in MCP_HEADERS.items() if k != "X-API-Key"}, "X-Real-IP": "198.51.100.9"}
    call = {"name": "search_papers", "arguments": {"query": "quota test query", "top_k": 1}}
    assert _rpc(client, "tools/call", call, headers=headers).status_code == 200
    assert _rpc(client, "tools/call", call, headers=headers).status_code == 429
    assert _rpc(client, "tools/list", headers=headers).status_code == 200      # listing is free


def test_signup_disabled_until_configured(backend):
    client, _ = backend
    page = client.get("/signup")
    assert page.status_code == 200 and "coming soon" in page.text and "searches per day" in page.text
    r = client.post("/v1/keys/request", json={"email": "a@b.org", "turnstile_token": "x"})
    assert r.status_code == 503


def test_access_keys_cli_functions(tmp_path, monkeypatch):
    import access

    monkeypatch.setattr(access, "DB_PATH", str(tmp_path / "a.sqlite3"))
    legacy = tmp_path / "api_keys.txt"
    legacy.write_text("# comment\nlegacy-key-1\n\n")
    assert access.import_legacy_keys(str(legacy)) == 1 and access.import_legacy_keys(str(legacy)) == 0
    assert access.lookup("legacy-key-1")[0] == "partner"
    key = access.create_key("free", email="x@y.org")
    assert access.lookup(key)[0] == "free"
    assert access.signup_allowed("1.2.3.4", "x@y.org") is None
    access.record_signup("1.2.3.4", "x@y.org")          # revokes older free keys of this address
    assert access.lookup(key) is None
    assert access.signup_allowed("1.2.3.4", "x@y.org") is not None


def test_mcp_initialize_list_and_call(backend):
    client, _ = backend
    init = _rpc(client, "initialize", {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "pytest", "version": "1"}})
    assert init.status_code == 200, init.text
    assert init.json()["result"]["serverInfo"]["name"] == "Manuscript Search"

    tools = _rpc(client, "tools/list", rid=2).json()["result"]["tools"]
    assert {t["name"] for t in tools} == {"search_papers", "database_info", "find_similar"}

    call = _rpc(client, "tools/call", {"name": "search_papers", "arguments": {
        "query": "binary embeddings for literature search", "top_k": 3,
        "include_abstracts": False}}, rid=3).json()["result"]
    assert not call.get("isError"), call
    results = call["structuredContent"]["results"]
    assert len(results) == 3 and "abstract" not in results[0] and results[0]["links"]

    bad = _rpc(client, "tools/call", {"name": "search_papers", "arguments": {
        "query": "x y z", "top_k": 50}}, rid=4).json()["result"]
    assert bad["isError"] and "top_k" in bad["content"][0]["text"]

    info = _rpc(client, "tools/call", {"name": "database_info", "arguments": {}}, rid=5).json()["result"]
    assert info["structuredContent"]["sources"]["PubMed"]["papers"] == 600


def test_mcp_accepts_bearer_token(backend):
    client, _ = backend
    headers = {k: v for k, v in MCP_HEADERS.items() if k != "X-API-Key"}
    headers["Authorization"] = "Bearer pub-key"
    assert _rpc(client, "tools/list", headers=headers).status_code == 200


def test_mcp_path_without_trailing_slash(backend):
    client, _ = backend
    r = client.post("/mcp", headers=MCP_HEADERS, follow_redirects=False,
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    assert r.status_code == 200 and r.json()["result"]["tools"]


# ---------------------------------------------------------------------------
# Active-user counter
# ---------------------------------------------------------------------------

def test_active_users_migrates_legacy_table(tmp_path, monkeypatch):
    import ui_data

    db = tmp_path / "sessions.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE session_history "
                     "(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, last_seen INTEGER)")
        conn.executemany("INSERT INTO session_history (session_id, last_seen) VALUES (?, ?)",
                         [("old", 0)] * 3)

    class FakeSessionState(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    monkeypatch.setattr(ui_data.st, "session_state", FakeSessionState())
    assert ui_data.get_current_active_users(str(db)) == 1
    # A second session is counted, and the same session is not double-counted.
    monkeypatch.setattr(ui_data.st, "session_state", FakeSessionState())
    assert ui_data.get_current_active_users(str(db)) == 2
    assert ui_data.get_current_active_users(str(db)) == 2


def test_similar_endpoint_and_mcp_tool(backend):
    client, _ = backend
    ref = _search(client, top_k=3).json()["results"][0]["ref"]
    assert ref.startswith("PubMed:")
    r = client.post("/v1/similar", json={"refs": [ref], "top_k": 5}, headers=PUBLIC)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_results"] == 5 and body["seeds"][0]["ref"] == ref
    assert ref not in {p["ref"] for p in body["results"]}
    assert client.post("/v1/similar", json={"refs": ["PubMed:99999999"]}, headers=PUBLIC).status_code == 422
    assert client.post("/v1/similar", json={"refs": ["nonsense"]}, headers=PUBLIC).status_code == 422

    call = _rpc(client, "tools/call", {"name": "find_similar", "arguments": {
        "paper_refs": [ref], "top_k": 3, "include_abstracts": False}}, rid=9).json()["result"]
    assert not call.get("isError"), call
    assert len(call["structuredContent"]["results"]) == 3


def test_forwarded_ip_trusted_only_from_local_proxy():
    import search_api
    assert search_api._client_ip("127.0.0.1", {"x-real-ip": "203.0.113.5"}) == "203.0.113.5"
    assert search_api._client_ip("198.51.100.2", {"x-real-ip": "203.0.113.5"}) == "198.51.100.2"
