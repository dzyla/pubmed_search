"""
Manuscript Search backend — the one process that owns the embedding model and
the search index. Everything else talks to it:

    Streamlit UI (pbmss_app.py)  ──► POST /v1/search  (internal key)
    REST API clients             ──► POST /search, /v1/search  (X-API-Key)
    AI agents (MCP)              ──► /mcp  (X-API-Key or Authorization: Bearer)
    anyone                       ──► GET /health, GET /v1/stats

Searches are CPU-bound, so at most MSS_MAX_CONCURRENT_SEARCHES run at once;
others wait up to MSS_QUEUE_TIMEOUT_S and then get 503 "busy". This process is
also the only one that picks up new embedding files (hourly).

Run:  python search_api.py     (binds MSS_API_HOST:MSS_API_PORT, default 127.0.0.1:8080)

Environment (all optional):
    MSS_CONFIG_PATH              config_mss.yaml (default: next to this file)
    MSS_API_KEYS_FILE            public API keys, one per line (default: api_keys.txt here)
    MSS_INTERNAL_KEY             key the UI uses; allows up to 50 results and long queries
    MSS_API_HOST / MSS_API_PORT  bind address (default 127.0.0.1 / 8080)
    MSS_MAX_CONCURRENT_SEARCHES  default 2
    MSS_QUEUE_TIMEOUT_S          default 20
    MSS_UPDATE_INTERVAL_S        default 3600
    MSS_CORS_ORIGINS             comma-separated, default *
    MSS_ACCESS_DB                key/quota database (default data/access.sqlite3)
    MSS_TURNSTILE_SITEKEY / MSS_TURNSTILE_SECRET     Cloudflare Turnstile (signup bot check)
    MSS_SMTP_HOST / _PORT / _USER / _PASSWORD / _FROM  sender for signup emails
Keys: python access.py create|revoke|list|usage
"""

import asyncio
import json
import logging
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import List, Literal, Optional

from dataclasses import dataclass

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# Ensure local modules are importable when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import access
import embedder
import paper_links
from config_loader import DEFAULT_SOURCES, read_source_configs
from mcp_server import SOURCES, SearchFailed, build_mcp
import pandas as pd
from map_index import MapIndex
from search_logic import (
    _SEARCHER_CACHE, UnknownReference, combined_search_orchestrator, get_or_create_searcher, similar_search,
    trigger_database_updates, warm_up_indexes,
)
from utils import get_clean_doi

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("MSS_CONFIG_PATH", os.path.join(_HERE, "config_mss.yaml"))
API_KEYS_FILE = os.environ.get("MSS_API_KEYS_FILE", os.path.join(_HERE, "api_keys.txt"))
INTERNAL_KEY = os.environ.get("MSS_INTERNAL_KEY", "")
API_HOST = os.environ.get("MSS_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("MSS_API_PORT", "8080"))
MAX_CONCURRENT_SEARCHES = int(os.environ.get("MSS_MAX_CONCURRENT_SEARCHES", "2"))
QUEUE_TIMEOUT_S = float(os.environ.get("MSS_QUEUE_TIMEOUT_S", "20"))
UPDATE_INTERVAL_S = int(os.environ.get("MSS_UPDATE_INTERVAL_S", str(60 * 60)))

# Limits per caller tier. The UI (internal key) may ask for more.
PUBLIC_MAX_TOP_K, PUBLIC_MAX_QUERY_CHARS = 10, 2000
INTERNAL_MAX_TOP_K, INTERNAL_MAX_QUERY_CHARS = 50, 8192


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
@dataclass
class Caller:
    tier: str                 # internal | partner | free | anonymous
    subject: str              # quota bucket: key hash prefix or client IP
    limit: Optional[int]      # searches per UTC day (None = unlimited)


def _client_ip(host: Optional[str], headers: dict) -> str:
    """nginx passes the real client address; trust it only from a local proxy."""
    if host in ("127.0.0.1", "::1") and headers.get("x-real-ip"):
        return headers["x-real-ip"]
    return host or "unknown"


def resolve_caller(key: Optional[str], ip: str) -> Optional[Caller]:
    """Caller for a key (None if the key is invalid) or the anonymous tier."""
    if key:
        if INTERNAL_KEY and secrets.compare_digest(key, INTERNAL_KEY):
            return Caller("internal", "internal", None)
        found = access.lookup(key)
        return Caller(*found) if found else None
    return Caller("anonymous", f"ip:{ip}", access.LIMITS["anonymous"])


def _seconds_to_utc_midnight() -> int:
    now = datetime.now(timezone.utc)
    return int(86400 - (now.hour * 3600 + now.minute * 60 + now.second))


class QuotaExceeded(Exception):
    def __init__(self, caller: Caller):
        hint = ("Get a free API key for a higher limit at /signup."
                if caller.tier == "anonymous" else "The limit resets at 00:00 UTC.")
        super().__init__(f"Daily limit of {caller.limit} searches reached. {hint}")


def consume_quota(caller: Caller) -> Optional[int]:
    """Counts one search; returns remaining searches (None = unlimited)."""
    if caller.limit is None:
        return None
    allowed, remaining = access.consume(caller.subject, caller.limit)
    if not allowed:
        raise QuotaExceeded(caller)
    return remaining


# ---------------------------------------------------------------------------
# Search index configs
# ---------------------------------------------------------------------------
def _load_configs(yaml_path: str):
    try:
        configs = list(read_source_configs(yaml_path).values())
        LOGGER.info(f"Loaded search configs from {yaml_path}")
        return configs
    except FileNotFoundError:
        LOGGER.error(f"Config file not found: {yaml_path}")
    except Exception as exc:
        LOGGER.error(f"Failed to parse config {yaml_path}: {exc}")
    return None


CONFIGS = _load_configs(CONFIG_PATH)

# The paper map (tiles + reference papers), shipped with the aux indexes
_AUX_ROOT = next((c.get("aux_index_root") for c in CONFIGS or [] if c and c.get("aux_index_root")), None)
MAP = MapIndex(_AUX_ROOT) if _AUX_ROOT else None


def _configs_for(sources: Optional[List[str]]) -> list:
    """Source configs in search_logic order, with unselected sources blanked out."""
    wanted = {s.lower() for s in (sources or DEFAULT_SOURCES)}
    return [cfg if name.lower() in wanted else {} for name, cfg in zip(SOURCES, CONFIGS)]


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
SourceName = Literal["PubMed", "BioRxiv", "MedRxiv", "arXiv", "ClinicalTrials", "Preprints", "Grants"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=INTERNAL_MAX_QUERY_CHARS, description=(
        f"Free-text search query, up to {PUBLIC_MAX_QUERY_CHARS} characters "
        "(e.g. 'CRISPR base editing off-target effects')."))
    top_k: int = Field(default=10, ge=1, le=INTERNAL_MAX_TOP_K,
                       description=f"Number of results, 1-{PUBLIC_MAX_TOP_K}.")
    start_date: Optional[date] = Field(default=None, examples=["2020-01-01"], description=(
        "Only papers published on or after this date (YYYY-MM-DD)."))
    end_date: Optional[date] = Field(default=None, examples=["2024-12-31"], description=(
        "Only papers published on or before this date (YYYY-MM-DD)."))
    high_quality_only: bool = Field(default=True, description=(
        "Exclude papers with very short or missing abstracts."))
    sources: Optional[List[SourceName]] = Field(default=None, description=(
        "Restrict to these databases; all except Grants if omitted."))

    model_config = {"json_schema_extra": {"example": {
        "query": "CRISPR base editing off-target effects",
        "top_k": 10, "start_date": "2020-01-01", "end_date": "2024-12-31",
        "high_quality_only": True,
    }}}


class SimilarRequest(BaseModel):
    refs: List[str] = Field(..., min_length=1, max_length=20, description=(
        "1-20 example papers, as the 'ref' values of search results (e.g. 'PubMed:123456'). "
        "Several examples are combined into one 'more like these' query."))
    top_k: int = Field(default=10, ge=1, le=INTERNAL_MAX_TOP_K, description=f"Number of results, 1-{PUBLIC_MAX_TOP_K}.")
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    high_quality_only: bool = True
    sources: Optional[List[SourceName]] = None


class Link(BaseModel):
    label: str
    url: str


class Paper(BaseModel):
    title: str
    authors: str
    journal: str
    doi: str
    abstract: str
    date: Optional[str] = Field(default=None, description="Publication or posting date (YYYY-MM-DD).")
    year: Optional[int]
    score: float = Field(description="Relevance score between 0.0 (no match) and 1.0 (perfect match).")
    source: str = Field(description="Database: PubMed | BioRxiv | MedRxiv | arXiv | ClinicalTrials | Preprints | Grants.")
    url: Optional[str] = Field(default=None, description="Best link to the paper (DOI, arXiv or PubMed).")
    links: List[Link] = Field(default_factory=list, description="All links: DOI, PubMed, PDF, published version, preprint.")
    pmid: Optional[str] = Field(default=None, description="PubMed ID, when available.")
    labels: List[str] = Field(default_factory=list, description=(
        "Notable publication types and status, e.g. Retracted, Review, Meta-analysis, RCT, Preprint."))
    retracted: bool = Field(default=False, description="True if the paper has been retracted.")
    published_doi: Optional[str] = Field(default=None, description="For preprints: DOI of the journal version.")
    preprint_doi: Optional[str] = Field(default=None, description="For journal articles: DOI of the merged preprint.")
    matched_terms: List[str] = Field(default_factory=list, description=(
        "Identifiers from the query (gene symbols, variants, compound or trial ids) found in this "
        "document, normalized to lower case without hyphens."))
    pmcid: Optional[str] = Field(default=None, description="PubMed Central id when free full text is available.")
    registry_id: Optional[str] = Field(default=None, description=(
        "ClinicalTrials.gov NCT id for trials, NIH core project number for grants."))
    ref: Optional[str] = Field(default=None, description=(
        "Stable reference for /v1/similar and the find_similar MCP tool, e.g. 'PubMed:123456'."))
    map_xy: Optional[List[float]] = Field(default=None, description=(
        "Position on the paper map (map coordinates, see /v1/map), when the map is available."))


class SearchResponse(BaseModel):
    query: str
    seeds: List["Paper"] = Field(default_factory=list, description="For /v1/similar: the example papers.")
    query_bits: Optional[str] = Field(default=None, description=(
        "The query's 384-bit binary code as hex — what the index matches against."))
    query_map_xy: Optional[List[float]] = Field(default=None, description=(
        "Position of the query on the paper map (see /v1/map)."))
    total_results: int
    search_time_seconds: float
    results: List[Paper]


def _optional_str(value) -> Optional[str]:
    text = str(value or "").strip()
    return None if text.lower() in ("", "none", "nan", "na") else text


def _ref(row):
    cid = row.get("corpus_id")
    try:
        return f"{row.get('source')}:{int(cid)}"
    except (TypeError, ValueError):
        return None


def _to_paper(row) -> Paper:
    row = row.copy()
    row["doi"] = get_clean_doi(str(row.get("doi", "") or ""))
    return Paper(
        title=str(row.get("title", "") or "").strip() or "N/A",
        authors=str(row.get("authors", "") or "").strip() or "N/A",
        journal=_optional_str(row.get("journal")) or "N/A",
        doi=row["doi"] or "N/A",
        abstract=str(row.get("abstract", "") or "").strip() or "N/A",
        date=_optional_str(row.get("date")),
        year=paper_links.year_of(row),
        score=round(float(row.get("score", 0.0)), 4),
        source=str(row.get("source", "") or "").strip(),
        url=paper_links.primary_link(row),
        links=[Link(label=label, url=url) for label, url in paper_links.build_links(row)],
        pmid=_optional_str(row.get("pmid")),
        labels=paper_links.badges(row),
        retracted=paper_links.is_retracted(row),
        published_doi=_optional_str(row.get("published_doi")),
        preprint_doi=_optional_str(row.get("preprint_doi")),
        matched_terms=list(row.get("matched_terms")) if isinstance(row.get("matched_terms"), (list, tuple, np.ndarray)) else [],
        pmcid=_optional_str(row.get("pmcid")),
        registry_id=_optional_str(row.get("nct_id")) or _optional_str(row.get("grant_id")),
        ref=_ref(row),
    )


# ---------------------------------------------------------------------------
# Search service (shared by REST and MCP)
# ---------------------------------------------------------------------------
_SEARCH_SLOTS = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)


class ServerBusy(SearchFailed):
    pass


class BadReference(SearchFailed):
    pass


async def _acquire_slot():
    try:
        await asyncio.wait_for(_SEARCH_SLOTS.acquire(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise ServerBusy("The server is busy. Please try again in a few seconds.") from None


def _config_for_source(source: str):
    return next((c for c in CONFIGS or [] if c and c.get("source_name") == source), None)


def _map_positions(rows: list, packed) -> tuple:
    """Map positions for the query code and for each result row (None where unknown)."""
    if MAP is None or not MAP.loaded:
        return None, [None] * len(rows)
    codes = np.zeros((len(rows), 48), dtype=np.uint8)
    known = np.zeros(len(rows), dtype=bool)
    by_source: dict = {}
    for i, row in enumerate(rows):
        try:
            by_source.setdefault(str(row.get("source")), []).append((i, int(row.get("corpus_id"))))
        except (TypeError, ValueError):
            pass
    for source, items in by_source.items():
        cfg = _config_for_source(source)
        if not cfg:
            continue
        idx, gids = zip(*items)
        found_codes, found = get_or_create_searcher(cfg).codes_for(list(gids))
        codes[list(idx)] = found_codes
        known[list(idx)] = found
    xy = MAP.place(np.vstack([np.asarray(packed, dtype=np.uint8).reshape(1, -1), codes]))
    rounded = [[round(float(a), 4), round(float(b), 4)] for a, b in xy]
    return rounded[0], [p if k else None for p, k in zip(rounded[1:], known)]


def _with_map(response: "SearchResponse", rows: list, seed_rows: list, packed) -> "SearchResponse":
    try:
        query_xy, xys = _map_positions(seed_rows + rows, packed)
    except Exception as exc:              # the map is optional; never fail a search over it
        LOGGER.error(f"Map placement failed: {exc}")
        return response
    response.query_map_xy = query_xy
    for paper, xy in zip(response.seeds + response.results, xys):
        paper.map_xy = xy
    return response


async def run_similar(refs: List[str], top_k: int = 10, start_date: Optional[str] = None,
                      end_date: Optional[str] = None, high_quality_only: bool = True,
                      sources: Optional[List[str]] = None) -> dict:
    """Papers similar to example papers; raises SearchFailed with a client-safe message."""
    if CONFIGS is None:
        raise SearchFailed("Search index configuration is unavailable.")
    await _acquire_slot()
    t0 = time.perf_counter()
    try:
        df, seeds, packed = await asyncio.to_thread(
            similar_search, refs, CONFIGS, _configs_for(sources), top_k, start_date, end_date, high_quality_only)
    except UnknownReference as exc:
        raise BadReference(str(exc)) from None
    except Exception as exc:
        LOGGER.exception("Similar search failed")
        raise SearchFailed("Search failed. Please try again later.") from exc
    finally:
        _SEARCH_SLOTS.release()
    elapsed = round(time.perf_counter() - t0, 2)
    seed_papers = [_to_paper(pd.Series(sd)) for sd in seeds]
    titles = "; ".join(p.title for p in seed_papers)
    LOGGER.info(f"Similar ({len(refs)} example(s)) → {len(df)} results in {elapsed}s")
    response = SearchResponse(
        query=f"Similar to: {titles}"[:500], seeds=seed_papers, query_bits=packed.tobytes().hex(),
        total_results=len(df), search_time_seconds=elapsed,
        results=[_to_paper(row) for _, row in df.iterrows()],
    )
    response = await asyncio.to_thread(_with_map, response, [r for _, r in df.iterrows()], list(seeds), packed)
    return response.model_dump()


async def run_search(query: str, top_k: int = 10, start_date: Optional[str] = None,
                     end_date: Optional[str] = None, high_quality_only: bool = True,
                     sources: Optional[List[str]] = None) -> dict:
    """Embeds the query and searches; raises SearchFailed with a client-safe message."""
    if CONFIGS is None:
        raise SearchFailed("Search index configuration is unavailable.")
    await _acquire_slot()

    t0 = time.perf_counter()
    try:
        packed, query_float = await asyncio.to_thread(embedder.encode_query, query)
        results_df = await asyncio.to_thread(
            combined_search_orchestrator, packed, _configs_for(sources), top_k,
            start_date, end_date, high_quality_only, query_float, query,
        )
    except embedder.EmbeddingError as exc:
        raise SearchFailed("Embedding model unavailable. Please try again later.") from exc
    except Exception as exc:
        LOGGER.exception("Search failed")
        raise SearchFailed("Search failed. Please try again later.") from exc
    finally:
        _SEARCH_SLOTS.release()

    elapsed = round(time.perf_counter() - t0, 2)
    LOGGER.info(f"Search ({len(query)} chars, top_k={top_k}) → {len(results_df)} results in {elapsed}s")
    papers = [_to_paper(row) for _, row in results_df.iterrows()]
    response = SearchResponse(query=query, query_bits=packed.tobytes().hex(), total_results=len(papers),
                              search_time_seconds=elapsed, results=papers)
    response = await asyncio.to_thread(_with_map, response, [r for _, r in results_df.iterrows()], [], packed)
    return response.model_dump()


_STATS_CACHE: dict = {"at": 0.0, "value": None}


def _compute_stats() -> dict:
    from utils import report_dates_from_metadata

    sources = {}
    for name, cfg in zip(SOURCES, CONFIGS or []):
        if not cfg:
            continue
        info = {"papers": 0, "updated": None}
        meta_path = cfg.get("metadata_path", "")
        try:
            with open(meta_path) as f:
                info["papers"] = int(json.load(f).get("total_rows", 0))
            info["updated"] = datetime.fromtimestamp(os.path.getmtime(meta_path), timezone.utc).date().isoformat()
        except Exception:
            pass
        searcher = _SEARCHER_CACHE.get(cfg.get("chunk_dir"))
        if searcher is not None and searcher.superseded_count:
            # count unique papers, not the older record versions that search hides
            info["papers"] -= searcher.superseded_count
            info["older_versions_hidden"] = searcher.superseded_count
        if searcher is not None and searcher.data_problem:
            info["available"] = False
            info["problem"] = searcher.data_problem
        if name in ("BioRxiv", "MedRxiv"):
            fetched = report_dates_from_metadata(cfg)
            if fetched and fetched != "N/A":
                info["updated"] = fetched
        sources[name] = info
    return {
        "total_papers": sum(s["papers"] for s in sources.values()),
        "sources": sources,
        "embedding_model": embedder.MODEL_ID,
    }


async def get_stats() -> dict:
    if CONFIGS is None:
        raise SearchFailed("Search index configuration is unavailable.")
    if _STATS_CACHE["value"] is None or time.monotonic() - _STATS_CACHE["at"] > 300:
        _STATS_CACHE["value"] = await asyncio.to_thread(_compute_stats)
        _STATS_CACHE["at"] = time.monotonic()
    return _STATS_CACHE["value"]


# ---------------------------------------------------------------------------
# MCP endpoint for AI agents
# ---------------------------------------------------------------------------
mcp = build_mcp(
    search=lambda **kw: run_search(**kw, high_quality_only=True),
    stats=get_stats,
    similar=lambda **kw: run_similar(**kw, high_quality_only=True),
    max_top_k=PUBLIC_MAX_TOP_K,
)
_mcp_app = mcp.streamable_http_app()   # also creates mcp.session_manager


_QUOTA_TOOLS = {"search_papers", "find_similar"}


class _McpAccess:
    """
    ASGI wrapper for /mcp. Keys via X-API-Key or 'Authorization: Bearer <key>';
    without a key the anonymous tier applies. Every search tool call counts
    against the caller's daily quota; listing tools etc. is free.
    """

    def __init__(self, app):
        self.app = app

    @staticmethod
    async def _reply(send, status: int, detail: str, extra=()):
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), *extra]})
        await send({"type": "http.response.body", "body": json.dumps({"detail": detail}).encode()})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        key = headers.get("x-api-key", "")
        auth = headers.get("authorization", "")
        if not key and auth.lower().startswith("bearer "):
            key = auth[7:].strip()
        caller = resolve_caller(key, _client_ip((scope.get("client") or (None,))[0], headers))
        if caller is None:
            await self._reply(send, 401, "Invalid API key (X-API-Key or Bearer).", [(b"www-authenticate", b"Bearer")])
            return

        # Buffer the request body to see whether it calls a search tool.
        body, more = b"", True
        while more:
            message = await receive()
            body += message.get("body", b"")
            more = message.get("more_body", False)
        try:
            payload = json.loads(body) if body else None
            messages = payload if isinstance(payload, list) else [payload]
            calls = sum(1 for m in messages if isinstance(m, dict) and m.get("method") == "tools/call"
                        and (m.get("params") or {}).get("name") in _QUOTA_TOOLS)
        except ValueError:
            calls = 0
        try:
            for _ in range(calls):
                consume_quota(caller)
        except QuotaExceeded as exc:
            await self._reply(send, 429, str(exc), [(b"retry-after", str(_seconds_to_utc_midnight()).encode())])
            return

        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


# ---------------------------------------------------------------------------
# Lifespan: model, index, periodic updates, MCP sessions
# ---------------------------------------------------------------------------
async def _periodic_update_checker():
    """Background task: wakes hourly to pick up new embedding files."""
    while True:
        await asyncio.sleep(UPDATE_INTERVAL_S)
        if MAP is not None:
            try:
                await asyncio.to_thread(MAP.refresh)
            except Exception as exc:
                LOGGER.error(f"Map refresh failed: {exc}")
        if CONFIGS:
            try:
                if await asyncio.to_thread(trigger_database_updates, CONFIGS):
                    LOGGER.info("Periodic update: new data found and indexed.")
                    _STATS_CACHE["value"] = None
            except Exception as exc:
                LOGGER.error(f"Periodic update check failed: {exc}")


@asynccontextmanager
async def lifespan(app_: FastAPI):
    global _SEARCH_SLOTS
    _SEARCH_SLOTS = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)   # bound to the serving loop
    tasks = []
    try:
        imported = await asyncio.to_thread(access.import_legacy_keys, API_KEYS_FILE)
        if imported:
            LOGGER.info(f"Imported {imported} key(s) from {API_KEYS_FILE} as partner keys")
    except Exception as exc:
        LOGGER.error(f"API key import failed: {exc}")
    try:
        await asyncio.to_thread(embedder.load_model)
    except Exception as exc:
        LOGGER.error(f"Embedding model failed to load: {exc}")
    if CONFIGS:
        LOGGER.info("Startup: checking for new embedding files …")
        try:
            await asyncio.to_thread(trigger_database_updates, CONFIGS)
        except Exception as exc:
            LOGGER.error(f"Startup update check failed: {exc}")
        warm_up_indexes(CONFIGS, background=True)
        if MAP is not None:
            try:
                await asyncio.to_thread(MAP.refresh)
            except Exception as exc:
                LOGGER.error(f"Map failed to load: {exc}")
        tasks.append(asyncio.create_task(_periodic_update_checker()))
    async with mcp.session_manager.run():
        yield
    for task in tasks:
        task.cancel()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Manuscript Search API",
    description=(
        "Semantic search across ~49 M biomedical and scientific abstracts "
        "(PubMed, bioRxiv, medRxiv, arXiv). Authenticate with the X-API-Key header. "
        "AI agents can connect over MCP at /mcp with the same key."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
_cors_origins = [o for o in os.environ.get("MSS_CORS_ORIGINS", "").split(",") if o] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,  # wildcard origin + credentials is invalid per CORS spec
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type", "Authorization", "Mcp-Session-Id"],
)


class _McpRoute:
    """Serves /mcp and /mcp/ alike (no redirect — some MCP clients do not re-POST)."""

    def __init__(self, app, mcp_app):
        self.app, self.mcp_app = app, mcp_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].rstrip("/") == "/mcp":
            scope = dict(scope, path="/", raw_path=b"/")
            await self.mcp_app(scope, receive, send)
            return
        await self.app(scope, receive, send)


app.add_middleware(_McpRoute, mcp_app=_McpAccess(_mcp_app))

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _caller(request: Request, api_key: Optional[str] = Security(_api_key_header)) -> Caller:
    headers = {k.lower(): v for k, v in request.headers.items()}
    caller = resolve_caller(api_key, _client_ip(request.client.host if request.client else None, headers))
    if caller is None:
        raise HTTPException(status_code=401, detail="Invalid API key. Pass it as the X-API-Key header.",
                            headers={"WWW-Authenticate": "ApiKey"})
    return caller


def _charge(caller: Caller, response: Response):
    try:
        remaining = consume_quota(caller)
    except QuotaExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc),
                            headers={"Retry-After": str(_seconds_to_utc_midnight())}) from None
    if remaining is not None:
        response.headers["X-RateLimit-Limit"] = str(caller.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["status"], summary="Health check (no auth required)")
async def health():
    ready = CONFIGS is not None and embedder.is_loaded()
    body = {
        "status": "ok" if ready else "starting",
        "configs_loaded": CONFIGS is not None,
        "model_loaded": embedder.is_loaded(),
    }
    if not ready:
        raise HTTPException(status_code=503, detail=body)
    return body


@app.get("/v1/stats", tags=["status"], summary="Corpus sizes and last update per database")
async def stats():
    try:
        return await get_stats()
    except SearchFailed as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@app.post("/search", response_model=SearchResponse, tags=["search"],
          summary="Semantic search for scientific abstracts")
@app.post("/v1/search", response_model=SearchResponse, tags=["search"], include_in_schema=False)
async def search(req: SearchRequest, response: Response, caller: Caller = Depends(_caller)):
    """
    Performs semantic search across PubMed, bioRxiv, medRxiv and arXiv.

    - **query**: natural-language search string (≤ 2,000 characters)
    - **top_k**: 1–10 results (default 10)
    - **start_date / end_date**: optional date range filter (YYYY-MM-DD)
    - **high_quality_only**: skip papers without meaningful abstracts
    - **sources**: optional subset of databases

    Without an API key: 20 searches per day per IP; free keys from /signup: 1,000/day.
    Returns 429 when the daily limit is reached and 503 (Retry-After) when the server is busy.
    """
    max_top_k, max_chars = ((INTERNAL_MAX_TOP_K, INTERNAL_MAX_QUERY_CHARS) if caller.tier == "internal"
                            else (PUBLIC_MAX_TOP_K, PUBLIC_MAX_QUERY_CHARS))
    if req.top_k > max_top_k:
        raise HTTPException(status_code=422, detail=f"top_k must be between 1 and {max_top_k}.")
    if len(req.query) > max_chars:
        raise HTTPException(status_code=422, detail=f"query must be at most {max_chars} characters.")
    if req.start_date and req.end_date and req.start_date > req.end_date:
        raise HTTPException(status_code=422, detail="start_date must not be after end_date.")
    _charge(caller, response)

    try:
        return await run_search(
            req.query, req.top_k,
            req.start_date.isoformat() if req.start_date else None,
            req.end_date.isoformat() if req.end_date else None,
            req.high_quality_only, req.sources,
        )
    except ServerBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "10"}) from None
    except SearchFailed as exc:
        raise HTTPException(status_code=503 if "unavailable" in str(exc) else 500,
                            detail=str(exc)) from None


@app.post("/v1/similar", response_model=SearchResponse, tags=["search"],
          summary="Papers similar to one or more example papers")
async def similar(req: SimilarRequest, response: Response, caller: Caller = Depends(_caller)):
    """
    "More like this": pass the `ref` of one or more results (up to 20). Their stored
    embeddings are averaged into one query; the examples themselves are left out.
    """
    max_top_k = INTERNAL_MAX_TOP_K if caller.tier == "internal" else PUBLIC_MAX_TOP_K
    if req.top_k > max_top_k:
        raise HTTPException(status_code=422, detail=f"top_k must be between 1 and {max_top_k}.")
    if req.start_date and req.end_date and req.start_date > req.end_date:
        raise HTTPException(status_code=422, detail="start_date must not be after end_date.")
    _charge(caller, response)
    try:
        return await run_similar(req.refs, req.top_k,
                                 req.start_date.isoformat() if req.start_date else None,
                                 req.end_date.isoformat() if req.end_date else None,
                                 req.high_quality_only, req.sources)
    except BadReference as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except ServerBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "10"}) from None
    except SearchFailed as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None


# ---------------------------------------------------------------------------
# Paper map
# ---------------------------------------------------------------------------
# 1x1 transparent PNG for empty map areas
_EMPTY_TILE = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                            "1f15c4890000000d49444154789c63000100000500010d0a2db40000000049454e44ae426082")
_TILE_CACHE = {"Cache-Control": "public, max-age=31536000, immutable"}


def _require_map():
    if MAP is None or not MAP.loaded:
        raise HTTPException(status_code=503, detail="The paper map is not available right now.")
    return MAP


@app.get("/v1/map", tags=["map"], summary="Paper map: version, coordinate system, topic labels")
async def map_info():
    """
    Everything a client needs to draw the map: tile URL template, the world square
    (map coordinates -> tile pixels at `max_zoom`), source colours and topic labels.
    Search results carry `map_xy` in the same coordinates.
    """
    m = _require_map()
    return {**m.meta, "version": m.version, "tiles": f"/map/tiles/{m.version}/{{z}}/{{x}}/{{y}}.png",
            "labels": m.labels}


@app.get("/v1/map/nearby", tags=["map"], summary="Papers at a point of the map")
async def map_nearby(x: float, y: float, k: int = 8):
    """The papers nearest to map point (x, y) among the 1.52M reference papers, with their `ref`."""
    m = _require_map()
    k = max(1, min(int(k), 20))
    found = await asyncio.to_thread(m.nearby, x, y, k * 2)
    papers = await asyncio.to_thread(_nearby_papers, found, k)
    return {"x": x, "y": y, "papers": papers}


def _nearby_papers(found: list, k: int) -> list:
    wanted = []
    for source, stem, row, dist in found:
        cfg = _config_for_source(source)
        if not cfg:
            continue
        searcher = get_or_create_searcher(cfg)
        starts = {iv["source_stem"]: iv["global_start"] - iv["source_local_start"] for iv in searcher.intervals}
        if stem not in starts:
            continue
        gid = int(starts[stem] + row)
        if not searcher.is_superseded([gid])[0]:
            wanted.append((source, gid, dist))
    wanted = wanted[:k]
    out = {}
    for source in {w[0] for w in wanted}:
        cands = [{"corpus_id": g, "score": 0.0} for s, g, _ in wanted if s == source]
        df = get_or_create_searcher(_config_for_source(source)).fetch_rows(cands)
        for _, row in df.iterrows():
            row["source"] = source
            out[(source, int(row["corpus_id"]))] = row
    papers = []
    for source, gid, dist in wanted:
        row = out.get((source, gid))
        if row is None:
            continue
        p = _to_paper(row)
        papers.append({"title": p.title, "year": p.year, "journal": p.journal, "source": p.source,
                       "url": p.url, "ref": p.ref, "labels": p.labels, "distance": round(dist, 4)})
    return papers


@app.get("/map/tiles/{version}/{z}/{x}/{y}.png", include_in_schema=False)
async def map_tile(version: str, z: int, x: int, y: int):
    m = _require_map()
    if version != m.version:
        # an older version's URL (cached page): serve the current tiles, briefly cacheable
        path, headers = m.tile_path(z, x, y), {"Cache-Control": "public, max-age=300"}
    else:
        path, headers = m.tile_path(z, x, y), _TILE_CACHE
    if path is None:
        return Response(content=_EMPTY_TILE, media_type="image/png", headers=headers)
    return FileResponse(path, media_type="image/png", headers=headers)


# ---------------------------------------------------------------------------
# Self-service API keys (/signup)
# ---------------------------------------------------------------------------
TURNSTILE_SITEKEY = os.environ.get("MSS_TURNSTILE_SITEKEY", "")
TURNSTILE_SECRET = os.environ.get("MSS_TURNSTILE_SECRET", "")
SMTP = {k: os.environ.get(f"MSS_SMTP_{k.upper()}", "") for k in ("host", "port", "user", "password", "from")}
SIGNUP_ENABLED = bool(TURNSTILE_SITEKEY and TURNSTILE_SECRET and SMTP["host"] and SMTP["from"])
_EMAIL_RE = __import__("re").compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[A-Za-z]{2,24}$")


class KeyRequest(BaseModel):
    email: str = Field(..., max_length=254)
    turnstile_token: str = Field(..., max_length=4096)


def _send_key_email(to: str, key: str):
    import smtplib
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["Subject"] = "Your Manuscript Search API key"
    msg["From"], msg["To"] = SMTP["from"], to
    msg.set_content(
        f"Here is your API key:\n\n    {key}\n\n"
        f"Use it in the X-API-Key header (REST) or as a Bearer token (MCP at /mcp).\n"
        f"Limit: {access.LIMITS['free']} searches per day. Docs: https://manuscript-search.org/docs\n\n"
        "If you did not request this key, ignore this email.")
    with smtplib.SMTP(SMTP["host"], int(SMTP["port"] or 587), timeout=30) as smtp:
        smtp.starttls()
        if SMTP["user"]:
            smtp.login(SMTP["user"], SMTP["password"])
        smtp.send_message(msg)


@app.get("/signup", response_class=HTMLResponse, include_in_schema=False)
async def signup_page():
    with open(os.path.join(_HERE, "signup.html")) as f:
        page = f.read()
    if SIGNUP_ENABLED:
        form = (f'<form id="f"><label for="email">Email address</label>'
                f'<input id="email" name="email" type="email" required autocomplete="email">'
                f'<div class="cf-turnstile" data-sitekey="{TURNSTILE_SITEKEY}"></div>'
                f'<button id="b" type="submit">Email me a key</button></form><p id="result" role="status"></p>')
        script = ("<script>document.getElementById('f').addEventListener('submit', async e => {"
                  "e.preventDefault(); const b=document.getElementById('b'); b.disabled=true;"
                  "const r=await fetch('/v1/keys/request',{method:'POST',headers:{'Content-Type':'application/json'},"
                  "body:JSON.stringify({email:document.getElementById('email').value,"
                  "turnstile_token:(document.querySelector('[name=cf-turnstile-response]')||{}).value||''})});"
                  "const j=await r.json(); document.getElementById('result').textContent=j.message||j.detail;"
                  "b.disabled=false; if(window.turnstile) turnstile.reset();});</script>")
        turnstile = '<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>'
    else:
        form = ("<p><strong>Self-service keys are coming soon.</strong> Until then you can use the "
                "API without a key, within the daily limit above.</p>")
        script = turnstile = ""
    page = (page.replace("{form}", form).replace("{form_script}", script).replace("{turnstile_script}", turnstile)
                .replace("{anon}", f"{access.LIMITS['anonymous']:,}").replace("{free}", f"{access.LIMITS['free']:,}"))
    return HTMLResponse(page)


@app.post("/v1/keys/request", include_in_schema=False)
async def request_key(req: KeyRequest, request: Request):
    if not SIGNUP_ENABLED:
        raise HTTPException(status_code=503, detail="Self-service keys are not enabled yet.")
    email = req.email.strip()
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="Please enter a valid email address.")
    headers = {k.lower(): v for k, v in request.headers.items()}
    ip = _client_ip(request.client.host if request.client else None, headers)
    reason = await asyncio.to_thread(access.signup_allowed, ip, email)
    if reason:
        raise HTTPException(status_code=429, detail=reason)

    def _verify() -> bool:
        import requests as _rq
        r = _rq.post("https://challenges.cloudflare.com/turnstile/v0/siteverify", timeout=15,
                     data={"secret": TURNSTILE_SECRET, "response": req.turnstile_token, "remoteip": ip})
        return bool(r.ok and r.json().get("success"))

    if not await asyncio.to_thread(_verify):
        raise HTTPException(status_code=400, detail="The bot check failed; please try again.")
    await asyncio.to_thread(access.record_signup, ip, email)
    key = await asyncio.to_thread(access.create_key, "free", "self-service", email)
    try:
        await asyncio.to_thread(_send_key_email, email, key)
    except Exception:
        LOGGER.exception("Signup email failed")
        raise HTTPException(status_code=502, detail="We could not send the email; please try again later.") from None
    return {"message": f"Done — your key is on its way to {email}."}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("search_api:app", host=API_HOST, port=API_PORT, reload=False)
