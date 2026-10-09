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
    MSS_ADMIN_SECRET             enables POST /keys/generate
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

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# Ensure local modules are importable when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import embedder
import paper_links
from config_loader import read_source_configs
from mcp_server import SOURCES, SearchFailed, build_mcp
from search_logic import (
    _SEARCHER_CACHE, combined_search_orchestrator, trigger_database_updates, warm_up_indexes,
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
def _load_api_keys(path: str) -> set:
    keys: set = set()
    try:
        with open(path) as f:
            for line in f:
                key = line.strip()
                if key and not key.startswith("#"):
                    keys.add(key)
        LOGGER.info(f"Loaded {len(keys)} API key(s) from {path}")
    except FileNotFoundError:
        LOGGER.warning(f"API keys file not found at {path}; only the internal key is accepted.")
    return keys


VALID_KEYS: set = _load_api_keys(API_KEYS_FILE)


def _key_tier(key: Optional[str]) -> Optional[str]:
    """'internal', 'public', or None for an unknown key."""
    if not key:
        return None
    if INTERNAL_KEY and secrets.compare_digest(key, INTERNAL_KEY):
        return "internal"
    return "public" if key in VALID_KEYS else None


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


def _configs_for(sources: Optional[List[str]]) -> list:
    """Source configs in search_logic order, with unselected sources blanked out."""
    if not sources:
        return CONFIGS
    wanted = {s.lower() for s in sources}
    return [cfg if name.lower() in wanted else {} for name, cfg in zip(SOURCES, CONFIGS)]


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
SourceName = Literal["PubMed", "BioRxiv", "MedRxiv", "arXiv"]


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
        "Restrict to these databases; all if omitted."))

    model_config = {"json_schema_extra": {"example": {
        "query": "CRISPR base editing off-target effects",
        "top_k": 10, "start_date": "2020-01-01", "end_date": "2024-12-31",
        "high_quality_only": True,
    }}}


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
    source: str = Field(description="Database the paper came from: PubMed | BioRxiv | MedRxiv | arXiv.")
    url: Optional[str] = Field(default=None, description="Best link to the paper (DOI, arXiv or PubMed).")
    links: List[Link] = Field(default_factory=list, description="All links: DOI, PubMed, PDF, published version, preprint.")
    pmid: Optional[str] = Field(default=None, description="PubMed ID, when available.")
    labels: List[str] = Field(default_factory=list, description=(
        "Notable publication types and status, e.g. Retracted, Review, Meta-analysis, RCT, Preprint."))
    retracted: bool = Field(default=False, description="True if the paper has been retracted.")
    published_doi: Optional[str] = Field(default=None, description="For preprints: DOI of the journal version.")
    preprint_doi: Optional[str] = Field(default=None, description="For journal articles: DOI of the merged preprint.")


class SearchResponse(BaseModel):
    query: str
    query_bits: Optional[str] = Field(default=None, description=(
        "The query's 384-bit binary code as hex — what the index matches against."))
    total_results: int
    search_time_seconds: float
    results: List[Paper]


def _optional_str(value) -> Optional[str]:
    text = str(value or "").strip()
    return None if text.lower() in ("", "none", "nan", "na") else text


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
    )


# ---------------------------------------------------------------------------
# Search service (shared by REST and MCP)
# ---------------------------------------------------------------------------
_SEARCH_SLOTS = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)


class ServerBusy(SearchFailed):
    pass


async def run_search(query: str, top_k: int = 10, start_date: Optional[str] = None,
                     end_date: Optional[str] = None, high_quality_only: bool = True,
                     sources: Optional[List[str]] = None) -> dict:
    """Embeds the query and searches; raises SearchFailed with a client-safe message."""
    if CONFIGS is None:
        raise SearchFailed("Search index configuration is unavailable.")
    try:
        await asyncio.wait_for(_SEARCH_SLOTS.acquire(), timeout=QUEUE_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise ServerBusy("The server is busy. Please try again in a few seconds.") from None

    t0 = time.perf_counter()
    try:
        packed, query_float = await asyncio.to_thread(embedder.encode_query, query)
        results_df = await asyncio.to_thread(
            combined_search_orchestrator, packed, _configs_for(sources), top_k,
            start_date, end_date, high_quality_only, query_float,
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
    return SearchResponse(query=query, query_bits=packed.tobytes().hex(), total_results=len(papers),
                          search_time_seconds=elapsed, results=papers).model_dump()


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
    max_top_k=PUBLIC_MAX_TOP_K,
)
_mcp_app = mcp.streamable_http_app()   # also creates mcp.session_manager


class _RequireApiKey:
    """ASGI wrapper: accepts X-API-Key or 'Authorization: Bearer <key>'."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            key = headers.get("x-api-key", "")
            auth = headers.get("authorization", "")
            if not key and auth.lower().startswith("bearer "):
                key = auth[7:].strip()
            if _key_tier(key) is None:
                body = json.dumps({"detail": "Invalid or missing API key (X-API-Key or Bearer)."}).encode()
                await send({"type": "http.response.start", "status": 401, "headers": [
                    (b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Lifespan: model, index, periodic updates, MCP sessions
# ---------------------------------------------------------------------------
async def _periodic_update_checker():
    """Background task: wakes hourly to pick up new embedding files."""
    while True:
        await asyncio.sleep(UPDATE_INTERVAL_S)
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
app.mount("/mcp", _RequireApiKey(_mcp_app))

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_admin_key_header = APIKeyHeader(name="X-Admin-Secret", auto_error=False)
_ADMIN_SECRET: str = os.environ.get("MSS_ADMIN_SECRET", "")


async def _require_key(api_key: Optional[str] = Security(_api_key_header)) -> str:
    tier = _key_tier(api_key)
    if tier is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Pass it as the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return tier


async def _require_admin(admin_secret: Optional[str] = Security(_admin_key_header)) -> None:
    if not _ADMIN_SECRET:
        raise HTTPException(status_code=503, detail=(
            "Admin endpoint disabled. Set MSS_ADMIN_SECRET environment variable to enable."))
    if not admin_secret or not secrets.compare_digest(admin_secret, _ADMIN_SECRET):
        raise HTTPException(status_code=403, detail=(
            "Invalid or missing admin secret. Pass it as the X-Admin-Secret header."))


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
        "api_keys_loaded": len(VALID_KEYS),
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
async def search(req: SearchRequest, tier: str = Depends(_require_key)):
    """
    Performs semantic search across PubMed, bioRxiv, medRxiv and arXiv.

    - **query**: natural-language search string (≤ 2,000 characters)
    - **top_k**: 1–10 results (default 10)
    - **start_date / end_date**: optional date range filter (YYYY-MM-DD)
    - **high_quality_only**: skip papers without meaningful abstracts
    - **sources**: optional subset of databases

    Returns 503 with a Retry-After header when the server is busy.
    """
    max_top_k, max_chars = ((INTERNAL_MAX_TOP_K, INTERNAL_MAX_QUERY_CHARS) if tier == "internal"
                            else (PUBLIC_MAX_TOP_K, PUBLIC_MAX_QUERY_CHARS))
    if req.top_k > max_top_k:
        raise HTTPException(status_code=422, detail=f"top_k must be between 1 and {max_top_k}.")
    if len(req.query) > max_chars:
        raise HTTPException(status_code=422, detail=f"query must be at most {max_chars} characters.")
    if req.start_date and req.end_date and req.start_date > req.end_date:
        raise HTTPException(status_code=422, detail="start_date must not be after end_date.")

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


@app.post("/keys/generate", tags=["admin"],
          summary="Generate a new random API key (requires X-Admin-Secret header)",
          dependencies=[Depends(_require_admin)])
async def generate_key(prefix: str = "mss"):
    """
    Generates a secure random API key. The key is NOT saved automatically —
    append it to api_keys.txt and restart the service.
    """
    token = f"{prefix}_{secrets.token_urlsafe(32)}"
    return {"api_key": token,
            "note": f"Add this key to {API_KEYS_FILE} (one key per line) and restart the server."}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("search_api:app", host=API_HOST, port=API_PORT, reload=False)
