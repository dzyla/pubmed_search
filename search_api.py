"""
PubMed Semantic Search — REST API
Serves semantic search results over PubMed, BioRxiv, MedRxiv, and arXiv.

Start with:
    uvicorn search_api:app --host 0.0.0.0 --port 8080

Environment variables (all optional):
    MSS_CONFIG_PATH    Path to config_mss.yaml  (default: /root/pubmed_search/config_mss.yaml)
    MSS_MODEL_URL      Embedding server URL      (default: http://localhost:8000/encode)
    MSS_API_KEYS_FILE  Path to api_keys.txt      (default: api_keys.txt)
    MSS_API_PORT       Port when run directly    (default: 8080)
"""

import os
import sys
import time
import asyncio
import logging
import secrets
import yaml
import requests
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware


# Ensure local modules are importable when run from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from search_logic import combined_search_orchestrator, trigger_database_updates, warm_up_indexes
from utils import get_clean_doi
from api_handler import get_query_embedding_packed, EmbeddingError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
CONFIG_PATH    = os.environ.get("MSS_CONFIG_PATH",    "/root/pubmed_search/config_mss.yaml")
MODEL_URL      = os.environ.get("MSS_MODEL_URL",      "http://localhost:8000/encode")
API_KEYS_FILE  = os.environ.get("MSS_API_KEYS_FILE",  os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_keys.txt"))
API_PORT       = int(os.environ.get("MSS_API_PORT",   "8080"))

# ---------------------------------------------------------------------------
# API key store — loaded once at startup
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
        LOGGER.warning(
            f"API keys file not found at {path}. "
            "All requests will be rejected. "
            "Create the file with one key per line."
        )
    return keys


VALID_KEYS: set = _load_api_keys(API_KEYS_FILE)

# ---------------------------------------------------------------------------
# Search index configs — loaded once at startup (no Streamlit dependency)
# ---------------------------------------------------------------------------
def _load_configs(yaml_path: str):
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        configs = [
            data.get("pubmed_config",  {}),
            data.get("biorxiv_config", {}),
            data.get("medrxiv_config", {}),
            data.get("arxiv_config",   {}),
        ]
        LOGGER.info(f"Loaded search configs from {yaml_path}")
        return configs
    except FileNotFoundError:
        LOGGER.error(f"Config file not found: {yaml_path}")
        return None
    except Exception as exc:
        LOGGER.error(f"Failed to parse config {yaml_path}: {exc}")
        return None


CONFIGS = _load_configs(CONFIG_PATH)

# How often (seconds) to check source dirs for new embedding files after startup.
UPDATE_INTERVAL_S = int(os.environ.get("MSS_UPDATE_INTERVAL_S", str(60 * 60)))  # default: 1 h


async def _periodic_update_checker():
    """Background task: wakes hourly to pick up new embedding files."""
    while True:
        await asyncio.sleep(UPDATE_INTERVAL_S)
        if CONFIGS:
            try:
                updated = await asyncio.to_thread(trigger_database_updates, CONFIGS)
                if updated:
                    LOGGER.info("Periodic update: new data found and indexed.")
            except Exception as exc:
                LOGGER.error(f"Periodic update check failed: {exc}")


@asynccontextmanager
async def lifespan(app_: FastAPI):
    """Run startup tasks then yield control to the request loop."""
    if CONFIGS:
        LOGGER.info("Startup: checking for new embedding files …")
        try:
            await asyncio.to_thread(trigger_database_updates, CONFIGS)
        except Exception as exc:
            LOGGER.error(f"Startup update check failed: {exc}")
        warm_up_indexes(CONFIGS, background=True)
        asyncio.create_task(_periodic_update_checker())
    yield


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PubMed Semantic Search API",
    description=(
        "Semantic search across 50 M+ biomedical and scientific abstracts "
        "(PubMed, BioRxiv, MedRxiv, arXiv). "
        "Authenticate via the X-API-Key request header."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
# Add this right after app = FastAPI(...)
_cors_origins = [o for o in os.environ.get("MSS_CORS_ORIGINS", "").split(",") if o] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,  # wildcard origin + credentials is invalid per CORS spec
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_admin_key_header = APIKeyHeader(name="X-Admin-Secret", auto_error=False)

# Admin secret for privileged endpoints. Set MSS_ADMIN_SECRET to enable.
_ADMIN_SECRET: str = os.environ.get("MSS_ADMIN_SECRET", "")


async def _require_key(api_key: Optional[str] = Security(_api_key_header)) -> str:
    if api_key and api_key in VALID_KEYS:
        return api_key
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key. Pass it as the X-API-Key header.",
        headers={"WWW-Authenticate": "ApiKey"},
    )


async def _require_admin(admin_secret: Optional[str] = Security(_admin_key_header)) -> None:
    if not _ADMIN_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Admin endpoint disabled. Set MSS_ADMIN_SECRET environment variable to enable.",
        )
    if not admin_secret or not secrets.compare_digest(admin_secret, _ADMIN_SECRET):
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing admin secret. Pass it as the X-Admin-Secret header.",
        )

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Free-text search query (e.g. 'CRISPR base editing off-target effects').",
    )
    top_k: int = Field(
        default=10,
        ge=5,
        le=10,
        description="Number of results to return. Must be between 5 and 10.",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Restrict results to papers published on or after this date (YYYY-MM-DD).",
        examples=["2020-01-01"],
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Restrict results to papers published on or before this date (YYYY-MM-DD).",
        examples=["2024-12-31"],
    )
    high_quality_only: bool = Field(
        default=True,
        description="When true, papers with very short or missing abstracts are excluded.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "query": "CRISPR base editing off-target effects",
            "top_k": 10,
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "high_quality_only": True,
        }
    }}


class Paper(BaseModel):
    title: str
    authors: str
    journal: str
    doi: str
    abstract: str
    year: Optional[int]
    score: float = Field(description="Relevance score between 0.0 (no match) and 1.0 (perfect match).")
    source: str = Field(description="Database the paper came from: PubMed | BioRxiv | MedRxiv | arXiv.")


class SearchResponse(BaseModel):
    query: str
    total_results: int
    search_time_seconds: float
    results: List[Paper]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["status"], summary="Health check (no auth required)")
async def health():
    """Returns service status and whether the embedding server is reachable."""
    model_ok = False
    try:
        # Run the blocking requests.get in a thread so the event loop stays free.
        def _ping() -> bool:
            r = requests.get(MODEL_URL.replace("/encode", "/health"), timeout=2)
            return r.status_code == 200
        model_ok = await asyncio.to_thread(_ping)
    except Exception:
        pass
    return {
        "status": "ok",
        "configs_loaded": CONFIGS is not None,
        "api_keys_loaded": len(VALID_KEYS),
        "embedding_server": MODEL_URL,
        "embedding_server_reachable": model_ok,
    }


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["search"],
    summary="Semantic search for scientific abstracts",
    dependencies=[Depends(_require_key)],
)
async def search(req: SearchRequest):
    """
    Performs semantic search across PubMed, BioRxiv, MedRxiv, and arXiv.

    - **query**: natural-language search string
    - **top_k**: 5–10 results (default 10)
    - **start_date / end_date**: optional date range filter (YYYY-MM-DD)
    - **high_quality_only**: skip papers without meaningful abstracts

    A single search scans 50 M+ embeddings and typically completes in 4–8 s.
    """
    if CONFIGS is None:
        raise HTTPException(status_code=503, detail="Search index configuration is unavailable.")

    t0 = time.perf_counter()

    # --- Encode query via shared api_handler (no Streamlit dependency) ---
    try:
        query_packed = get_query_embedding_packed(req.query, server_url=MODEL_URL)
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # --- Run the CPU-bound search in a thread so the event loop stays free ---
    try:
        results_df = await asyncio.to_thread(
            combined_search_orchestrator,
            query_packed,
            CONFIGS,
            req.top_k,
            req.start_date,
            req.end_date,
            req.high_quality_only,
        )
    except Exception as exc:
        LOGGER.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search error: {exc}")

    elapsed = round(time.perf_counter() - t0, 2)
    LOGGER.info(f"Search '{req.query[:60]}' → {len(results_df)} results in {elapsed}s")

    if results_df.empty:
        return SearchResponse(
            query=req.query,
            total_results=0,
            search_time_seconds=elapsed,
            results=[],
        )

    papers: List[Paper] = []
    for _, row in results_df.iterrows():
        # Parse year from date string
        year: Optional[int] = None
        date_val = row.get("date")
        if date_val:
            try:
                year = int(str(date_val)[:4])
            except (ValueError, TypeError):
                pass

        papers.append(Paper(
            title=str(row.get("title",    "") or "").strip() or "N/A",
            authors=str(row.get("authors",  "") or "").strip() or "N/A",
            journal=str(row.get("journal",  "") or "").strip() or "N/A",
            doi=get_clean_doi(str(row.get("doi", "") or "")) or "N/A",
            abstract=str(row.get("abstract", "") or "").strip() or "N/A",
            year=year,
            score=round(float(row.get("score", 0.0)), 4),
            source=str(row.get("source",  "") or "").strip(),
        ))

    return SearchResponse(
        query=req.query,
        total_results=len(papers),
        search_time_seconds=elapsed,
        results=papers,
    )


@app.post(
    "/keys/generate",
    tags=["admin"],
    summary="Generate a new random API key (requires X-Admin-Secret header)",
    dependencies=[Depends(_require_admin)],
)
async def generate_key(prefix: str = "mss"):
    """
    Generates a secure random API key.  The key is NOT saved automatically —
    append it to api_keys.txt manually then restart the server.

    Requires the X-Admin-Secret header to match MSS_ADMIN_SECRET env var.
    """
    token = f"{prefix}_{secrets.token_urlsafe(32)}"
    return {
        "api_key": token,
        "note": f"Add this key to {API_KEYS_FILE} (one key per line) and restart the server.",
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("search_api:app", host="0.0.0.0", port=API_PORT, reload=False)
