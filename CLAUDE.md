# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

Manuscript Search (manuscript-search.org): semantic search over ~49M abstracts
(PubMed, bioRxiv, medRxiv, arXiv). Queries are embedded with BAAI/bge-small-en-v1.5,
binarized to 384 bits, searched with FAISS `IndexBinaryFlat` over chunked memmapped
`.npy` files, re-ranked with the float query, and joined to parquet metadata.

## Architecture (two processes)

- `search_api.py` — **the backend**. Loads the model (`embedder.py`) and the index
  once; serves REST (`/search`, `/v1/search`, `/v1/stats`, `/health`) and MCP (`/mcp`,
  tools from `mcp_server.py`); limits concurrent searches; the only process that picks
  up new embedding files (startup + hourly). Keys: `api_keys.txt` (public tier,
  ≤10 results, ≤2000 chars) and `MSS_INTERNAL_KEY` (UI tier, ≤50 results).
- `pbmss_app.py` — Streamlit UI, a thin client via `backend_client.py`. Never import
  `search_logic` from the UI: that would load a second index copy (~2.4 GB).

```bash
MSS_INTERNAL_KEY=dev python search_api.py            # :8080
MSS_INTERNAL_KEY=dev streamlit run pbmss_app.py      # :8501
python -m pytest tests/                              # synthetic data, no network
```
Use the `pubmed_search` conda env locally (same streamlit/pandas/faiss as the server).

## Search pipeline (`search_logic.py`)

1. Per source, `ChunkedSearcher` builds/caches FAISS indexes per chunk file.
2. Per chunk: Hamming top-k (5,000; 20,000 with a date filter), then
   `rescore_with_float_query` (lookup-table dot product of float query vs ±1 bits).
3. Global sort, then batches: date filters screen candidates on the `date` column only;
   full rows fetched via `data_handler.fetch_specific_rows`; `_add_or_merge` dedups by
   DOI/title and merges preprints with journal versions (published DOI or title).
4. LRU result cache keyed by query + filters + index generation.

## Constraints that matter

- **Server**: 4 vCPU, 7.8 GB RAM, no GPU. Every 1M docs = 48 MB RAM. Never add work
  that loads a second index or re-embeds on the server.
- **Don't trigger needless recomputation**: changing a source `.npy` mtime makes the
  backend rebuild that source's chunks; extending the last chunk must only evict that
  chunk's index.
- PubMed vectors were embedded **with** the BGE query prefix, the other sources
  without (`eval/README.md`); the measured effect on ranking is negligible.
- bioRxiv/medRxiv: row i of the combined parquet ↔ row i of the `.npy`. A row-count
  mismatch disables the source (`ChunkedSearcher.data_problem`).
- Parquet should use small row groups (2,000); one huge row group makes every fetch
  decode the whole file (`tools/rechunk_parquet.py`).
- Metadata JSON is written atomically (`_write_json_atomic`); keep it that way.
- All data-derived text rendered with `unsafe_allow_html` must go through
  `ui_components._e()` / `_safe_url()`.

## Data pipeline

`update_database/` scripts run on the lab desktop GPU from this repo's checkout
(cron there; paths hardcoded to `/mnt/h/...`): bioRxiv/medRxiv 02:00, PubMed
02:10/02:15 nightly, arXiv weekly (Sun 01:30, `arxiv_oai_update.py` via OAI-PMH);
02:30 rsync of `snowflake/` (.npy + .parquet) to the server, where the backend
picks the files up within the hour. Editing these scripts changes production data.

## Deploy

`deploy/push.sh` (local) then `deploy/install.sh` on the server (`--rollback` to undo).
`api_keys.txt` and `.env` live only on the server and are never committed.
