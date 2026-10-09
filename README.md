# Manuscript Search

Semantic search over ~50 million scientific abstracts and trial registrations
(PubMed, bioRxiv, medRxiv, arXiv, ClinicalTrials.gov) —
[manuscript-search.org](https://manuscript-search.org). Describe a finding
or paste an abstract; papers are ranked by meaning, not keywords.

## How it fits together

```
                        ┌──────────────────────────── server (4 vCPU, 7.8 GB) ───────────────┐
 browser ──► nginx ──►  │ mss-ui      Streamlit (pbmss_app.py)  ── HTTP, internal key ──┐   │
                │       │                                                               ▼   │
 REST / agents ─┴────►  │ mss-backend search_api.py: BGE model + binary FAISS index     │   │
   /search /v1/*        │             REST /search /v1/search /v1/stats /health  ◄──────┘   │
   /mcp                 │             MCP  /mcp (search_papers, database_info)              │
                        │             hourly pickup of new .npy/.parquet files               │
                        └──────────────────────────────────────────────────────────────────┘
                                         ▲ rsync nightly
 lab desktop (GPU): update_database/*.py download + embed new papers ──┘
```

- **One backend process** owns the embedding model and the only copy of the index,
  limits concurrent searches (others get 503 + Retry-After), and is the only process
  that ingests new data.
- **Search**: Hamming search over 384-bit codes, then candidates are re-ranked with the
  float query (+6–7 points top-1 accuracy, see `eval/README.md`). Preprints and their
  journal versions are merged; retracted papers are flagged.
- **The UI** is a thin client. **AI agents** connect over MCP with an API key.

## Run locally

```bash
pip install -r requirements.txt
export MSS_INTERNAL_KEY=dev-key MSS_CONFIG_PATH=./config_mss.yaml
python search_api.py                                   # backend on 127.0.0.1:8080
MSS_INTERNAL_KEY=dev-key streamlit run pbmss_app.py   # UI on :8501
python -m pytest tests/                                # no data or network needed
```

## Use it from code

```bash
curl -X POST https://manuscript-search.org/search \
  -H "X-API-Key: <key>" -H "Content-Type: application/json" \
  -d '{"query": "CRISPR base editing off-target effects", "top_k": 10}'
```
Interactive docs: `/docs`. Results carry `url`, `links`, `labels` (Retracted, Review,
Preprint, …), `date`, `pmid`, and for merged preprints `preprint_doi`.

## Use it from an AI agent (MCP)

Remote (streamable HTTP), e.g. Claude Code:
```bash
claude mcp add --transport http manuscript-search https://manuscript-search.org/mcp \
  --header "Authorization: Bearer <key>"
```
Local stdio bridge (for clients without remote MCP support):
```json
{"mcpServers": {"manuscript-search": {
  "command": "python", "args": ["/path/to/mcp_stdio.py"],
  "env": {"MSS_API_KEY": "<key>"}}}}
```
Tools: `search_papers(query, top_k, start_date, end_date, sources, include_abstracts)`
(sources: PubMed, BioRxiv, MedRxiv, arXiv, ClinicalTrials)
and `database_info()`.

## Deploy

```bash
bash deploy/push.sh                         # on your machine: code only, archives the old code
ssh root@server 'cd /root/pubmed_search && bash deploy/install.sh'
# back out:  bash deploy/install.sh --rollback
```
`install.sh` installs the two systemd units (`mss-backend`, `mss-ui`, with memory
limits), creates `.env` with an internal key, replaces the old three services, and
installs the nginx routes (API, MCP, rate limits, maintenance page) with automatic
restore if `nginx -t` fails.

## Repository map

| Path | What |
|---|---|
| `search_api.py` | Backend: REST, MCP mount, concurrency limit, stats |
| `search_logic.py` | Chunked binary index, retrieval, rescoring, date screening, dedup, result cache |
| `data_handler.py` | Parquet row fetching |
| `embedder.py` | In-process BGE query encoder |
| `mcp_server.py`, `mcp_stdio.py` | MCP tools; stdio bridge |
| `paper_links.py` | Links and labels per paper (shared by UI and API) |
| `pbmss_app.py`, `ui_components.py`, `ui_data.py`, `backend_client.py`, `.streamlit/` | Web UI |
| `gemini_handler.py` | Optional AI summary / chat with the user's own Gemini key |
| `update_database/` | Ingestion scripts (run on the GPU desktop) |
| `tools/rechunk_parquet.py` | Rewrite metadata parquet with small row groups |
| `eval/` | Offline ranking evaluation |
| `deploy/` | systemd, nginx, push/install scripts |
| `docs/database_expansion.md` | Which databases to add next, and how |
