# Manuscript Search

Semantic search over ~50 million scientific abstracts and trial registrations
(PubMed, bioRxiv, medRxiv, arXiv, ClinicalTrials.gov) —
[manuscript-search.org](https://manuscript-search.org). Describe a finding
or paste an abstract; papers are ranked by meaning, not keywords.

## How it fits together

```
                        ┌──────────────────────────── server (CPU only) ────────────────────┐
 browser ──► nginx ──►  │ mss-ui      Streamlit (pbmss_app.py)  ── HTTP, internal key ──┐   │
                │       │                                                               ▼   │
 REST / agents ─┴────►  │ mss-backend search_api.py: BGE model + binary FAISS index     │   │
   /search /v1/*        │             REST /search /v1/search /v1/stats /health  ◄──────┘   │
   /mcp                 │             MCP  /mcp (search, similar, library, citations)       │
                        │             hourly pickup of new .npy/.parquet files               │
                        └──────────────────────────────────────────────────────────────────┘
                                         ▲ new data files
 GPU machine: update_database/*.py download + embed new papers ──────┘
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
cp config_mss.example.yaml config_mss.yaml           # then point the paths at your data
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
Interactive docs: `/docs`. Results carry `url`, `links`, `labels` (Published or Preprint,
Retracted, Review, …), `date`, `pmid`, a `ref` for follow-up calls, and for merged
preprints `preprint_doi`.

| Endpoint | What |
|---|---|
| `POST /search` | Search by meaning (query, dates, sources) |
| `POST /v1/similar` | Papers like 1–20 results (`refs`) |
| `POST /v1/library` | Papers like your library: up to 500 DOIs / PMIDs / arXiv ids, or the `text` of a BibTeX, RIS or PubMed export. Grouped into up to four `themes`; your own papers are left out |
| `POST /v1/citations` | Papers that cite a paper (`cited_by`) or that it cites (`references`), from OpenAlex, matched to the index |
| `GET /v1/stats` | Database sizes and update dates |

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
(sources: PubMed, BioRxiv, MedRxiv, arXiv, ClinicalTrials, Preprints, OpenAlex, Grants),
`find_similar(paper_refs, …)`, `find_papers_like(paper_ids, …)`,
`get_citations(paper, direction)` and `database_info()`.

## Machine-specific settings

Data locations and the contact address sent to public APIs are not stored in the
code. They are read from environment variables, or from `~/.config/mss/paths.env`
(`KEY=VALUE` lines) for scheduled jobs — see `mss_env.py`:

| Variable | Used by | Default |
|---|---|---|
| `MSS_CONFIG_PATH` | `search_api.py` | `config_mss.yaml` next to the code |
| `MSS_DATA_ROOT` | `update_database/`, `map/` | `./data` in the repository |
| `MSS_MAP_WORK` | `map/` | `$MSS_DATA_ROOT/map_work` |
| `MSS_PIPELINE_CONFIG` | PubMed and bioRxiv/medRxiv download scripts | `config_mss.yaml` in the repository |
| `MSS_CONTACT_EMAIL` | Crossref / API polite-pool headers | none sent |
| `MSS_RERANKER_DIR` | `reranker.py` | `./models/medcpt-ce-int8` in the repository |

## Repository map

| Path | What |
|---|---|
| `search_api.py` | Backend: REST, MCP mount, concurrency limit, stats |
| `search_logic.py` | Chunked binary index, retrieval, rescoring, date screening, dedup, result cache |
| `data_handler.py` | Parquet row fetching |
| `embedder.py` | In-process BGE query encoder |
| `mcp_server.py`, `mcp_stdio.py` | MCP tools; stdio bridge |
| `paper_links.py` | Links and labels per paper (shared by UI and API) |
| `identifiers.py`, `aux_index.py` | Exact identifiers (gene symbols, variants, trial ids) and paper ids (DOI, PMID) -> indexed rows |
| `cite_graph.py` | Citation links from OpenAlex |
| `pbmss_app.py`, `ui_components.py`, `ui_data.py`, `backend_client.py`, `.streamlit/` | Web UI |
| `ai_assistant.py` | Optional AI summary / chat with the user's own key: Google Gemini, Anthropic, OpenAI, OpenRouter, Groq, Mistral, DeepSeek |
| `update_database/` | Ingestion scripts (download + embed on a GPU machine) |
| `tools/rechunk_parquet.py` | Rewrite metadata parquet with small row groups |
| `eval/` | Offline ranking evaluation |
| `mss_env.py` | Machine-specific settings (environment / `~/.config/mss/paths.env`) |
| `docs/database_expansion.md` | Which databases to add next, and how |
