# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

PubMed Manuscript Semantic Search (MSS) — a biomedical literature search engine over PubMed, BioRxiv, MedRxiv, and arXiv. Queries are encoded into binary embeddings and searched against FAISS binary indexes built from chunked memory-mapped .npy files. Results are displayed in a Streamlit UI with optional Google Gemini AI summaries and chat.

## Running the services

There are three independent processes; all three must be running for the Streamlit UI to work:

```bash
# 1. Embedding server (port 8000) — must start first
python model_api.py

# 2. Streamlit UI
streamlit run pbmss_app.py
# or with a custom config:
streamlit run pbmss_app.py -- --config ./config_mss.yaml

# 3. REST search API (port 8080) — optional, for programmatic access
uvicorn search_api:app --host 0.0.0.0 --port 8080
```

## Architecture

### Three-process design
- **`model_api.py`** — FastAPI server (port 8000). Loads `BAAI/bge-small-en-v1.5` via SentenceTransformers, encodes a query string to a binary-quantized uint8 numpy array (384 float dims → 48 bytes).
- **`pbmss_app.py`** — Streamlit frontend. Calls the embedding server via `api_handler.py`, runs `search_logic.combined_search_orchestrator`, and renders results.
- **`search_api.py`** — FastAPI REST API (port 8080). Wraps the same search logic for programmatic access; authenticates via API keys in `api_keys.txt`.

### Search pipeline (`search_logic.py`)
1. `api_handler.get_query_embedding_packed()` → POST to `:8000/encode` → uint8 numpy array
2. `combined_search_orchestrator()` fans out across all four sources in parallel
3. Each source uses a `ChunkedSearcher` (cached at module level) which manages:
   - Chunked memory-mapped `.npy` files containing binary embeddings
   - FAISS `IndexBinaryFlat` indexes (also cached at module level)
   - Mapping from FAISS hit IDs back to source parquet rows via sorted interval tables
4. `data_handler.fetch_specific_rows()` reads metadata (title, abstract, DOI, date, authors) from parquet files using PyArrow

### Configuration (`config_mss.yaml`)
Four stanzas — `pubmed_config`, `biorxiv_config`, `medrxiv_config`, `arxiv_config` — each specifying:
- `embeddings_directory` — raw `.npy` embedding files
- `chunk_dir` — chunked memory-mapped files (built automatically on first run)
- `metadata_path` — JSON file with chunk layout and `total_rows`
- `data_folder` — parquet files with paper metadata

`config_loader.py` loads this YAML and is decorated with `@st.cache_data` so it's read once per Streamlit session.

### Module-level caches
`search_logic.py` keeps two in-process caches that survive Streamlit reruns:
- `_INDEX_CACHE`: `chunk_path → faiss.IndexBinaryFlat` (built on first query, warmed up in background thread at startup)
- `_SEARCHER_CACHE`: `chunk_dir → ChunkedSearcher`

Both caches are invalidated when `trigger_database_updates()` detects new embedding files.

### AI features (`gemini_handler.py`)
Uses `google.genai` with `gemini-3-flash-preview` (v1alpha API). Activated in the UI via a toggle + user-supplied Google AI Studio API key. Provides: search result summarization, suggested questions, and a chat interface over the returned abstracts.

### Database update scripts (`update_database/`)
Standalone scripts for ingesting new data:
- `pubmed_download_parquet.py` — downloads PubMed XML and converts to parquet
- `pubmed_embed_bge.py` — generates binary embeddings for PubMed
- `biorxiv_medarxiv_update_bge.py` — updates BioRxiv/MedRxiv embeddings
- `arxiv_download_embed_update.py` — updates arXiv embeddings

### Parametric UMAP (`umap_train.py`, `umap_generate.py`, `umap_final.py`)
PyTorch Lightning–based parametric UMAP for 2D visualization of the embedding space. Uses the local `umap_pytorch/` package. Separate from the live search pipeline — run offline to produce visualization assets.

## Key design constraints

- Embeddings are **binary-quantized** (384 float → 48 uint8 bytes via `np.packbits`). The BGE query prefix `"Represent this sentence for searching relevant passages: "` must be prepended before encoding; this is done server-side in `model_api.py`.
- FAISS indexes are `IndexBinaryFlat` (Hamming distance), not float L2/IP.
- Citation counts come from the Crossref API (`crossref.restful`) and are fetched lazily after initial results render to avoid blocking the UI.
- Session tracking uses a local SQLite file (`sessions_history.db`).
- The REST API (`search_api.py`) reads valid keys from `api_keys.txt` (one key per line); `#` lines are comments.
