# Deployment Guide

This document explains how to deploy the Manuscript Semantic Search (MSS) stack and prepare the searchable data. It covers single-VM deployments, Docker Compose, prewarming the index, and operational considerations. No cloud-specific lock-in; the same steps work on AWS, GCP, Azure, or any standard Linux host.

## Components

- Model API: `model_api.py` (FastAPI on port 8000) encodes queries to embeddings.
- Search App: `pbmss_app.py` (Streamlit, default port 8501) loads chunked corpora and performs semantic search.
- Data: Parquet metadata and per-file `.npy` embeddings per source; chunked memmaps + metadata built on first use (or via a prewarm script).

## Prerequisites

- Python 3.10+
- Adequate disk: PubMed is very large. Start with bioRxiv/medRxiv to validate.
- Optional GPU: only required for high-throughput embedding generation; the app runs fine on CPU.

## Configuration

Server-side salt for deterministic author ordering:

- Preferred: environment variable (don’t commit secrets)

  - macOS/Linux (zsh):

    ```zsh
    export MSS_AUTHOR_SALT="<YOUR_SECRET_SALT>"
    ```

- Alternative: Streamlit secrets (`.streamlit/secrets.toml`, not in git)

  ```toml
  author_salt = "<YOUR_SECRET_SALT>"
  ```

Data/configs (edit `config_mss.yaml`): for each source specify:

- `data_folder`: Parquet files
- `embeddings_directory`: `.npy` embedding files
- `npy_files_pattern`: e.g., `*.npy`
- `chunk_dir`: directory to store chunked memmaps
- `metadata_path`: JSON metadata file describing chunks

## Initialize the database

You need Parquet + `.npy` embeddings for each source. Options:

- bioRxiv/medRxiv: run `biorxiv_update_api.py` or `update_arxiv.py`-style scripts to fetch, convert to Parquet, and produce embeddings.
- PubMed: run `pubmed_update_api.py` to download XML, convert to Parquet, and organize for embedding generation.
- arXiv: use `arxiv_download_process.py` to prepare Parquet and embeddings.

You can start small: just bioRxiv/medRxiv first, then add PubMed/arXiv later.

## Prewarm the chunked index (recommended)

On first search, the app builds chunked memmaps per source (fast on subsequent runs). To avoid doing this during first user request, use the included prewarm script:

```zsh
python scripts/prewarm_chunks.py --config config_mss.yaml
```

This scans each configured source and creates chunked memmaps + metadata if missing or out-of-date.

## Run locally (two processes)

- Terminal 1 (Model API):

  ```zsh
  source .venv/bin/activate  # or your env activation
  export MSS_AUTHOR_SALT="<YOUR_SECRET_SALT>"
  python model_api.py
  ```

- Terminal 2 (App):

  ```zsh
  source .venv/bin/activate
  export MSS_AUTHOR_SALT="<YOUR_SECRET_SALT>"
  streamlit run pbmss_app.py
  ```

## Docker Compose

A simple Compose setup is provided (`docker-compose.yml`) with two services: `model_api` and `app`. Mount your data volume so both can read the Parquet/.npy/chunk files.

```zsh
export MSS_AUTHOR_SALT="<YOUR_SECRET_SALT>"
docker compose up -d --build
```

- Model API: <http://localhost:8000>
- App: <http://localhost:8501>

## Reverse proxy & TLS

Put Nginx/Caddy/Traefik in front to serve HTTPS and route traffic:

- `/` → Streamlit app (8501)
- `/encode` → Model API (8000)

Lock down the model API to internal networks if desired.

## Systemd services (single VM)

Templates are provided in `ops/systemd/`:

- `mss-model.service`: runs `python model_api.py`
- `mss-app.service`: runs `streamlit run pbmss_app.py`

Copy to `/etc/systemd/system/`, adjust `User`, `WorkingDirectory`, and `Environment` (set `MSS_AUTHOR_SALT`), then:

```zsh
sudo systemctl daemon-reload
sudo systemctl enable mss-model mss-app
sudo systemctl start mss-model mss-app
```

## Operations tips

- Backups: keep Parquet/.npy and chunk metadata safe. Rebuilding chunks is cheaper than recomputing embeddings.
- Monitoring: add simple health endpoints or process supervision; Streamlit logs show search timing.
- Scaling: most load is on reading memmaps and computing query embeddings (cheap). Horizontal scale is usually unnecessary unless you expect heavy traffic.

---

## Appendix: Updating data regularly

- Schedule your update scripts (e.g., `biorxiv_update_api.py`, `pubmed_update_api.py`) via cron or a scheduler.
- After embedding generation, re-run `scripts/prewarm_chunks.py` to update chunked memmaps and metadata.
