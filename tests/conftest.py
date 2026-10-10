"""Shared fixtures: a small synthetic corpus (no real data or model server needed)."""
import os
import sys

import hashlib

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMB_BYTES = 48
N_FILES = 6
ROWS_PER_FILE = 100


@pytest.fixture
def corpus(tmp_path):
    """Six source files of 100 random binary embeddings each, with one parquet per file."""
    return build_corpus(tmp_path)


@pytest.fixture(scope="module")
def corpus_module(tmp_path_factory):
    return build_corpus(tmp_path_factory.mktemp("corpus"))


def build_corpus(tmp_path):
    rng = np.random.default_rng(0)
    emb_dir = tmp_path / "embed"
    data_dir = tmp_path / "data"
    emb_dir.mkdir()
    data_dir.mkdir()

    all_emb = []
    for f in range(N_FILES):
        emb = rng.integers(0, 256, size=(ROWS_PER_FILE, EMB_BYTES), dtype=np.uint8)
        np.save(emb_dir / f"part_{f:02d}.npy", emb)
        pd.DataFrame({
            "title": [f"Paper {f}-{i}" for i in range(ROWS_PER_FILE)],
            "abstract": ["x" * 100] * ROWS_PER_FILE,
            "doi": [f"10.1234/test.{f}.{i}" for i in range(ROWS_PER_FILE)],
            "date": [f"20{10 + f}-01-{1 + i % 28:02d}" for i in range(ROWS_PER_FILE)],
            "authors": ["A. Author"] * ROWS_PER_FILE,
            "journal": ["J Test"] * ROWS_PER_FILE,
        }).to_parquet(data_dir / f"part_{f:02d}.parquet")
        all_emb.append(emb)

    config = {
        "embeddings_directory": str(emb_dir),
        "npy_files_pattern": "*.npy",
        "chunk_dir": str(tmp_path / "chunks") + "/",
        "metadata_path": str(tmp_path / "meta.json"),
        "data_folder": str(data_dir),
        # Small chunks so the corpus spans several chunk files.
        "chunk_size_bytes": EMB_BYTES * 250,
    }
    return config, np.concatenate(all_emb)


PUBLIC, INTERNAL = {"X-API-Key": "pub-key"}, {"X-API-Key": "int-key"}


def fake_encode(text):
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    vec = np.random.default_rng(seed).standard_normal(EMB_BYTES * 8).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return np.packbits(vec > 0)[None, :], vec


@pytest.fixture(scope="session")
def backend_corpus(tmp_path_factory):
    return build_corpus(tmp_path_factory.mktemp("backend_corpus"))


@pytest.fixture(scope="session")
def backend(backend_corpus, tmp_path_factory):
    """The backend app with the synthetic corpus as PubMed; one per test session (MCP can start once)."""
    import access
    import embedder
    import search_api

    config, _ = backend_corpus
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
