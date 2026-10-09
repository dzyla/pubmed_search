"""Shared fixtures: a small synthetic corpus (no real data or model server needed)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

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
