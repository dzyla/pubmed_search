"""
Tests for search_logic on a small synthetic corpus (no real data or model server needed).

Run:  python -m pytest tests/
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import search_logic  # noqa: E402

EMB_BYTES = 48
N_FILES = 6
ROWS_PER_FILE = 100


@pytest.fixture
def corpus(tmp_path):
    """Six source files of 100 random binary embeddings each, with one parquet per file."""
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


def _true_top_scores(all_emb, query, k):
    dist = np.unpackbits(all_emb ^ query, axis=1).sum(axis=1)
    return sorted((1.0 - dist / (EMB_BYTES * 8)).tolist(), reverse=True)[:k]


@pytest.mark.parametrize("seed", range(5))
def test_returns_true_top_k(corpus, seed):
    config, all_emb = corpus
    query = np.random.default_rng(100 + seed).integers(0, 256, size=(1, EMB_BYTES), dtype=np.uint8)

    df = search_logic.combined_search_orchestrator(query, [config, {}, {}, {}], top_k=10)

    assert len(df) == 10
    got = sorted(df["score"].tolist(), reverse=True)
    assert got == pytest.approx(_true_top_scores(all_emb, query, 10))


def test_start_date_alone_is_applied(corpus):
    config, _ = corpus
    query = np.zeros((1, EMB_BYTES), dtype=np.uint8)

    df = search_logic.combined_search_orchestrator(
        query, [config, {}, {}, {}], top_k=10, start_date="2015-01-01"
    )

    assert len(df) == 10
    assert (pd.to_datetime(df["date"]) >= "2015-01-01").all()


def test_end_date_alone_is_applied(corpus):
    config, _ = corpus
    query = np.zeros((1, EMB_BYTES), dtype=np.uint8)

    df = search_logic.combined_search_orchestrator(
        query, [config, {}, {}, {}], top_k=10, end_date="2010-12-31"
    )

    assert len(df) == 10
    assert (pd.to_datetime(df["date"]) <= "2010-12-31").all()


def test_unreadable_metadata_on_reload_keeps_current_index(corpus):
    """A metadata file caught mid-write must not wipe the searcher's state."""
    config, _ = corpus
    searcher = search_logic.ChunkedSearcher(config)
    chunks_before = searcher.metadata["chunks"]

    with open(config["metadata_path"], "w") as f:
        f.write('{"total_rows": 6')          # truncated JSON
    os.utime(config["metadata_path"], (1e10, 1e10))  # look externally modified

    searcher._reload_metadata_if_externally_changed()

    assert searcher.metadata["chunks"] == chunks_before


def test_unreadable_metadata_at_startup_does_not_delete_chunks(corpus, monkeypatch):
    """A transient read failure must not trigger a full regeneration."""
    config, _ = corpus
    search_logic.ChunkedSearcher(config)  # builds chunks + metadata
    good = open(config["metadata_path"]).read()

    calls = {"n": 0}
    real_load = json.load

    def flaky_load(f):
        calls["n"] += 1
        if calls["n"] == 1:
            raise json.JSONDecodeError("truncated", "", 0)
        return real_load(f)

    monkeypatch.setattr(search_logic.json, "load", flaky_load)
    monkeypatch.setattr(search_logic.time, "sleep", lambda s: None)
    regen = []
    monkeypatch.setattr(search_logic.ChunkedSearcher, "_full_regeneration", lambda self: regen.append(1))

    search_logic.ChunkedSearcher(config)

    assert regen == []
    assert open(config["metadata_path"]).read() == good


def test_metadata_written_atomically(corpus):
    config, _ = corpus
    search_logic.ChunkedSearcher(config)
    meta_dir = os.path.dirname(config["metadata_path"])
    leftovers = [p for p in os.listdir(meta_dir) if ".tmp" in p]
    assert leftovers == []
    with open(config["metadata_path"]) as f:
        assert json.load(f)["total_rows"] == N_FILES * ROWS_PER_FILE
