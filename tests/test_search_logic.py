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

from conftest import EMB_BYTES, N_FILES, ROWS_PER_FILE  # noqa: E402


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


@pytest.mark.parametrize("seed", range(3))
def test_float_rescoring_returns_true_float_top_k(corpus, seed):
    config, all_emb = corpus
    q_float = np.random.default_rng(200 + seed).standard_normal(EMB_BYTES * 8).astype(np.float32)
    q_float /= np.linalg.norm(q_float)
    query = np.packbits(q_float > 0)[None, :]

    df = search_logic.combined_search_orchestrator(
        query, [config, {}, {}, {}], top_k=10, query_float=q_float
    )

    expected = sorted(search_logic.rescore_with_float_query(q_float, all_emb).tolist(), reverse=True)[:10]
    assert sorted(df["score"].tolist(), reverse=True) == pytest.approx(expected, abs=1e-6)
    # and it is a different ranking from plain Hamming for at least part of the list
    hamming_top = _true_top_scores(all_emb, query, 10)
    assert hamming_top != pytest.approx(expected)


def test_rescore_matches_plain_float_dot_product():
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 256, size=(500, EMB_BYTES), dtype=np.uint8)
    q = rng.standard_normal(EMB_BYTES * 8).astype(np.float32)
    q /= np.linalg.norm(q)
    signs = np.unpackbits(codes, axis=1).astype(np.float64) * 2 - 1
    expected = (1 + signs @ q / np.sqrt(EMB_BYTES * 8)) / 2
    assert search_logic.rescore_with_float_query(q, codes) == pytest.approx(expected, abs=1e-5)


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


# ---------------------------------------------------------------------------
# Dedup / preprint ↔ published merge
# ---------------------------------------------------------------------------

def _row(doi, score, source="PubMed", published_doi=None, title=None):
    return {"doi": doi, "score": score, "source": source,
            "published_doi": published_doi, "title": title or f"title {doi}"}


def _dedup(rows):
    kept, seen = [], {}
    for r in rows:
        search_logic._add_or_merge(kept, seen, r)
    return kept


def test_preprint_ranked_first_is_replaced_by_published_version():
    kept = _dedup([
        _row("10.1101/2020.01.01.1", 0.90, "BioRxiv", published_doi="10.1038/abc"),
        _row("10.5555/other", 0.85),
        _row("10.1038/ABC", 0.80),
    ])
    assert [r["doi"] for r in kept] == ["10.1038/ABC", "10.5555/other"]
    assert kept[0]["score"] == 0.90
    assert kept[0]["preprint_doi"] == "10.1101/2020.01.01.1"


def test_preprint_after_published_version_is_folded_in():
    kept = _dedup([
        _row("10.1038/abc", 0.90),
        _row("10.1101/2020.01.01.1", 0.80, "BioRxiv", published_doi="10.1038/abc"),
    ])
    assert len(kept) == 1
    assert kept[0]["preprint_doi"] == "10.1101/2020.01.01.1"


def test_same_doi_and_doi_less_title_duplicates_are_dropped():
    kept = _dedup([
        _row("10.1/a", 0.9, title="Same Title"),
        _row("10.1/A", 0.8),                           # same DOI, different case
        _row("", 0.7, title="same title"),            # no DOI, title already seen
        _row("10.1/b", 0.6, title="Same Title"),      # has its own DOI → kept
    ])
    assert [r["doi"] for r in kept] == ["10.1/a", "10.1/b"]


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------

def test_repeat_search_is_served_from_cache(corpus, monkeypatch):
    config, _ = corpus
    search_logic.get_or_create_searcher(config)   # build chunks first (bumps the generation)
    query = np.full((1, EMB_BYTES), 7, dtype=np.uint8)
    first = search_logic.combined_search_orchestrator(query, [config, {}, {}, {}], top_k=10)

    calls = []
    monkeypatch.setattr(search_logic, "_run_search", lambda *a: calls.append(a))
    second = search_logic.combined_search_orchestrator(query, [config, {}, {}, {}], top_k=10)

    assert calls == []
    pd.testing.assert_frame_equal(first, second)
    second.loc[0, "title"] = "mutated"         # callers get a copy
    third = search_logic.combined_search_orchestrator(query, [config, {}, {}, {}], top_k=10)
    assert third.loc[0, "title"] != "mutated"


def test_index_change_invalidates_cache(corpus):
    config, _ = corpus
    search_logic.get_or_create_searcher(config)
    query = np.full((1, EMB_BYTES), 9, dtype=np.uint8)
    search_logic.combined_search_orchestrator(query, [config, {}, {}, {}], top_k=10)
    assert search_logic._RESULT_CACHE

    search_logic._clear_index_cache_for_dir(config["chunk_dir"])

    assert not search_logic._RESULT_CACHE
