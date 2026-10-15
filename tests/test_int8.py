"""int8 rescoring (int8_codes.py, search_logic._rescore_head_int8) and the int8 backfill."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "update_database"))

import int8_codes  # noqa: E402
import search_logic  # noqa: E402

from conftest import EMB_BYTES, build_corpus  # noqa: E402


def _floats(n, seed=0):
    """Unit vectors shaped like bge-small's (per-dimension ranges of the int8 calibration)."""
    lo, step = int8_codes.load_calibration()
    hi = lo + 255 * step
    f = (lo + hi) / 2 + np.random.default_rng(seed).uniform(-0.3, 0.3, (n, EMB_BYTES * 8)) * (hi - lo)
    return (f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def int8_corpus(tmp_path):
    """The synthetic corpus plus int8 files that match its binary codes."""
    config, _ = build_corpus(tmp_path)
    edir = config["embeddings_directory"]
    names = sorted(os.listdir(edir))
    floats = _floats(100 * len(names))
    os.makedirs(int8_codes.int8_dir(edir))
    for i, name in enumerate(names):
        np.save(os.path.join(edir, name), int8_codes.binary(floats[100 * i:100 * (i + 1)]))
        np.save(os.path.join(int8_codes.int8_dir(edir), name), int8_codes.quantize(floats[100 * i:100 * (i + 1)]))
    return config, int8_codes.binary(floats), floats


def _query(seed):
    q = np.random.default_rng(seed).standard_normal(EMB_BYTES * 8).astype(np.float32)
    q /= np.linalg.norm(q)
    return np.packbits(q > 0)[None, :], q


def test_quantize_round_trip_keeps_cosines():
    f = _floats(200, seed=1)
    q = f[0]
    approx = int8_codes.cosine(q, int8_codes.quantize(f))
    assert approx == pytest.approx(f @ q, abs=0.002)


@pytest.mark.parametrize("seed", range(3))
def test_int8_rescoring_returns_true_int8_top_k(int8_corpus, seed):
    config, all_emb, floats = int8_corpus
    packed, q = _query(300 + seed)
    df = search_logic.combined_search_orchestrator(packed, [config, {}, {}, {}], top_k=10, query_float=q)

    exact = (1 + int8_codes.cosine(q, int8_codes.quantize(floats))) / 2
    assert sorted(df["score"], reverse=True) == pytest.approx(sorted(exact, reverse=True)[:10], abs=1e-5)
    assert df["score"].is_monotonic_decreasing


def test_int8_row_of_another_document_is_ignored(int8_corpus):
    config, all_emb, floats = int8_corpus
    packed, q = _query(400)
    # the true best document's int8 row now describes a different document
    edir = config["embeddings_directory"]
    exact = int8_codes.cosine(q, int8_codes.quantize(floats))
    best = int(np.argmax(exact))
    name = sorted(os.listdir(edir))[best // 100]
    path = os.path.join(int8_codes.int8_dir(edir), name)
    rows = np.load(path)
    rows[best % 100] = int8_codes.quantize(_floats(1, seed=9))[0]
    np.save(path, rows)

    searcher = search_logic.get_or_create_searcher(config)
    _, ok = searcher.checked_int8_rows([best, (best + 1) % len(floats)])
    assert ok.tolist() == [False, True]
    df = search_logic.combined_search_orchestrator(packed, [config, {}, {}, {}], top_k=10, query_float=q)
    assert len(df) == 10       # still served, with its binary score mapped onto the int8 scale


def test_missing_int8_files_fall_back_to_binary_rescoring(corpus):
    config, all_emb = corpus
    packed, q = _query(500)
    df = search_logic.combined_search_orchestrator(packed, [config, {}, {}, {}], top_k=10, query_float=q)
    expected = sorted(search_logic.rescore_with_float_query(q, all_emb).tolist(), reverse=True)[:10]
    assert sorted(df["score"], reverse=True) == pytest.approx(expected, abs=1e-6)


def test_int8_rescoring_with_date_filter(int8_corpus):
    config, all_emb, floats = int8_corpus
    packed, q = _query(600)
    df = search_logic.combined_search_orchestrator(packed, [config, {}, {}, {}], top_k=10, query_float=q,
                                                   start_date="2013-01-01")
    dates = pd.to_datetime(df["date"])
    assert len(df) == 10 and (dates >= "2013-01-01").all()
    # rows 300+ are the files dated 2013 and later
    exact = (1 + int8_codes.cosine(q, int8_codes.quantize(floats[300:]))) / 2
    assert sorted(df["score"], reverse=True) == pytest.approx(sorted(exact, reverse=True)[:10], abs=1e-5)


def test_head_rescoring_keeps_tail_below_head(monkeypatch, int8_corpus):
    config, all_emb, floats = int8_corpus
    monkeypatch.setattr(search_logic, "INT8_RESCORE_DEPTH", 50)
    packed, q = _query(700)
    searcher = search_logic.get_or_create_searcher(config)
    cands = [{"corpus_id": i, "score": float(s), "source": "PubMed"}
             for i, s in enumerate(search_logic.rescore_with_float_query(q, all_emb))]
    cands.sort(key=lambda c: c["score"], reverse=True)
    out = search_logic._rescore_head_int8(cands, 50, {"PubMed": searcher}, q)
    scores = [c["score"] for c in out]
    assert scores[:50] == sorted(scores[:50], reverse=True)
    assert max(scores[50:]) <= min(scores[:50])
    assert [c["corpus_id"] for c in out[50:]] == [c["corpus_id"] for c in cands[50:]]


def test_backfill_keeps_matching_rows_and_redoes_changed_ones(tmp_path, monkeypatch):
    import int8_backfill
    floats = _floats(6, seed=2)
    codes = int8_codes.binary(floats)
    npy = tmp_path / "x_embed" / "a.npy"
    npy.parent.mkdir()
    np.save(npy, codes)
    pd.DataFrame({"title": list("abcdef"), "abstract": list("uvwxyz")}).to_parquet(tmp_path / "a.parquet")
    old = int8_codes.quantize(floats[:4])
    old[1] = int8_codes.quantize(_floats(1, seed=9))[0]   # row 1 was replaced since
    os.makedirs(int8_codes.int8_dir(str(npy.parent)))
    np.save(int8_codes.int8_path(str(npy)), old)

    p = int8_backfill.plan("BioRxiv", str(npy), str(tmp_path / "a.parquet"))
    assert p["redo"].tolist() == [False, True, False, False, True, True]
    assert p["texts"] == ["b. v", "e. y", "f. z"]
    assert (p["out"][[0, 2, 3]] == old[[0, 2, 3]]).all()
    assert int8_backfill.plan("PubMed", str(npy), str(tmp_path / "missing.parquet"))["error"]
