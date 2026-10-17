"""Precise ranking (reranker.py): ordering logic with a stub cross-encoder."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_reranker_puts_best_blended_first():
    import pandas as pd

    import reranker
    df = pd.DataFrame({"title": [f"t{i}" for i in range(25)], "abstract": ["a"] * 25,
                       "score": [1.0 - 0.01 * i for i in range(25)]})
    # the cross-encoder strongly prefers results 12 and 19 (both inside the top 20)
    rr = np.zeros(20)
    rr[[12, 19]] = 50.0
    out = reranker.rerank(df, "q", scorer=lambda q, docs: rr[:len(docs)])
    titles = out["title"].tolist()
    assert set(titles[:2]) == {"t12", "t19"}
    assert len(out) == 25 and titles[20:] == [f"t{i}" for i in range(20, 25)]   # beyond 20: unchanged
    assert reranker.rerank(df, "q", scorer=lambda q, d: 1 / 0)["title"].tolist() == df["title"].tolist()


def test_idle_model_is_unloaded(monkeypatch):
    import time

    import reranker
    monkeypatch.setattr(reranker, "_model", ("tok", "sess", set()))
    monkeypatch.setattr(reranker, "_last_used", time.time())
    assert not reranker.unload_if_idle(60)
    monkeypatch.setattr(reranker, "_last_used", time.time() - 120)
    assert reranker.unload_if_idle(60) and reranker._model is None
