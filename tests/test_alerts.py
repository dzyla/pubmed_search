"""Saved-search alerts (alerts.py) and their API."""
import os
import sys
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import alerts  # noqa: E402
import int8_codes  # noqa: E402
import search_logic  # noqa: E402

from conftest import build_corpus  # noqa: E402
from test_int8 import _floats  # noqa: E402


def _add_file(config, name, floats, dates, titles):
    edir = config["embeddings_directory"]
    np.save(os.path.join(edir, f"{name}.npy"), int8_codes.binary(floats))
    np.save(os.path.join(int8_codes.int8_dir(edir), f"{name}.npy"), int8_codes.quantize(floats))
    pd.DataFrame({"title": titles, "abstract": ["x" * 100] * len(titles),
                  "doi": [f"10.9999/{name}.{i}" for i in range(len(titles))], "date": dates,
                  "authors": ["A. Author"] * len(titles), "journal": ["J New"] * len(titles)}
                 ).to_parquet(os.path.join(config["data_folder"], f"{name}.parquet"))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts, "DB_PATH", str(tmp_path / "alerts.db"))
    config, _ = build_corpus(tmp_path)
    edir = config["embeddings_directory"]
    os.makedirs(int8_codes.int8_dir(edir))
    names = sorted(os.listdir(edir))
    floats = _floats(100 * len(names))
    for i, name in enumerate(names):
        np.save(os.path.join(edir, name), int8_codes.binary(floats[100 * i:100 * (i + 1)]))
        np.save(os.path.join(int8_codes.int8_dir(edir), name), int8_codes.quantize(floats[100 * i:100 * (i + 1)]))
    searcher = search_logic.get_or_create_searcher(config)
    return config, searcher, floats


def test_new_recent_matches_reach_the_feed(setup):
    config, searcher, floats = setup
    assert alerts.run({"PubMed": searcher}) == {}                # first run: snapshot only
    q = _floats(1, seed=42)[0]
    cal = search_logic.alert_calibration(np.packbits(q > 0)[None, :], q, [config], rank=5)
    token = alerts.create("my topic", ["PubMed"], q, cal["threshold"], "focused")

    today = date.today()
    near = q + 0.02 * _floats(1, seed=7)[0]                     # a new paper very close to the query
    new = np.vstack([near / np.linalg.norm(near), _floats(3, seed=8)])
    _add_file(config, "part_99", new,
              [today.isoformat(), today.isoformat(), (today - timedelta(days=3000)).isoformat(), today.isoformat()],
              ["Close new paper", "Unrelated new paper", "Old record re-issued", "Another unrelated"])
    assert searcher.check_for_updates()
    checked = alerts.run({"PubMed": searcher})
    assert checked == {"PubMed": 3}                              # the old re-issued record is skipped
    got = alerts.items(token)
    assert [it["title"] for it in got] == ["Close new paper"]
    assert got[0]["doi"] == "10.9999/part_99.0" and got[0]["journal"] == "J New"
    assert alerts.run({"PubMed": searcher}) == {}                # nothing new the second time

    feed = alerts.rss(token, "https://x/feed.xml", "https://x")
    channel = ET.fromstring(feed).find("channel")
    assert channel.find("title").text == "my topic · Manuscript Search"
    assert [i.find("title").text for i in channel.findall("item")] == ["Close new paper"]


def test_bar_is_close_to_the_true_rank(setup, monkeypatch):
    # Regression: with rank == the int8 re-scoring depth, the bar was the lowest int8 score
    # among the binary top-N, which on the real corpus sat at rank 100k-1.4M instead of 2,000.
    config, searcher, floats = setup
    rank = 10
    monkeypatch.setattr(search_logic, "INT8_RESCORE_DEPTH", rank)
    q = _floats(1, seed=42)[0]
    cal = search_logic.alert_calibration(np.packbits(q > 0)[None, :], q, [config], rank=rank)
    exact = (1.0 + int8_codes.cosine(q, int8_codes.quantize(floats))) / 2.0
    true_rank = int((exact >= cal["threshold"] - 1e-6).sum())
    assert rank <= true_rank <= 2 * rank


def test_recalibration_raises_the_bar_and_prunes_items(setup):
    config, searcher, floats = setup
    q = floats[0]
    token = alerts.create("t", ["PubMed"], q, 0.5, "balanced")
    alerts.create("current", ["PubMed"], q, 0.7, "focused")
    with alerts._db() as conn:                                   # made before the calibration fix
        conn.execute("UPDATE alerts SET calibration=1 WHERE token=?", (token,))
    alerts.add_items(token, [{"title": "weak", "doi": "10.1/a", "score": 0.6},
                             {"title": "strong", "doi": "10.1/b", "score": 0.9}])
    stale = alerts.stale_calibrations()
    assert [s["token"] for s in stale] == [token] and stale[0]["rank"] == alerts.SENSITIVITY["balanced"]
    assert np.allclose(stale[0]["vec"], q) and stale[0]["sources"] == ["PubMed"]
    assert alerts.set_threshold(token, 0.8) == 1                 # one item below the new bar removed
    assert [it["title"] for it in alerts.items(token)] == ["strong"]
    assert alerts.get(token)["threshold"] == pytest.approx(0.8)
    assert alerts.stale_calibrations() == []


def test_alert_only_watches_its_databases(setup):
    config, searcher, floats = setup
    alerts.run({"PubMed": searcher})
    q = floats[0]
    token = alerts.create("t", ["arXiv"], q, 0.0, "broad")       # bar 0: everything would match
    _add_file(config, "part_98", floats[:2], [date.today().isoformat()] * 2, ["a", "b"])
    searcher.check_for_updates()
    alerts.run({"PubMed": searcher})
    assert alerts.items(token) == []

