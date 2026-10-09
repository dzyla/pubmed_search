import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import filter_language  # noqa: E402
from textlang import english_score, is_english  # noqa: E402

EN = "We measured the effect of the drug on tumour growth in mice and found that it was reduced by half."
PT = "Este estudo analisa os caminhos discursivos da geometria no ensino fundamental e suas implicações."


def test_english_score():
    assert is_english(EN) and not is_english(PT)
    assert english_score("CRISPR") == 1.0          # too short to judge -> kept


def test_filter_pair_keeps_alignment(tmp_path):
    df = pd.DataFrame({"title": ["A", "B", "C"], "abstract": [EN, PT, EN]})
    bits = np.arange(3 * 48, dtype=np.uint8).reshape(3, 48)
    df.to_parquet(tmp_path / "c.parquet")
    np.save(tmp_path / "c.npy", bits)
    assert filter_language.filter_pair(str(tmp_path / "c.parquet"), str(tmp_path / "c.npy")) == 1
    out, kept = pd.read_parquet(tmp_path / "c.parquet"), np.load(tmp_path / "c.npy")
    assert out["title"].tolist() == ["A", "C"] and (kept == bits[[0, 2]]).all()
