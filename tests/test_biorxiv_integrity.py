import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sentence_transformers")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "update_database"))
import biorxiv_medarxiv_update_bge as u  # noqa: E402


class FakeModel:
    """Deterministic 384-d 'embedding' per text."""

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **_):
        seeds = [int(hashlib.md5(t.encode()).hexdigest()[:8], 16) for t in texts]
        return np.stack([np.random.default_rng(s).standard_normal(384) for s in seeds]).astype(np.float32)


def test_integrity_check_repairs_a_shifted_block(tmp_path):
    model = FakeModel()
    df = pd.DataFrame({"title": [f"paper {i}" for i in range(300)], "abstract": ["text"] * 300})
    good = u.generate_embeddings_batched(model, u.build_input_texts(df))
    shifted = good.copy()
    shifted[100:140] = good[105:145]                   # a block holding the codes of rows 5 further down
    meta, emb = str(tmp_path / "m.parquet"), str(tmp_path / "e.npy")
    df.to_parquet(meta)
    np.save(emb, shifted)
    u.check_database_integrity(meta, emb, model, sample=50, recent=200)   # newest 200 rows cover the block: deterministic
    assert (np.load(emb) == good).all()
    assert len(pd.read_parquet(meta)) == 300


def test_unreadable_downloads_are_counted_and_removed(tmp_path):
    (tmp_path / "json").mkdir()
    (tmp_path / "pq").mkdir()
    (tmp_path / "json" / "details_data_a.json").write_text('[{"title": "ok", "abstract": "x"}]')
    (tmp_path / "json" / "details_data_b.json").write_text('[{"title": "cut off by a full dis')
    files, unreadable = u.convert_json_to_parquet(str(tmp_path / "json"), str(tmp_path / "pq"))
    assert len(files) == 1 and unreadable == 1
    assert not (tmp_path / "json" / "details_data_b.json").exists()
