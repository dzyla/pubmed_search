import numpy as np
import pytest

from conftest import EMB_BYTES


# ---------------------------------------------------------------------------

def test_embedder_packs_float_query(monkeypatch):
    import embedder

    class FakeModel:
        def encode(self, texts, **kwargs):
            assert texts[0].startswith(embedder.QUERY_PREFIX)
            v = np.linspace(-1, 1, EMB_BYTES * 8, dtype=np.float32)
            return (v / np.linalg.norm(v))[None, :]

    monkeypatch.setattr(embedder, "load_model", lambda: FakeModel())
    packed, vec = embedder.encode_query("query")
    assert packed.shape == (1, EMB_BYTES) and vec.shape == (EMB_BYTES * 8,)
    assert np.array_equal(np.unpackbits(packed[0]), (vec > 0).astype(np.uint8))
    with pytest.raises(embedder.EmbeddingError):
        embedder.encode_query("   ")


