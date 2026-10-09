"""
In-process query embedder (BAAI/bge-small-en-v1.5), loaded once by the backend.

encode_query(text) -> (packed, float_vec):
  packed    — (1, 48) uint8, binary-quantized query for FAISS Hamming search
  float_vec — (384,) float32, normalized query for rescoring candidates
"""
import logging
import os
import threading

import numpy as np

LOGGER = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MSS_MODEL_ID", "BAAI/bge-small-en-v1.5")
# BGE query instruction — must match how documents were indexed (see eval/README.md).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_SEQ_LENGTH = 512
# Keep encoding light: FAISS searches need the cores more than one short query does.
TORCH_THREADS = int(os.environ.get("MSS_TORCH_THREADS", "1"))

_model = None
_lock = threading.Lock()


class EmbeddingError(RuntimeError):
    """Raised when the query cannot be embedded (model missing or failed)."""


def load_model():
    """Loads the model once; safe to call from several threads."""
    global _model
    with _lock:
        if _model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(TORCH_THREADS)
            LOGGER.info(f"Loading {MODEL_ID} (CPU, {TORCH_THREADS} thread(s))…")
            model = SentenceTransformer(MODEL_ID, device="cpu")
            model.max_seq_length = MAX_SEQ_LENGTH
            _model = model
            LOGGER.info("Embedding model ready.")
    return _model


def is_loaded() -> bool:
    return _model is not None


def encode_query(text: str):
    text = (text or "").strip()
    if not text:
        raise EmbeddingError("Empty query.")
    try:
        model = load_model()
        emb = model.encode(
            [QUERY_PREFIX + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        LOGGER.exception("Query embedding failed")
        raise EmbeddingError("Embedding model unavailable.") from exc
    float_vec = np.asarray(emb[0], dtype=np.float32)
    packed = np.packbits(float_vec > 0)[np.newaxis, :]
    return packed, float_vec
