import requests
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

MODEL_SERVER_URL = "http://localhost:8000/encode"


class EmbeddingError(RuntimeError):
    """Raised when the embedding server cannot be reached or returns an error."""


def get_query_embeddings(query: str, server_url: str = MODEL_SERVER_URL):
    """
    Returns (packed, float_vec):
      packed    — (1, D_bytes) uint8 array for FAISS binary search
      float_vec — (D_bits,) float32 normalized query for rescoring, or None if
                  the embedding server is an older version that does not send it
    Raises EmbeddingError on any failure — callers decide how to surface it.
    """
    try:
        response = requests.post(server_url, json={"text": query}, timeout=5)
    except requests.exceptions.ConnectionError:
        raise EmbeddingError(
            "Model API not available. Ensure the embedding server is running."
        )
    except Exception as exc:
        raise EmbeddingError(f"Unexpected error contacting embedding server: {exc}")

    if response.status_code != 200:
        raise EmbeddingError(
            f"Model API returned error {response.status_code}: {response.text}"
        )

    payload = response.json()
    packed_list = payload.get("embedding", [])
    if not packed_list:
        raise EmbeddingError("Received empty embedding from server.")

    arr = np.array(packed_list, dtype=np.uint8)
    packed = arr[np.newaxis, :] if arr.ndim == 1 else arr

    float_list = payload.get("embedding_float")
    float_vec = np.asarray(float_list, dtype=np.float32) if float_list else None
    if float_vec is not None and float_vec.shape != (packed.shape[1] * 8,):
        LOGGER.warning(f"Ignoring float embedding with unexpected shape {float_vec.shape}")
        float_vec = None
    return packed, float_vec


def get_query_embedding_packed(query: str, server_url: str = MODEL_SERVER_URL) -> np.ndarray:
    """Returns only the (1, D_bytes) uint8 array for FAISS binary search."""
    return get_query_embeddings(query, server_url)[0]


