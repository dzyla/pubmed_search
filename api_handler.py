import requests
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

MODEL_SERVER_URL = "http://localhost:8000/encode"


class EmbeddingError(RuntimeError):
    """Raised when the embedding server cannot be reached or returns an error."""


def get_query_embedding_packed(query: str, server_url: str = MODEL_SERVER_URL) -> np.ndarray:
    """
    Returns a (1, D_bytes) uint8 numpy array for FAISS binary search.
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

    packed_list = response.json().get("embedding", [])
    if not packed_list:
        raise EmbeddingError("Received empty embedding from server.")

    arr = np.array(packed_list, dtype=np.uint8)
    return arr[np.newaxis, :] if arr.ndim == 1 else arr


