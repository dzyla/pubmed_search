"""
int8 document vectors, used to re-rank the best binary-search candidates.

Each source keeps, next to its binary codes (<embeddings_directory>/<stem>.npy,
48 bytes/doc, searched in RAM), the same rows as int8 vectors in
<embeddings_directory without trailing '/'>_int8/<stem>.npy (384 bytes/doc, read
from disk only for the top candidates). Row i of one file is row i of the other.

Quantization is per dimension with one fixed calibration (int8_calibration.npy,
float32 (2, 384): lower bound and step), shared by every source and every future
update so scores stay comparable:

    doc ≈ (code + 128) * step + lo

and the float query's cosine with a document is computed without dequantizing:

    q · doc = codes · (q * step) + q · (lo + 128 * step)

Benchmark (2026-10-14, 100 topical queries, graded): binary search + this int8
rescoring nDCG@10 0.725 vs 0.661 for float-query-vs-bits rescoring (eval/README.md).
"""
import os

import numpy as np

CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "int8_calibration.npy")
DIM = 384

_calibration = None


def int8_dir(embeddings_directory: str) -> str:
    return embeddings_directory.rstrip("/") + "_int8"


def int8_path(binary_npy_path: str) -> str:
    """Path of the int8 file that belongs to a binary-code .npy file."""
    d, name = os.path.split(os.path.abspath(binary_npy_path))
    return os.path.join(int8_dir(d), name)


def load_calibration():
    global _calibration
    if _calibration is None:
        cal = np.load(CALIBRATION_PATH).astype(np.float32)
        if cal.shape != (2, DIM):
            raise ValueError(f"{CALIBRATION_PATH}: expected (2, {DIM}), got {cal.shape}")
        _calibration = cal
    return _calibration


def calibrate(sample: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    """Per-dimension range from a sample of normalized float embeddings."""
    lo = np.percentile(sample, lo_pct, axis=0)
    hi = np.percentile(sample, hi_pct, axis=0)
    return np.stack([lo, (hi - lo) / 255.0]).astype(np.float32)


def quantize(emb: np.ndarray, calibration=None) -> np.ndarray:
    lo, step = load_calibration() if calibration is None else calibration
    q = np.rint((np.asarray(emb, dtype=np.float32) - lo) / step) - 128
    return np.clip(q, -128, 127).astype(np.int8)


def binary(emb: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(emb) > 0, axis=1)


def query_weights(query_float, calibration=None):
    """(weights (384,), bias) such that cosine(query, doc) ≈ codes · weights + bias."""
    lo, step = load_calibration() if calibration is None else calibration
    q = np.asarray(query_float, dtype=np.float32).reshape(-1)
    return q * step, float(q @ (lo + 128.0 * step))


def cosine(query_float, codes: np.ndarray, calibration=None) -> np.ndarray:
    w, b = query_weights(query_float, calibration)
    return codes.astype(np.float32) @ w + b


def save_pair(binary_npy_path: str, emb: np.ndarray):
    """Writes a source's binary codes and int8 vectors for the same rows (atomically, int8 first)."""
    emb = np.asarray(emb, dtype=np.float32)
    ipath = int8_path(binary_npy_path)
    os.makedirs(os.path.dirname(ipath), exist_ok=True)
    for path, arr in ((ipath, quantize(emb).reshape(-1, DIM)), (binary_npy_path, binary(emb).reshape(-1, DIM // 8))):
        tmp = path + ".tmp"            # not *.npy: the indexer globs those
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, path)
