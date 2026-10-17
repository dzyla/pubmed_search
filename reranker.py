"""
Optional "precise ranking": re-orders the first results with a biomedical cross-encoder.

MedCPT cross-encoder (ncbi/MedCPT-Cross-Encoder), int8 ONNX (per-channel, MatMul only;
built by tools/export_reranker.py), reading the query with each result's
"title. abstract" cut to 64 tokens. The best CANDIDATES results are re-scored and
the best SHOWN of them, by 0.5 · z(cross-encoder) + 0.5 · z(search score), are
put first; everything else keeps its order.

Benchmark (2026-10-16, 100 graded topical queries, eval/README.md): nDCG@10
0.667 → 0.702 (+0.035, 95% CI +0.022..+0.049; 65 better / 31 worse); finding a
known paper is unchanged. Costs ~0.39 s per search on 4 CPU threads, so it
is off unless the user turns it on.
"""
import gc
import logging
import os
import threading
import time

import numpy as np

import mss_env

LOGGER = logging.getLogger(__name__)

MODEL_DIR = mss_env.setting("MSS_RERANKER_DIR", os.path.join(mss_env.REPO_ROOT, "models", "medcpt-ce-int8"))
CANDIDATES = 20
SHOWN = 10
MAX_TOKENS = 64
ALPHA = 0.5
THREADS = int(os.environ.get("MSS_RERANKER_THREADS", "4"))

_lock = threading.Lock()
_model = None          # (tokenizer, session, input names), or False when unavailable
_last_used = 0.0
IDLE_UNLOAD_S = 30 * 60   # the model takes ~350 MB; precise ranking is opt-in and often idle


def _load():
    global _model
    with _lock:
        if _model is None:
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(MODEL_DIR)
                so = ort.SessionOptions()
                so.intra_op_num_threads, so.inter_op_num_threads = THREADS, 1
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"), so,
                                            providers=["CPUExecutionProvider"])
                _model = (tok, sess, {i.name for i in sess.get_inputs()})
                LOGGER.info(f"Reranker loaded from {MODEL_DIR}")
            except Exception as exc:
                LOGGER.warning(f"Precise ranking unavailable ({type(exc).__name__}: {exc})")
                _model = False
    return _model


def available() -> bool:
    return bool(_load())


def doc_text(title, abstract) -> str:
    t = "" if title is None or (isinstance(title, float) and np.isnan(title)) else str(title).strip()
    a = "" if abstract is None or (isinstance(abstract, float) and np.isnan(abstract)) else str(abstract).strip()
    if t and a:
        return f"{t} {a}" if t.endswith((".", "?", "!")) else f"{t}. {a}"
    return t or a


def unload_if_idle(idle_s: float = IDLE_UNLOAD_S) -> bool:
    """Frees the model when no precise search ran for idle_s seconds (called by the hourly check)."""
    global _model
    with _lock:
        if not _model or time.time() - _last_used < idle_s:
            return False
        _model = None
    gc.collect()
    LOGGER.info("Reranker unloaded (idle)")
    return True


def score(query: str, docs: list) -> np.ndarray:
    global _last_used
    _last_used = time.time()
    tok, sess, names = _load()
    enc = tok([query] * len(docs), docs, truncation="only_second", max_length=MAX_TOKENS, padding=True,
              return_tensors="np")
    return sess.run(None, {k: v.astype(np.int64) for k, v in enc.items() if k in names})[0][:, 0]


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def rerank(df, query: str, scorer=None):
    """
    Results DataFrame (best first, with title/abstract/score) -> the same rows with the
    best SHOWN of the first CANDIDATES re-ordered first. Returns df unchanged on any error.
    """
    n = min(CANDIDATES, len(df))
    if n < 2:
        return df
    try:
        head = df.iloc[:n]
        docs = [doc_text(r.get("title"), r.get("abstract")) for r in head.to_dict("records")]
        rr = np.asarray((scorer or score)(query, docs), dtype=np.float64)
        blended = ALPHA * _z(rr) + (1 - ALPHA) * _z(head["score"].to_numpy(dtype=np.float64))
        order = np.argsort(-blended, kind="stable")
        first = [int(i) for i in order[:SHOWN]]      # best blended first
        chosen = set(first)
        rest = [i for i in range(n) if i not in chosen]   # the other candidates keep their order
        return df.iloc[first + rest + list(range(n, len(df)))].reset_index(drop=True)
    except Exception:
        LOGGER.exception("Reranking failed; returning the search order")
        return df
