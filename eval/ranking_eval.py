"""
Offline ranking evaluation for the binary-embedding search (read-only on data).

Queries: synthetic queries written for known PubMed papers
(pubmed_synthetic_queries_10k.parquet: title, abstract, synthetic_queries).
Distractors: real stored PubMed vectors (production .npy files), plus
"no-prefix" documents re-embedded the way bioRxiv/medRxiv/arXiv were.

Experiments
  1. Single source (PubMed-style docs): Hamming ranking vs. rescoring the top
     candidates with the float query, score = q_float · (2·bits − 1).
  2. Mixed doc styles: PubMed docs were embedded WITH the BGE query prefix,
     the other sources WITHOUT. Measures how this skews a merged ranking and
     whether a constant per-style score offset corrects it.

Usage:
  python eval/ranking_eval.py --data-dir <pubmed_semantic_search dir> \
      --queries <pubmed_synthetic_queries_10k.parquet> [--n-queries 2000]
"""
import argparse
import re
import time

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

PREFIX = "Represent this sentence for searching relevant passages: "
D_BITS = 384


def parse_first_query(text: str):
    m = re.search(r'^\s*1\.\s*(.+?)\s*$', str(text), flags=re.M)
    if not m:
        return None
    q = m.group(1).strip().strip('"“”[]').strip()
    return q if len(q) > 5 else None


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def embed(model, texts, batch_size=256):
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)


def to_bits(emb):
    return np.packbits(emb > 0, axis=1)


def hamming_scores(q_bits, doc_bits):
    """1 - hamming/384, same as production."""
    return 1.0 - np.unpackbits(doc_bits ^ q_bits, axis=1).sum(axis=1) / D_BITS


def float_scores(q_float, doc_bits):
    """Asymmetric score: float query · ±1 document bits."""
    signs = np.unpackbits(doc_bits, axis=1).astype(np.float32) * 2 - 1
    return signs @ q_float


def metrics(ranks, label):
    ranks = np.asarray(ranks, dtype=float)  # 1-based, inf = not retrieved
    r1 = np.mean(ranks <= 1)
    r10 = np.mean(ranks <= 10)
    r100 = np.mean(ranks <= 100)
    mrr = np.mean(np.where(ranks <= 10, 1.0 / ranks, 0.0))
    print(f"  {label:44s} R@1 {r1:.3f}  R@10 {r10:.3f}  R@100 {r100:.3f}  MRR@10 {mrr:.3f}")
    return {"label": label, "R@1": r1, "R@10": r10, "R@100": r100, "MRR@10": mrr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--n-pubmed-files", type=int, default=60)
    ap.add_argument("--n-noprefix-docs", type=int, default=150_000)
    ap.add_argument("--candidates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # --- Eval queries + their positive papers ---
    qdf = pd.read_parquet(args.queries)
    qdf["query"] = qdf["synthetic_queries"].map(parse_first_query)
    qdf = qdf[qdf["query"].notna() & (qdf["abstract"].fillna("").str.len() > 100)]
    qdf = qdf.sample(n=min(args.n_queries, len(qdf)), random_state=args.seed).reset_index(drop=True)
    eval_titles = set(qdf["title"].map(norm_title))
    print(f"{len(qdf)} eval queries")

    # --- Distractors A: real stored (prefixed) PubMed vectors ---
    import glob
    npys = sorted(glob.glob(f"{args.data_dir}/pubmed26_update_embed/pubmed26n*.npy"))
    picks = rng.choice(len(npys), size=args.n_pubmed_files + 6, replace=False)
    a_files, b_files = [npys[i] for i in picks[: args.n_pubmed_files]], [npys[i] for i in picks[args.n_pubmed_files:]]
    a_bits = []
    for f in a_files:
        bits = np.load(f, mmap_mode="r", allow_pickle=False)
        titles = pd.read_parquet(f.replace("pubmed26_update_embed", "pubmed26_parquet_files")
                                  .replace(".npy", ".parquet"), columns=["title"])["title"]
        keep = ~titles.map(norm_title).isin(eval_titles).to_numpy()[: len(bits)]
        a_bits.append(np.asarray(bits)[: len(keep)][keep])
    a_bits = np.concatenate(a_bits)
    print(f"A: {len(a_bits):,} stored prefixed PubMed vectors ({time.time() - t0:.0f}s)")

    # --- Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    # --- Distractors B: PubMed docs re-embedded WITHOUT prefix (bioRxiv style) ---
    b_texts = []
    for f in b_files:
        d = pd.read_parquet(f.replace("pubmed26_update_embed", "pubmed26_parquet_files")
                            .replace(".npy", ".parquet"), columns=["title", "abstract"])
        d = d[~d["title"].map(norm_title).isin(eval_titles)]
        b_texts += (d["title"].fillna("") + ". " + d["abstract"].fillna("")).tolist()
    b_texts = [b_texts[i] for i in rng.choice(len(b_texts), size=min(args.n_noprefix_docs, len(b_texts)), replace=False)]
    b_bits = to_bits(embed(model, b_texts))
    print(f"B: {len(b_bits):,} no-prefix vectors ({time.time() - t0:.0f}s)")

    # --- Positives (both styles) and queries ---
    pos_text = (qdf["title"].fillna("") + ". " + qdf["abstract"].fillna("")).tolist()
    pos_pref = to_bits(embed(model, [PREFIX + t for t in pos_text]))
    pos_nopref = to_bits(embed(model, pos_text))
    q_float = embed(model, [PREFIX + q for q in qdf["query"]])
    q_bits = to_bits(q_float)
    print(f"embedded positives + queries ({time.time() - t0:.0f}s)")

    n_q = len(qdf)
    K = args.candidates

    def run(corpus_bits, pos_bits, pos_offset_ids, style_of, offsets, label_prefix):
        """
        corpus_bits: distractor bits; positives appended after them (one per query).
        style_of: array, doc style per row (0 = prefixed, 1 = no-prefix).
        offsets: dict name -> {"hamming"|"float": {style: additive offset}}.
        """
        full = np.concatenate([corpus_bits, pos_bits])
        index = faiss.IndexBinaryFlat(D_BITS)
        index.add(full)
        _, cand = index.search(q_bits, K)
        styles = np.concatenate([style_of, pos_offset_ids])
        results = {}
        for name, off in offsets.items():
            ranks = {"hamming": [], "float": []}
            for qi in range(n_q):
                c = cand[qi][cand[qi] >= 0]
                target = len(corpus_bits) + qi
                for kind, s in (("hamming", hamming_scores(q_bits[qi:qi + 1], full[c])),
                                ("float", float_scores(q_float[qi], full[c]) / np.sqrt(D_BITS))):
                    adj = np.array([off[kind][st] for st in styles[c]])
                    order = c[np.argsort(-(s + adj), kind="stable")]
                    hit = np.flatnonzero(order == target)
                    ranks[kind].append(hit[0] + 1 if len(hit) else np.inf)
            for kind in ("hamming", "float"):
                results[(name, kind)] = metrics(ranks[kind], f"{label_prefix} {name} / {kind}")
        return results

    print(f"\n[1] Single source: positives prefixed among {len(a_bits):,} prefixed PubMed docs")
    run(a_bits, pos_pref, np.zeros(n_q, int), np.zeros(len(a_bits), int),
        {"no offset": {"hamming": {0: 0.0, 1: 0.0}, "float": {0: 0.0, 1: 0.0}}}, "")

    # Offset estimate: mean score gap between styles for the same doc and query
    # (prefixed − no-prefix), measured on the positives themselves.
    def gaps(score_fn):
        rel = np.mean([score_fn(i, pos_pref[i:i + 1])[0] - score_fn(i, pos_nopref[i:i + 1])[0]
                       for i in range(n_q)])
        return rel

    h_fn = lambda i, bits: hamming_scores(q_bits[i:i + 1], bits)                  # noqa: E731
    f_fn = lambda i, bits: float_scores(q_float[i], bits) / np.sqrt(D_BITS)      # noqa: E731
    gap_h, gap_f = gaps(h_fn), gaps(f_fn)
    rnd = rng.choice(len(b_texts), size=500, replace=False)
    pref_rnd = to_bits(embed(model, [PREFIX + b_texts[i] for i in rnd]))
    gap_h_rnd = np.mean([h_fn(i, pref_rnd[i:i + 1])[0] - h_fn(i, b_bits[rnd][i:i + 1])[0] for i in range(500)])
    gap_f_rnd = np.mean([f_fn(i, pref_rnd[i:i + 1])[0] - f_fn(i, b_bits[rnd][i:i + 1])[0] for i in range(500)])
    print(f"\nScore gap prefixed − no-prefix:  Hamming relevant {gap_h:+.4f} random {gap_h_rnd:+.4f} | "
          f"float relevant {gap_f:+.4f} random {gap_f_rnd:+.4f}")

    mixed = np.concatenate([a_bits, b_bits])
    style = np.concatenate([np.zeros(len(a_bits), int), np.ones(len(b_bits), int)])
    offsets = {
        "no offset": {"hamming": {0: 0.0, 1: 0.0}, "float": {0: 0.0, 1: 0.0}},
        "offset (relevant gap)": {"hamming": {0: 0.0, 1: gap_h}, "float": {0: 0.0, 1: gap_f}},
        "offset (random gap)": {"hamming": {0: 0.0, 1: gap_h_rnd}, "float": {0: 0.0, 1: gap_f_rnd}},
    }

    print("\n[2a] Mixed styles, positive is NO-PREFIX (like bioRxiv/medRxiv/arXiv)")
    run(mixed, pos_nopref, np.ones(n_q, int), style, offsets, "")
    print("\n[2b] Mixed styles, positive is PREFIXED (like PubMed)")
    run(mixed, pos_pref, np.zeros(n_q, int), style, offsets, "")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
