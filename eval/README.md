# Offline evaluation

Read-only scripts that measure search quality on real data. Run them on the lab
desktop (GPU, data on H:), never on the server.

## `check_doc_prefix.py` — how were stored vectors made?

Re-embeds a sample of rows with and without the BGE query prefix and compares
bits with the stored `.npy`.

| Source (2026-10-08) | with prefix | no prefix |
|---|---|---|
| PubMed (`pubmed26n0500`, `n1600`) | **0.998–0.999** | 0.924–0.953 |
| bioRxiv (`biorxiv_binary_bge`) | 0.947 | **0.9996** |

PubMed documents were embedded **with** `"Represent this sentence for searching
relevant passages: "`, bioRxiv/medRxiv/arXiv without (the BGE convention).

## `ranking_eval.py` — Hamming vs. float-query rescoring

Queries: first synthetic query of each paper in `pubmed_synthetic_queries_10k.parquet`
(2,000 sampled). Corpus: ~1.7M real stored PubMed vectors (60 random files)
plus each query's own paper.

| Seed | Ranking | R@1 | R@10 | R@100 | MRR@10 |
|---|---|---|---|---|---|
| 0 | Hamming (production before) | 0.735 | 0.920 | 0.983 | 0.801 |
| 0 | float-query rescoring | **0.794** | **0.959** | **0.995** | **0.853** |
| 1 | Hamming | 0.676 | 0.909 | 0.982 | 0.758 |
| 1 | float-query rescoring | **0.747** | **0.949** | **0.989** | **0.818** |

Mixing the two document styles (prefixed PubMed + unprefixed "bioRxiv-style"
docs) does **not** skew a merged ranking: the score gap between styles for the
same document is ~0.002, and a per-style offset changes nothing. No correction
is applied in the app.

```bash
python eval/ranking_eval.py \
  --data-dir /mnt/h/pubmed_semantic_search/pubmed_semantic_search \
  --queries ~/pubmed_search/snowflake_code/pubmed_synthetic_queries_10k.parquet
```
~5 min on an RTX 5080 (most of it re-embedding the no-prefix distractors).
