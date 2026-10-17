# Offline evaluation

Read-only scripts that measure search quality on real data. Run them on a GPU
machine with a copy of the data, not on the search server.

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
  --data-dir <data dir> \
  --queries <synthetic queries>.parquet
```
~5 min on an RTX 5080 (most of it re-embedding the no-prefix distractors).

## Exact-term (identifier) matching — calibration, 2026-10-09

40 real PubMed files (~650k papers), real BGE model, top-10 results that
literally contain the query's identifier. Superseded-version filtering was
switched off for this subset only (the newest versions of its records live in
files outside the subset).

| Query | Papers containing it | Semantic only | Hybrid (boost 0.08, ×2 for identifier-only queries) |
|---|---|---|---|
| `TMEM175` | 15 | 1/10 | **9/10** |
| `TMEM175 lysosomal potassium channel in Parkinson disease` | 15 | 2/10 | **8/10** |
| `rs429358` | 17 | 0/10 | **10/10** |
| `APOE rs429358 and Alzheimer disease risk` | 17 | 0/10 | **9/10** |
| `BMS-986165` | 1 | 0 | **ranked #1** |
| `NCT04368728` | 1 | 0 | **ranked #1** |
| `KRAS G12C inhibitor resistance` | 190 | 9/10 | 10/10 |
| two questions without identifiers | — | unchanged | unchanged |

Index (all sources): 9.4M identifiers, 113M postings, ~0.8 GB on disk,
memory-mapped by the backend; built nightly in ~15 min on the GPU machine.

## int8 rescoring — 2026-10-15

Benchmark (separate benchmark scripts, 400k-document sample of all sources, 100 topical queries
with pooled graded judgments): binary search + float-query-vs-bits rescoring
nDCG@10 0.661 → binary search + int8 rescoring **0.725** (69 queries better, 31 worse).
int8 equals exact float rescoring (0.7172 vs 0.7181). Keeping the BGE query prefix
on PubMed documents was better than dropping it (0.717 vs 0.703), so the re-embed
used the same recipe; every one of the 56.4M re-embedded rows matched its stored
binary code (median sign agreement 1.000).

Calibration (`int8_calibration.npy`): per-dimension 0.01–99.99 percentile range of
the 400k sample (min/max and 0.1–99.9 were equivalent; 1–99 lost 0.007 nDCG).

Depth (separate test script, 250 queries, all ~56M vectors, production code):
agreement with re-scoring all ~42k candidates of a search, before dedup —

| Re-scored | top-1 | top-10 | top-50 |
|---|---|---|---|
| 0 (binary only) | 0.59 | 0.56 | 0.56 |
| 250 | 1.00 | 0.985 | 0.898 |
| 500 | 1.00 | 0.994 | 0.957 |
| 1,000 | 1.00 | 0.999 | 0.984 |
| **2,000 (default)** | 1.00 | 1.000 | 0.997 |

The bits → int8 score slope is 0.99 (IQR 0.92–1.07), so `EXACT_BOOST` keeps its value.
On the production CPU server, re-scoring 1,000 candidates takes 0.06–0.09 s (cold reads included);
the default is 2,000 (`MSS_INT8_DEPTH`).
Re-embedding everything: 56.4M documents in 4.9 h on the RTX 5080 (`fast_embed.py`).
