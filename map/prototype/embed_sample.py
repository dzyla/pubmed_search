"""Step 2: re-embed the sample's "title. abstract" to float with BGE-small (GPU, fp16),
exactly as the stored codes were made (PubMed with the BGE query prefix, others without),
and check bit agreement with the stored codes (validates row alignment)."""
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

WORK = "/mnt/h/pubmed_semantic_search/umap_work"
PREFIX = "Represent this sentence for searching relevant passages: "

meta = pd.read_parquet(f"{WORK}/sample_meta.parquet", columns=["source", "text"])
codes = np.load(f"{WORK}/sample_codes.npy")
texts = np.where(meta.source == "pubmed", PREFIX + meta.text, meta.text).tolist()
del meta

model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
model.half()
model.max_seq_length = 512

order = np.argsort([len(t) for t in texts])[::-1]  # long first -> early OOM detection
emb = np.zeros((len(texts), 384), np.float16)
t0 = time.time()
B = 50_000
for s in range(0, len(order), B):
    idx = order[s:s + B]
    e = model.encode([texts[i] for i in idx], batch_size=256, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
    emb[idx] = e.astype(np.float16)
    if (s // B) % 5 == 0:
        print(f"{s + len(idx):,}/{len(order):,} {time.time() - t0:.0f}s", flush=True)
dt = time.time() - t0
print(f"embedded {len(texts):,} in {dt:.0f}s = {len(texts) / dt:.0f} docs/s")
np.save(f"{WORK}/sample_emb_f16.npy", emb)

src = pd.read_parquet(f"{WORK}/sample_meta.parquet", columns=["source"]).source.values
fresh = np.packbits(emb.astype(np.float32) > 0, axis=1)
agree = 1 - np.unpackbits(fresh ^ codes, axis=1).mean(1)
for s in np.unique(src):
    a = agree[src == s]
    print(f"bit agreement {s}: mean={a.mean():.4f} p1={np.percentile(a, 1):.3f} frac<0.9={np.mean(a < 0.9):.4f}")
np.save(f"{WORK}/sample_bit_agreement.npy", agree.astype(np.float32))
