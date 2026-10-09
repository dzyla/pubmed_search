"""
Checks which text recipe produced the stored binary embeddings for a source:
re-embeds a sample of rows with and without the BGE query prefix and compares
bits against the stored .npy (read-only).

Usage: python eval/check_doc_prefix.py <file.npy> <file.parquet> [n_rows]
"""
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PREFIX = "Represent this sentence for searching relevant passages: "

npy_path, parquet_path = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 32

stored = np.load(npy_path, mmap_mode="r", allow_pickle=False)
df = pd.read_parquet(parquet_path, columns=["title", "abstract"])
idx = np.linspace(0, len(df) - 1, n).astype(int)
texts = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).iloc[idx].tolist()

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
for label, batch in [("with prefix", [PREFIX + t for t in texts]), ("no prefix", texts)]:
    emb = model.encode(batch, normalize_embeddings=True, convert_to_numpy=True)
    bits = np.packbits(emb > 0, axis=1)
    agree = 1 - np.unpackbits(bits ^ stored[idx], axis=1).mean()
    print(f"{label:12s}: {agree:.4f} bit agreement with stored vectors")
