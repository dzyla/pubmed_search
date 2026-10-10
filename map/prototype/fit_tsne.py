"""openTSNE (CPU, FFT) on the exact kNN graph; Kobak & Berens large-data recipe:
PCA init rescaled, lr = N/12, early exaggeration 12 (250 it), then exaggeration EX (500 it),
optionally followed by exaggeration 1.  Saves ref_otsne_<kind>_ex<EX>.npy and ref_otsne_<kind>_ex1.npy.
  venv/bin/python fit_tsne.py float 2
"""
import sys
import time

import numpy as np
from openTSNE import TSNEEmbedding, affinity, initialization
from openTSNE.nearest_neighbors import PrecomputedNeighbors

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import WORK

kind = sys.argv[1]
EX = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
I = np.load(f"{WORK}/knn_{kind}_I.npy").astype(np.int64)
D = np.load(f"{WORK}/knn_{kind}_D.npy").astype(np.float64)
n = len(I)
t0 = time.time()
aff = affinity.PerplexityBasedNN(knn_index=PrecomputedNeighbors(I, D), perplexity=30, n_jobs=24)
print(f"affinities {time.time() - t0:.0f}s", flush=True)
# PCA init on the float vectors (or ±1 codes)
if kind == "float":
    X = np.load(f"{WORK}/sample_emb_f16.npy").astype(np.float32)
else:
    X = np.unpackbits(np.load(f"{WORK}/sample_codes.npy"), axis=1).astype(np.float32)
sub = X[::20]
mu = sub.mean(0)
_, _, Vt = np.linalg.svd(sub - mu, full_matrices=False)
init = (X - mu) @ Vt[:2].T
del X
init = initialization.rescale(init.astype(np.float64))
emb = TSNEEmbedding(init, aff, n_jobs=24, negative_gradient_method="fft", random_state=0)
lr = n / 12
emb = emb.optimize(n_iter=250, exaggeration=12, learning_rate=lr, momentum=0.5)
print(f"early exaggeration done {time.time() - t0:.0f}s", flush=True)
emb = emb.optimize(n_iter=500, exaggeration=EX, learning_rate=lr, momentum=0.8)
np.save(f"{WORK}/ref_otsne_{kind}_ex{EX:g}.npy", np.asarray(emb, dtype=np.float32))
print(f"ex{EX:g} done {time.time() - t0:.0f}s", flush=True)
if EX != 1:
    emb = emb.optimize(n_iter=300, exaggeration=1, learning_rate=lr, momentum=0.8)
    np.save(f"{WORK}/ref_otsne_{kind}_ex1.npy", np.asarray(emb, dtype=np.float32))
    print(f"ex1 done {time.time() - t0:.0f}s", flush=True)
