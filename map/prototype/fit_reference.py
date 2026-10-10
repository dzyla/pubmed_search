"""Step 3: fit reference 2D layouts on the full 1.52M sample with cuML (GPU).
Run with the scratch venv:  venv/bin/python fit_reference.py <variant> [...]
Variants:
  umap_float   UMAP on re-embedded float vectors (n_neighbors=30, min_dist=0.05)
  umap_binary  UMAP on the stored ±1 codes only (no re-embedding)
  tsne_float   cuML t-SNE (FFT) on float vectors
"""
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import WORK, codes_pm1


def load(kind):
    if kind == "float":
        X = np.load(f"{WORK}/sample_emb_f16.npy").astype(np.float32)
    else:
        X = codes_pm1(np.load(f"{WORK}/sample_codes.npy")) / np.sqrt(384)
    return X  # rows are unit-norm -> euclidean ranks == cosine ranks


def main():
    for v in sys.argv[1:]:
        t0 = time.time()
        if v.startswith("umap"):
            from cuml.manifold import UMAP
            kind = v.split("_")[1]
            nn = 30
            X = load(kind)
            r = UMAP(n_neighbors=nn, min_dist=0.05, n_epochs=500, build_algo="nn_descent",
                     build_kwds={"nnd_graph_degree": 64, "nnd_intermediate_graph_degree": 128},
                     init="spectral", random_state=None, verbose=False)
            Y = r.fit_transform(X)
        elif v.startswith("tsne"):
            from cuml.manifold import TSNE
            X = load(v.split("_")[1])
            Y = TSNE(n_components=2, perplexity=50, method="fft", n_neighbors=150,
                     learning_rate_method="adaptive", verbose=False).fit_transform(X)
        Y = np.asarray(Y, dtype=np.float32)
        np.save(f"{WORK}/ref_{v}.npy", Y)
        print(f"{v}: {Y.shape} in {time.time() - t0:.0f}s  range {Y.min(0)} {Y.max(0)} nan={np.isnan(Y).sum()}", flush=True)


if __name__ == "__main__":
    main()
