"""Demo of the per-query endpoint: place a query and its top-50 hits on the map, CPU only
(simulates the 4-vCPU server: faiss IndexBinaryFlat over the 1.52M reference codes + geometric median).
Hits come from the 1.52M sample (stand-in for the production search) to keep the demo self-contained.
  python demo_query.py <map_dir> <out.png> "query 1" "query 2" ...
"""
import json
import sys
import time

import faiss
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from PIL import Image
from sentence_transformers import SentenceTransformer

WORK = "/mnt/h/pubmed_semantic_search/umap_work"
PREFIX = "Represent this sentence for searching relevant passages: "
REF = "otsne_float_ex2"


class MapPlacer:
    """What the backend would hold: 1.52M x 48 B codes (73 MB) + 1.52M x 2 float32 (12 MB)."""

    def __init__(self, threads=4):
        faiss.omp_set_num_threads(threads)
        self.index = faiss.IndexBinaryFlat(384)
        self.index.add(np.load(f"{WORK}/sample_codes.npy"))
        self.xy = np.load(f"{WORK}/ref_{REF}.npy")

    def place(self, packed, k=10):
        _, I = self.index.search(np.ascontiguousarray(packed), k)
        nb = self.xy[I]
        m = np.median(nb, 1)
        for _ in range(10):
            w = 1 / np.maximum(np.linalg.norm(nb - m[:, None], axis=2), 1e-3)
            m = (nb * w[..., None]).sum(1) / w.sum(1, keepdims=True)
        return m


def main(map_dir, out, queries):
    meta = json.load(open(f"{map_dir}/map_meta.json"))
    w = meta["world"]
    img = np.asarray(Image.open(f"{map_dir}/full_sources.png").convert("RGB"))
    N = img.shape[0]
    to_px = lambda p: ((p[:, 0] - w["x0"]) / w["size"] * N, (1 - (p[:, 1] - w["y0"]) / w["size"]) * N)
    placer = MapPlacer()
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    codes = np.load(f"{WORK}/sample_codes.npy")
    titles = __import__("pandas").read_parquet(f"{WORK}/sample_meta.parquet", columns=["title"]).title.values
    fig = plt.figure(figsize=(N / 200, N / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.imshow(img * 0.6 / 255)
    cols = ["#00e5ff", "#ffeb3b", "#ff4081", "#76ff03", "#ffffff"]
    for qi, q in enumerate(queries):
        qv = model.encode([PREFIX + q], normalize_embeddings=True)
        qcode = np.packbits(qv > 0, axis=1)
        _, hits = placer.index.search(qcode, 50)               # stand-in for the production search
        t0 = time.time()
        pts = placer.place(np.vstack([qcode, codes[hits[0]]]))
        dt = (time.time() - t0) * 1000
        x, y = to_px(pts)
        ax.scatter(x[1:], y[1:], s=10, c=cols[qi % 5], edgecolors="black", linewidths=0.3)
        ax.scatter(x[:1], y[:1], s=120, marker="*", c=cols[qi % 5], edgecolors="black", linewidths=0.6)
        ax.text(x[0], y[0] - 25, q, color=cols[qi % 5], fontsize=8, ha="center",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
        spread = np.median(np.linalg.norm(pts[1:] - np.median(pts[1:], 0), axis=1)) / w["size"] * 100
        print(f"{q!r}: placed query+50 hits in {dt:.0f} ms (4 threads); hit spread median {spread:.1f}% of map width; "
              f"top hit: {titles[hits[0][0]][:80]}")
    fig.savefig(out, facecolor="black")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
