"""Render density / source / labelled map PNGs from 2D coordinates.
  python render.py <coords.npy> <sources.npy|meta> <out_prefix> [--labels] [--weights w.npy]
coords: (n,2) float; sources: (n,) uint8 codes in SOURCES order.
Pure numpy histogram + matplotlib (log colour scale), datashader-style.
"""
import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

SOURCES = ["pubmed", "arxiv", "biorxiv", "medrxiv", "clinicaltrials"]
NAMES = ["PubMed", "arXiv", "bioRxiv", "medRxiv", "ClinicalTrials.gov"]
COLORS = np.array([[0.30, 0.61, 0.91], [0.95, 0.56, 0.17], [0.88, 0.34, 0.35],
                   [0.35, 0.80, 0.31], [0.78, 0.48, 0.85]])


def extent(coords, q=0.3):
    lo = np.percentile(coords, q, axis=0); hi = np.percentile(coords, 100 - q, axis=0)
    pad = 0.03 * (hi - lo)
    return lo - pad, hi + pad


def hist(coords, lo, hi, res, weights=None):
    W = res
    H = int(round(res * (hi[1] - lo[1]) / (hi[0] - lo[0])))
    h, _, _ = np.histogram2d(coords[:, 1], coords[:, 0], bins=[H, W],
                             range=[[lo[1], hi[1]], [lo[0], hi[0]]], weights=weights)
    return h  # rows = y


def log_norm(h, floor_q=0.0):
    v = np.log1p(h)
    m = np.percentile(v[v > 0], 99.7) if (v > 0).any() else 1
    return np.clip(v / m, 0, 1)


def save(img, path, title=None, labels=None, lo=None, hi=None, legend=None):
    H, W = img.shape[:2]
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(img, origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]], interpolation="nearest")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    fs = max(9, W // 160)
    if title:
        ax.text(0.01, 0.99, title, transform=ax.transAxes, va="top", color="white", fontsize=fs * 1.3)
    if legend:
        for i, (name, col) in enumerate(legend):
            ax.text(0.01, 0.95 - i * 0.025, "■ " + name, transform=ax.transAxes, va="top", color=col, fontsize=fs)
    if labels:
        for (x, y, txt, size) in labels:
            ax.text(x, y, txt, color="white", fontsize=fs * size, ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")])
    fig.savefig(path, facecolor="black"); plt.close(fig)


def density_img(h):
    return plt.get_cmap("inferno")(log_norm(h))[..., :3]


def source_img(coords, src, lo, hi, res, per_source_norm=True):
    hs = np.stack([hist(coords[src == i], lo, hi, res) for i in range(len(SOURCES))])
    tot = hs.sum(0)
    w = hs / np.maximum(hs.reshape(len(SOURCES), -1).sum(1), 1)[:, None, None] if per_source_norm else hs
    rgb = np.einsum("shw,sc->hwc", w, COLORS) / np.maximum(w.sum(0), 1e-12)[..., None]
    bright = log_norm(tot) ** 0.8
    return rgb * bright[..., None]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("coords"); ap.add_argument("sources"); ap.add_argument("out")
    ap.add_argument("--res", type=int, default=2000)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    c = np.load(a.coords, mmap_mode="r"); s = np.load(a.sources, mmap_mode="r")
    c = np.asarray(c, dtype=np.float32); s = np.asarray(s)
    lo, hi = extent(c)
    h = hist(c, lo, hi, a.res)
    save(density_img(h), a.out + "_density.png", f"{a.title}  n={len(c):,}  (log density)", lo=lo, hi=hi)
    save(source_img(c, s, lo, hi, a.res), a.out + "_sources.png", f"{a.title}  colour = source share (each source normalised)",
         lo=lo, hi=hi, legend=[(n, tuple(col)) for n, col in zip(NAMES, COLORS)])
    print("occupied pixels", (h > 0).mean())
