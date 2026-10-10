"""Overlay topic labels (labels.py JSON, world coords) on a tiles.py overview PNG.
  python overlay_labels.py <overview.png> <map_meta.json> <labels.json> <out.png> [title]
"""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from render import COLORS, NAMES

img = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
meta = json.load(open(sys.argv[2])); labels = json.load(open(sys.argv[3]))
title = sys.argv[5] if len(sys.argv) > 5 else ""
w = meta["world"]; N = img.shape[0]
fig = plt.figure(figsize=(N / 200, N / 200), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.imshow(img)
for l in labels:
    px = (l["x"] - w["x0"]) / w["size"] * N
    py = (1 - (l["y"] - w["y0"]) / w["size"]) * N
    ax.text(px, py, l["text"], color="white", fontsize=6.5, ha="center", va="center", linespacing=0.95,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])
if title:
    ax.text(0.01, 0.995, title, transform=ax.transAxes, va="top", color="white", fontsize=11)
for i, (n, c) in enumerate(zip(NAMES, COLORS)):
    cnt = meta["counts"][meta["sources"][i]]
    ax.text(0.01, 0.97 - i * 0.017, f"■ {n}  ({cnt / 1e6:.2f}M)", transform=ax.transAxes, va="top",
            color=tuple(c), fontsize=8)
fig.savefig(sys.argv[4], facecolor="black"); print("saved", sys.argv[4])
