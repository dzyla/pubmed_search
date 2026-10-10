"""Step 7: full-corpus renders + XYZ tile pyramid from projected coordinates.
  python tiles.py <coords_dir> <out_dir> <max_zoom> [--ref otsne_float_ex2]
Reads coords_<source>.npy (float32, n x 2). Writes
  <out_dir>/full_density.png, full_sources.png   (4096 px overviews)
  <out_dir>/tiles/{z}/{x}/{y}.png               (256 px RGBA, transparent where empty; y=0 at top)
  <out_dir>/map_meta.json                       (world bounds -> pixel transform, counts, timings)
Colour: brightness = log(count) normalised per zoom level, hue = per-source share (each source
normalised by its own size, so the 0.09M medRxiv docs are visible next to 45M PubMed).
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from render import COLORS, NAMES, SOURCES

coords_dir, out, ZMAX = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
T0 = time.time()

C = {s: np.load(f"{coords_dir}/coords_{s}.npy", mmap_mode="r") for s in SOURCES}
allc = np.concatenate([np.asarray(C[s][::50]) for s in SOURCES])
lo = np.percentile(allc, 0.02, 0); hi = np.percentile(allc, 99.98, 0)
ctr = (lo + hi) / 2; half = (hi - lo).max() / 2 * 1.03       # square world, 3 % margin
x0, y0, size = ctr[0] - half, ctr[1] - half, 2 * half
W = 256 * 2 ** ZMAX
print(f"world square [{x0:.2f},{y0:.2f}] size {size:.2f}; top-level image {W}x{W}", flush=True)

# integer pixel coordinates at max zoom; y flipped so row 0 = top (slippy-map convention)
hists = []
for s in SOURCES:
    c = np.asarray(C[s], dtype=np.float32)
    px = np.clip(((c[:, 0] - x0) / size * W).astype(np.int64), 0, W - 1)
    py = np.clip(((1 - (c[:, 1] - y0) / size) * W).astype(np.int64), 0, W - 1)
    hists.append(np.bincount(py * W + px, minlength=W * W).astype(np.uint32).reshape(W, W))
    print(f"  hist {s}: {len(c):,} pts {time.time() - T0:.0f}s", flush=True)
H = np.stack(hists)                           # (S, W, W) uint32
ntot = np.array([h.sum() for h in hists], dtype=np.float64)


def down(h):                                  # exact 2x2 sum pyramid
    S_, n, _ = h.shape
    return h.reshape(S_, n // 2, 2, n // 2, 2).sum((2, 4), dtype=np.uint32)


def colorize(h, vmax):
    tot = h.sum(0).astype(np.float32)
    w = h / ntot[:, None, None]
    rgb = np.einsum("shw,sc->hwc", w, COLORS) / np.maximum(w.sum(0), 1e-30)[..., None]
    b = np.clip(np.log1p(tot) / np.log1p(vmax), 0, 1)
    rgb = rgb * (0.35 + 0.65 * b[..., None])   # keep hue visible in sparse areas
    a = np.where(tot > 0, 0.25 + 0.75 * b, 0)   # transparent background
    return (np.dstack([rgb, a]) * 255).astype(np.uint8)


levels = {ZMAX: H}
for z in range(ZMAX - 1, -1, -1):
    levels[z] = down(levels[z + 1])
stats = {"tiles": {}, "bytes": {}}
for z in range(ZMAX + 1):
    h = levels[z]
    tot = h.sum(0)
    vmax = np.percentile(tot[tot > 0], 99.9)
    n = 2 ** z
    cnt = 0; nbytes = 0
    for ty in range(n):
        for tx in range(n):
            t = h[:, ty * 256:(ty + 1) * 256, tx * 256:(tx + 1) * 256]
            if not t.any():
                continue
            d = f"{out}/tiles/{z}/{tx}"
            os.makedirs(d, exist_ok=True)
            p = f"{d}/{ty}.png"
            Image.fromarray(colorize(t, vmax), "RGBA").save(p, optimize=False, compress_level=6)
            cnt += 1; nbytes += os.path.getsize(p)
    stats["tiles"][z] = cnt; stats["bytes"][z] = nbytes
    print(f"  z{z}: {cnt} tiles, {nbytes / 1e6:.1f} MB, {time.time() - T0:.0f}s", flush=True)

# overview images (4096 px) with the same colouring, black background
zo = min(4, ZMAX)
h = levels[zo]
tot = h.sum(0)
import matplotlib.pyplot as plt  # noqa: E402
dens = plt.get_cmap("inferno")(np.clip(np.log1p(tot) / np.log1p(np.percentile(tot[tot > 0], 99.9)), 0, 1))
Image.fromarray((dens[..., :3] * 255).astype(np.uint8)).save(f"{out}/full_density.png")
rgba = colorize(h, np.percentile(tot[tot > 0], 99.9)).astype(np.float32) / 255
Image.fromarray((rgba[..., :3] * rgba[..., 3:] * 255).astype(np.uint8)).save(f"{out}/full_sources.png")
meta = dict(world=dict(x0=float(x0), y0=float(y0), size=float(size)), max_zoom=ZMAX, tile=256,
            sources=SOURCES, counts={s: int(n) for s, n in zip(SOURCES, ntot)}, overview_zoom=zo,
            seconds=round(time.time() - T0, 1), **stats)
json.dump(meta, open(f"{out}/map_meta.json", "w"), indent=1)
print(json.dumps(meta)[:600])
