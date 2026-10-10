"""
Packages a finished map build into the aux-index tree, which the nightly
aux-index sync ships to the server:

    <base>/aux_index/Map/<version>/
        tiles/{z}/{x}/{y}.png     density tiles (y = 0 at the top)
        meta.json                 world -> pixel transform, zoom range, sources, colours
        labels.json               topic labels at two levels (coarse for overview, fine for zoom >= 4)
        ref_codes.npy             (n, 48) uint8  binary codes of the reference papers
        ref_xy.npy                (n, 2) float32 their map positions
        ref_file.npy, ref_row.npy (n,) int32    which data file (stems.json) and row they are
        stems.json                [[source, file stem], ...]
        zzz_complete.json         written last

The server places any paper or query by the geometric median of the positions of
its 10 nearest reference papers (Hamming distance), and answers "what is here?"
clicks with the reference papers nearest to a point. See map/prototype/ for how
the layout itself was fitted (openTSNE on a 1.52M stratified sample).

    python map/package_map.py [--work /mnt/h/.../umap_work] [--base <snowflake>]
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "prototype"))
from labels import terms  # noqa: E402

SOURCE_NAMES = {"pubmed": "PubMed", "arxiv": "arXiv", "biorxiv": "BioRxiv", "medrxiv": "MedRxiv",
                "clinicaltrials": "ClinicalTrials"}
COLORS = {"PubMed": "#4d9be8", "arXiv": "#f28f2b", "BioRxiv": "#e05759", "MedRxiv": "#5ec962",
          "ClinicalTrials": "#c77ddb"}
# Embedding folders on the desktop, per source (file stems are the .npy names)
EMBED_DIRS = {"PubMed": "pubmed26_update_embed", "arXiv": "arxiv_embed", "BioRxiv": "biorxiv_embed_binary",
              "MedRxiv": "medarxiv_embed_binary", "ClinicalTrials": "clinicaltrials_embed"}
KEEP_VERSIONS = 2


def current_codes(base, stems, ref_file, rows):
    """Codes as stored in the data files now (what the server holds), for each reference paper."""
    out = np.zeros((len(rows), 48), dtype=np.uint8)
    for i, (source, stem) in enumerate(stems):
        m = ref_file == i
        arr = np.load(os.path.join(base, EMBED_DIRS[source], stem + ".npy"), mmap_mode="r")
        out[m] = arr[rows[m]]
    return out


def region_labels(xy, meta_terms, k, min_docs, seed=0):
    """k-means regions on the map, each named by its two most distinctive terms."""
    lo, hi = np.percentile(xy, 0.3, 0), np.percentile(xy, 99.7, 0)
    inside = np.all((xy >= lo) & (xy <= hi), 1)
    km = KMeans(k, n_init=3, random_state=seed).fit(xy[inside][::3])
    lab = np.full(len(xy), -1)
    lab[inside] = km.predict(xy[inside])
    has = np.array([len(t) > 0 for t in meta_terms])
    from collections import Counter
    overall = Counter(t for ts in meta_terms for t in ts)
    n_has = has.sum()
    out = []
    for c in range(k):
        idx = np.flatnonzero((lab == c) & has)
        if len(idx) < min_docs:
            continue
        cnt = Counter(t for i in idx for t in meta_terms[i])
        best = sorted(((n / len(idx)) * np.log((n / len(idx)) / (overall[t] / n_has)), t)
                      for t, n in cnt.items() if n / len(idx) >= 0.04)[::-1]
        if best:
            x, y = np.median(xy[lab == c], 0)
            out.append({"x": round(float(x), 3), "y": round(float(y), 3),
                        "text": "\n".join(t for _, t in best[:2]), "n": int((lab == c).sum())})
    return out


def make_preview(tiles_dir: str, out_path: str, zoom: int = 3, field=(20, 16, 42), width=1400, aspect=2.6):
    """Wide still of the map for the landing page: tiles of one zoom level on the map's
    dark background, cropped to a band through the middle (where the papers are)."""
    from PIL import Image
    n, size = 2 ** zoom, 256
    world = Image.new("RGB", (n * size, n * size), field)
    for x in range(n):
        for y in range(n):
            p = os.path.join(tiles_dir, str(zoom), str(x), f"{y}.png")
            if os.path.exists(p):
                tile = Image.open(p).convert("RGBA")
                world.paste(tile, (x * size, y * size), tile)
    band = int(world.width / aspect)
    top = (world.height - band) // 2
    world = world.crop((0, top, world.width, top + band)).resize((width, int(width / aspect)), Image.LANCZOS)
    world.save(out_path, "JPEG", quality=82, optimize=True, progressive=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="/mnt/h/pubmed_semantic_search/umap_work")
    ap.add_argument("--map-dir", default="map_gm_ex2")
    ap.add_argument("--ref", default="otsne_float_ex2")
    ap.add_argument("--base", default="/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake")
    args = ap.parse_args()
    t0 = time.time()

    meta_df = pd.read_parquet(os.path.join(args.work, "sample_meta.parquet"),
                              columns=["source", "file", "row", "mesh_terms", "categories", "category", "conditions"])
    codes = np.load(os.path.join(args.work, "sample_codes.npy"))
    xy = np.load(os.path.join(args.work, f"ref_{args.ref}.npy")).astype(np.float32)
    assert len(meta_df) == len(codes) == len(xy), (len(meta_df), len(codes), len(xy))

    pairs = list(zip(meta_df["source"].map(SOURCE_NAMES), meta_df["file"].str.removesuffix(".npy")))
    stems = sorted(set(pairs))
    stem_id = {p: i for i, p in enumerate(stems)}
    ref_file = np.array([stem_id[p] for p in pairs], dtype=np.int32)

    rows = meta_df["row"].to_numpy(np.int32)
    fresh = current_codes(args.base, stems, ref_file, rows)
    same = (fresh == codes).all(1)
    print(f"reference codes: {same.mean():.2%} unchanged since the map was built; using current ones")
    codes = fresh

    T = [terms(r) for r in meta_df.itertuples()]
    labels = {"coarse": region_labels(xy, T, 60, 200), "fine": region_labels(xy, T, 400, 60, seed=1)}
    print(f"labels: {len(labels['coarse'])} coarse, {len(labels['fine'])} fine ({time.time() - t0:.0f}s)")

    with open(os.path.join(args.work, args.map_dir, "map_meta.json")) as f:
        tile_meta = json.load(f)
    version = time.strftime("%Y%m%d-%H%M%S")
    root = os.path.join(args.base, "aux_index", "Map")
    out = os.path.join(root, version)
    os.makedirs(out)
    shutil.copytree(os.path.join(args.work, args.map_dir, "tiles"), os.path.join(out, "tiles"))
    make_preview(os.path.join(out, "tiles"), os.path.join(out, "preview.jpg"))
    np.save(os.path.join(out, "ref_codes.npy"), np.ascontiguousarray(codes, dtype=np.uint8))
    np.save(os.path.join(out, "ref_xy.npy"), xy)
    np.save(os.path.join(out, "ref_file.npy"), ref_file)
    np.save(os.path.join(out, "ref_row.npy"), rows)
    with open(os.path.join(out, "stems.json"), "w") as f:
        json.dump(stems, f)
    with open(os.path.join(out, "labels.json"), "w") as f:
        json.dump(labels, f)
    meta = {"version": version, "world": tile_meta["world"], "max_zoom": tile_meta["max_zoom"],
            "tile": tile_meta["tile"], "colors": COLORS,
            "counts": {SOURCE_NAMES[s]: n for s, n in tile_meta["counts"].items()},
            "reference_papers": int(len(xy))}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(out, "zzz_complete.json"), "w") as f:
        json.dump(meta, f)
    for old in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))[:-KEEP_VERSIONS]:
        shutil.rmtree(os.path.join(root, old), ignore_errors=True)
    print(json.dumps({**meta, "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
