"""
Server-side reader for the paper map (built on the desktop by
map/package_map.py and shipped with the aux indexes):

- places papers and queries on the map: the geometric median of the
  positions of their 10 nearest reference papers (Hamming distance over
  1.52M reference codes, ~55 ms for a query plus 50 results on 4 threads);
- answers "what is here?": the reference papers nearest to a map point;
- serves the density tiles of the loaded version.

Loads <root>/Map/<version>/ only once zzz_complete.json is present.
"""
import glob
import json
import logging
import os
import shutil

import numpy as np

LOGGER = logging.getLogger(__name__)
K_PLACE = 10


class MapIndex:
    def __init__(self, root: str):
        self.dir = os.path.join(root, "Map")
        self.version = None          # version id (directory name)
        self.path = None
        self.meta: dict = {}
        self.labels: dict = {}
        self.stems: list = []
        self._index = self._xy = self._file = self._row = self._tree = None

    @property
    def loaded(self) -> bool:
        return self._index is not None

    def refresh(self) -> bool:
        versions = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(self.dir, "*", "zzz_complete.json")))
        if not versions or versions[-1] == self.path:
            return False
        path = versions[-1]
        try:
            import faiss
            from scipy.spatial import cKDTree
            codes = np.load(os.path.join(path, "ref_codes.npy"))
            index = faiss.IndexBinaryFlat(codes.shape[1] * 8)
            index.add(codes)
            xy = np.load(os.path.join(path, "ref_xy.npy"))
            ref_file = np.load(os.path.join(path, "ref_file.npy"))
            ref_row = np.load(os.path.join(path, "ref_row.npy"))
            with open(os.path.join(path, "meta.json")) as f:
                meta = json.load(f)
            with open(os.path.join(path, "labels.json")) as f:
                labels = json.load(f)
            with open(os.path.join(path, "stems.json")) as f:
                stems = [tuple(s) for s in json.load(f)]
            tree = cKDTree(xy)
        except Exception as exc:
            LOGGER.error(f"Could not load map {path}: {exc}")
            return False
        previous = self.path
        self._index, self._xy, self._file, self._row, self._tree = index, xy, ref_file, ref_row, tree
        self.meta, self.labels, self.stems = meta, labels, stems
        self.path, self.version = path, os.path.basename(path)
        LOGGER.info(f"Map loaded: {path} ({len(xy):,} reference papers)")
        for old in versions[:-2]:
            if old != previous:
                shutil.rmtree(old, ignore_errors=True)
        return True

    def place(self, packed: np.ndarray) -> np.ndarray:
        """(n, 2) map positions for packed binary codes (n, 48)."""
        packed = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1, self._index.d // 8)
        _, nn = self._index.search(packed, K_PLACE)
        pts = self._xy[nn]                                      # (n, k, 2)
        m = np.median(pts, axis=1)
        for _ in range(10):                                     # Weiszfeld iterations
            w = 1 / np.maximum(np.linalg.norm(pts - m[:, None], axis=2), 1e-3)
            m = (pts * w[..., None]).sum(1) / w.sum(1, keepdims=True)
        return m

    def nearby(self, x: float, y: float, k: int = 10) -> list:
        """[(source, file stem, row, distance)] for the reference papers nearest to a point."""
        dist, idx = self._tree.query([x, y], k=k)
        return [(*self.stems[int(self._file[i])], int(self._row[i]), float(d))
                for d, i in zip(np.atleast_1d(dist), np.atleast_1d(idx))]

    def tile_path(self, z: int, x: int, y: int):
        """Path of a tile of the loaded version, or None if empty / out of range."""
        if not self.loaded or not (0 <= z <= self.meta.get("max_zoom", 0)):
            return None
        p = os.path.join(self.path, "tiles", str(z), str(x), f"{y}.png")
        return p if os.path.exists(p) else None
