"""
Server-side reader for the auxiliary indexes built by
update_database/build_aux_indexes.py:

- identifier postings (exact-term matching), memory-mapped from disk, so
  they cost page cache rather than process RAM;
- PubMed superseded rows (older copies of revised records), which the
  searcher turns into a bitmap over its row ids and hides.

Each source lives in <root>/<Source>/<version>/; only versions containing
zzz_complete.json are loaded (the builder writes it last, so a version that
is still being synced is ignored until complete).
"""
import glob
import json
import logging
import os
import shutil

import numpy as np

LOGGER = logging.getLogger(__name__)
MAX_POSTINGS_PER_TOKEN = 20_000


class AuxIndex:
    def __init__(self, root: str, source: str):
        self.dir = os.path.join(root, source)
        self.version = None
        self.stems: list = []
        self._hash = self._offsets = self._stem = self._row = None
        self.superseded = None          # (stem_ids, rows) or None
        self.meta: dict = {}

    def _complete_versions(self):
        return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(self.dir, "*", "zzz_complete.json")))

    def refresh(self) -> bool:
        """Loads the newest complete version if it changed. Returns True if it did."""
        versions = self._complete_versions()
        if not versions or versions[-1] == self.version:
            return False
        path = versions[-1]
        try:
            load = lambda name: np.load(os.path.join(path, name), mmap_mode="r")   # noqa: E731
            self._hash, self._offsets = load("hash.npy"), load("offsets.npy")
            self._stem, self._row = load("stem.npy"), load("row.npy")
            with open(os.path.join(path, "stems.json")) as f:
                self.stems = json.load(f)
            with open(os.path.join(path, "zzz_complete.json")) as f:
                self.meta = json.load(f)
            sup = os.path.join(path, "superseded_stem.npy")
            self.superseded = ((np.load(sup), np.load(os.path.join(path, "superseded_row.npy")))
                               if os.path.exists(sup) else None)
        except Exception as exc:
            LOGGER.error(f"Could not load aux index {path}: {exc}")
            return False
        previous, self.version = self.version, path
        LOGGER.info(f"Aux index loaded: {path} ({self.meta.get('tokens', 0):,} tokens, "
                    f"{self.meta.get('superseded_rows', 0):,} superseded rows)")
        # The synced copies of older versions are no longer needed (keep the previous one).
        for old in versions[:-2]:
            if old != previous:
                shutil.rmtree(old, ignore_errors=True)
        return True

    @property
    def loaded(self) -> bool:
        return self._hash is not None

    def postings(self, token_hash: int):
        """(stem_ids, rows) for one token hash, or None if absent / too common."""
        if not self.loaded:
            return None
        h = np.uint64(token_hash)
        i = int(np.searchsorted(self._hash, h))
        if i >= len(self._hash) or self._hash[i] != h:
            return None
        a, b = int(self._offsets[i]), int(self._offsets[i + 1])
        if b - a > MAX_POSTINGS_PER_TOKEN:
            return None
        return np.asarray(self._stem[a:b]), np.asarray(self._row[a:b])
