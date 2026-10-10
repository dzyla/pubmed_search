"""Paper map: placement, 'what is here' lookups, tiles and map positions in search results."""
import json
import os

import numpy as np
import pytest

from conftest import N_FILES, ROWS_PER_FILE
from map_index import MapIndex
from conftest import INTERNAL


def write_package(root, codes, xy, stems, ref_file, ref_row, version="20260101-000000"):
    out = os.path.join(root, "Map", version)
    os.makedirs(os.path.join(out, "tiles", "0", "0"))
    with open(os.path.join(out, "tiles", "0", "0", "0.png"), "wb") as f:
        f.write(b"\x89PNG-test-tile")
    np.save(os.path.join(out, "ref_codes.npy"), codes)
    np.save(os.path.join(out, "ref_xy.npy"), xy.astype(np.float32))
    np.save(os.path.join(out, "ref_file.npy"), np.asarray(ref_file, dtype=np.int32))
    np.save(os.path.join(out, "ref_row.npy"), np.asarray(ref_row, dtype=np.int32))
    meta = {"world": {"x0": -50.0, "y0": -50.0, "size": 100.0}, "max_zoom": 0, "tile": 256,
            "colors": {"PubMed": "#4d9be8"}, "counts": {"PubMed": len(xy)}}
    for name, obj in (("stems.json", stems), ("labels.json", {"coarse": [], "fine": []}), ("meta.json", meta),
                      ("zzz_complete.json", meta)):
        with open(os.path.join(out, name), "w") as f:
            json.dump(obj, f)
    return version


def test_place_lands_in_the_right_region(tmp_path):
    rng = np.random.default_rng(1)
    protos = rng.integers(0, 256, (2, 48), dtype=np.uint8)

    def noisy(p, n):        # codes a few bits away from a prototype
        bits = np.repeat(np.unpackbits(p)[None], n, 0)
        flips = rng.random(bits.shape) < 0.05
        return np.packbits(bits ^ flips, axis=1)

    codes = np.vstack([noisy(protos[0], 200), noisy(protos[1], 200)])
    xy = np.vstack([rng.normal((-20, 0), 2, (200, 2)), rng.normal((20, 0), 2, (200, 2))])
    write_package(str(tmp_path), codes, xy, [["PubMed", "x"]], [0] * 400, range(400))
    idx = MapIndex(str(tmp_path))
    assert idx.refresh() and not idx.refresh()          # second call: nothing new
    pts = idx.place(np.vstack([noisy(protos[0], 1), noisy(protos[1], 1)]))
    assert pts[0, 0] < -15 and pts[1, 0] > 15
    assert idx.nearby(20, 0, k=3)[0][0] == "PubMed"
    assert idx.tile_path(0, 0, 0) and idx.tile_path(0, 1, 0) is None and idx.tile_path(3, 0, 0) is None


@pytest.fixture
def map_backend(backend, backend_corpus, tmp_path, monkeypatch):
    """The shared backend with a map whose reference papers are the test corpus."""
    client, search_api = backend
    config, emb = backend_corpus
    stems = [["PubMed", f"part_{f:02d}"] for f in range(N_FILES)]
    ref_file = np.repeat(np.arange(N_FILES), ROWS_PER_FILE)
    ref_row = np.tile(np.arange(ROWS_PER_FILE), N_FILES)
    xy = np.random.default_rng(2).uniform(-40, 40, (len(emb), 2))
    xy[np.linalg.norm(xy - (33.0, -33.0), axis=1) < 5] += (-20, 20)
    xy[250] = (33.0, -33.0)                              # file 2, row 50: alone in its corner
    version = write_package(str(tmp_path), emb, xy, stems, ref_file, ref_row)
    index = MapIndex(str(tmp_path))
    assert index.refresh()
    monkeypatch.setattr(search_api, "MAP", index)
    monkeypatch.setattr(search_api, "CONFIGS", [{**config, "source_name": "PubMed"}, {}, {}, {}])
    return client, version


def test_map_info_and_tiles(map_backend):
    client, version = map_backend
    info = client.get("/v1/map").json()
    assert info["version"] == version and info["tiles"] == f"/map/tiles/{version}/{{z}}/{{x}}/{{y}}.png"
    tile = client.get(f"/map/tiles/{version}/0/0/0.png")
    assert tile.status_code == 200 and tile.content == b"\x89PNG-test-tile"
    assert "immutable" in tile.headers["cache-control"]
    empty = client.get(f"/map/tiles/{version}/0/5/5.png")          # empty area: transparent tile
    assert empty.status_code == 200 and empty.content.startswith(b"\x89PNG")


def test_map_nearby_resolves_papers(map_backend):
    client, _ = map_backend
    papers = client.get("/v1/map/nearby", params={"x": 33.0, "y": -33.0, "k": 3}).json()["papers"]
    assert papers[0]["title"] == "Paper 2-50" and papers[0]["ref"].startswith("PubMed:")


def test_search_results_carry_map_positions(map_backend):
    client, _ = map_backend
    body = client.post("/search", json={"query": "binary embeddings", "top_k": 5}, headers=INTERNAL).json()
    assert len(body["query_map_xy"]) == 2
    assert all(len(p["map_xy"]) == 2 for p in body["results"])


def test_leaflet_is_served_from_this_origin(map_backend):
    client, _ = map_backend
    js = client.get("/map/static/leaflet.js")
    assert js.status_code == 200 and js.text.startswith("/* @preserve") and "javascript" in js.headers["content-type"]
    assert client.get("/map/static/../search_api.py").status_code == 404
    assert client.get("/map/static/other.js").status_code == 404
