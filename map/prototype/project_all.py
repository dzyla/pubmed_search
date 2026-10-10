"""Step 6: project ALL stored codes (~50M); read .npy from H: (read-only).
  python project_all.py <mapper.pt | knn:<ref_name>> <out_dir>
  knn:<ref>  = geometric median of the 10 Hamming-nearest reference docs (GPU matmul on ±1)
Writes <out_dir>/coords_<source>.npy  float32 (n,2), row-aligned with the concatenation of that
source's .npy files in sorted order, plus files_<source>.txt (file, rows) and timing stats.
"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mappers import Mapper, unpack_gpu

ROOT = "/mnt/h/pubmed_semantic_search/pubmed_semantic_search/snowflake"
SRC = {
    "pubmed": sorted(glob.glob(f"{ROOT}/pubmed26_update_embed/*.npy")),
    "arxiv": sorted(glob.glob(f"{ROOT}/arxiv_embed/*.npy")),
    "biorxiv": [f"{ROOT}/biorxiv_embed_binary/biorxiv_binary_bge.npy"],
    "medrxiv": [f"{ROOT}/medarxiv_embed_binary/medarxiv_binary_bge.npy"],
    "clinicaltrials": sorted(glob.glob(f"{ROOT}/clinicaltrials_embed/*.npy")),
}


def geomed(nb, iters=10):
    """Weiszfeld geometric median of (n,k,2) neighbour positions (rotation invariant, no axis snapping)."""
    m = nb.median(dim=1).values
    for _ in range(iters):
        w = 1 / (nb - m[:, None]).norm(dim=2).clamp_min(1e-3)
        m = (nb * w[..., None]).sum(1) / w.sum(1, keepdim=True)
    return m


def main(mpath, out):
    os.makedirs(out, exist_ok=True)
    if mpath.startswith("knn:"):
        WORK = "/mnt/h/pubmed_semantic_search/umap_work"
        R = unpack_gpu(torch.from_numpy(np.load(f"{WORK}/sample_codes.npy")).cuda())
        RY = torch.from_numpy(np.load(f"{WORK}/ref_{mpath[4:]}.npy")).cuda()

        def m(x):
            out = []
            for s in range(0, len(x), 4096):
                i = (x[s:s + 4096].half() @ R.T).topk(10, dim=1).indices
                out.append(geomed(RY[i]))
            return torch.cat(out)
        BS = 1 << 16
    else:
        sd = torch.load(mpath, map_location="cuda")
        h = sd["net.0.weight"].shape[0]
        depth = sum(1 for k in sd if k.startswith("net.") and k.endswith(".weight") and sd[k].ndim == 2) - 1
        m = Mapper(h=h, depth=depth).cuda(); m.load_state_dict(sd); m.eval()
        BS = 1 << 18
    stats = {}
    T0 = time.time()
    for src, files in SRC.items():
        t0 = time.time(); t_read = 0.0; t_gpu = 0.0
        outs, manifest = [], []

        def load(f):
            return f, np.load(f)            # whole file into RAM (<= ~1.5 MB per PubMed file)

        with ThreadPoolExecutor(8) as ex:
            for f, a in ex.map(load, files):
                manifest.append((os.path.basename(f), len(a)))
                t1 = time.time()
                with torch.no_grad():
                    for s in range(0, len(a), BS):
                        x = unpack_gpu(torch.from_numpy(a[s:s + BS]).cuda()).float()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            outs.append(m(x).float().cpu().numpy().astype(np.float32))
                t_gpu += time.time() - t1
        coords = np.concatenate(outs) if outs else np.zeros((0, 2), np.float32)
        np.save(f"{out}/coords_{src}.npy", coords)
        with open(f"{out}/files_{src}.txt", "w") as fh:
            fh.writelines(f"{a}\t{b}\n" for a, b in manifest)
        dt = time.time() - t0
        stats[src] = dict(n=len(coords), seconds=round(dt, 1), gpu_seconds=round(t_gpu, 1),
                          docs_per_s=round(len(coords) / dt))
        print(src, stats[src], flush=True)
    stats["total_seconds"] = round(time.time() - T0, 1)
    stats["total_n"] = sum(v["n"] for k, v in stats.items() if isinstance(v, dict))
    json.dump(stats, open(f"{out}/projection_stats.json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
