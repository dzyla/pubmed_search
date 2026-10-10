"""Step 5: score 2D embeddings on the held-out pool (100k test docs, 5k queries).
  python evaluate.py name1 name2 ...
name forms: ref_<X> (reference layout restricted to pool), pred_<X>_<meth> (test-aligned predictions),
            codes (Hamming kNN in 384-bit space, the information ceiling, no 2D).
Ground truth = exact cosine kNN of the re-embedded float vectors within the pool.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import SCRATCH, WORK, evaluate, hd_neighbors, split

train, test, pool, queries = split()
pos_in_test = np.searchsorted(test, pool)
X = np.load(f"{WORK}/sample_emb_f16.npy", mmap_mode="r")[pool].astype(np.float32)
cache = f"{WORK}/hd_nn_pool.npy"
if os.path.exists(cache):
    hd_nn = np.load(cache)
else:
    hd_nn = hd_neighbors(X, queries, 50); np.save(cache, hd_nn)


def coords(name):
    if name.startswith("ref_"):
        return np.load(f"{WORK}/{name}.npy")[pool]
    return np.load(f"{WORK}/{name}.npy")[pos_in_test]


def ref_of(name):
    # pred_<ref>_<meth> -> ref_<ref>
    body = name[5:]
    refs = sorted([f[4:-4] for f in os.listdir(WORK) if f.startswith("ref_")], key=len, reverse=True)
    for r in refs:
        if body.startswith(r + "_"):
            return r
    return None


def agreement(P, R, k=10):
    """kNN overlap between predicted 2D and reference 2D within the pool + displacement stats."""
    Pt = torch.from_numpy(P).cuda(); Rt = torch.from_numpy(R).cuda()
    rec = []
    for s in range(0, len(queries), 1000):
        qi = torch.as_tensor(queries[s:s + 1000], device="cuda")
        a = torch.cdist(Pt[qi], Pt); a[torch.arange(len(qi)), qi] = float("inf")
        b = torch.cdist(Rt[qi], Rt); b[torch.arange(len(qi)), qi] = float("inf")
        na = a.topk(k, largest=False).indices; nb = b.topk(k, largest=False).indices
        rec.append((na.unsqueeze(2) == nb.unsqueeze(1)).any(2).float().mean(1).cpu())
    return float(torch.cat(rec).mean())


def off_manifold(P, refname, res=512):
    """share of predicted points landing in pixels where the reference (all 1.52M) is ~empty."""
    R = np.load(f"{WORK}/ref_{refname}.npy")
    lo = np.percentile(R, 0.3, 0); hi = np.percentile(R, 99.7, 0)
    h, xe, ye = np.histogram2d(R[:, 0], R[:, 1], bins=res, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
    ix = np.clip(((P[:, 0] - lo[0]) / (hi[0] - lo[0]) * res).astype(int), 0, res - 1)
    iy = np.clip(((P[:, 1] - lo[1]) / (hi[1] - lo[1]) * res).astype(int), 0, res - 1)
    return float((h[ix, iy] <= 1).mean())


_km = {}


def region_agreement(P, refname, k=100):
    """share of held-out docs whose predicted point falls in the same k-means region (k=100,
    fitted on the reference layout) as its reference position; + median displacement in % of diagonal."""
    from sklearn.cluster import KMeans
    R = np.load(f"{WORK}/ref_{refname}.npy")
    if refname not in _km:
        _km[refname] = KMeans(k, n_init=2, random_state=0).fit(R[::10])
    km = _km[refname]
    Rt = R[test]
    diag = np.linalg.norm(np.percentile(R, 99.7, 0) - np.percentile(R, 0.3, 0))
    return float((km.predict(P) == km.predict(Rt)).mean()), float(100 * np.median(np.linalg.norm(P - Rt, axis=1)) / diag)


results = {}
rf = f"{SCRATCH}/metrics.json"
if os.path.exists(rf):
    results = json.load(open(rf))
for name in sys.argv[1:]:
    if name == "codes":
        codes = np.load(f"{WORK}/sample_codes.npy")[pool]
        B = torch.from_numpy(np.unpackbits(codes, axis=1).astype(np.float32) * 2 - 1).cuda()
        rec = {10: [], 50: []}
        for s in range(0, len(queries), 1000):
            qi = torch.as_tensor(queries[s:s + 1000], device="cuda")
            sim = B[qi] @ B.T + torch.rand(len(qi), len(B), device="cuda") * 0.5   # random tie-break
            sim[torch.arange(len(qi)), qi] = -1e9
            nn_ = sim.topk(50).indices
            hn = torch.as_tensor(hd_nn[s:s + 1000], device="cuda")
            for k in rec:
                rec[k].append((nn_[:, :k].unsqueeze(2) == hn[:, :k].unsqueeze(1)).any(2).float().mean(1).cpu())
        r = {f"knn@{k}": float(torch.cat(v).mean()) for k, v in rec.items()}
    else:
        Y = coords(name).astype(np.float32)
        r = evaluate(Y, X, queries, hd_nn=hd_nn)
        if name.startswith("pred_"):
            ref = ref_of(name)
            if ref and ref != "old100k":
                r["agree_ref@10"] = agreement(Y, np.load(f"{WORK}/ref_{ref}.npy")[pool].astype(np.float32))
                r["off_manifold"] = off_manifold(np.load(f"{WORK}/{name}.npy"), ref)
                r["region@100"], r["med_err_pct_diag"] = region_agreement(np.load(f"{WORK}/{name}.npy"), ref)
    results[name] = r
    print(name, {k: round(v, 4) for k, v in r.items()}, flush=True)
json.dump(results, open(rf, "w"), indent=1)
