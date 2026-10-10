"""Rotation-invariant robust aggregators of the k Hamming-nearest reference positions
(the coordinate-wise median snaps x and y separately -> axis-aligned streaks in 50M renders).
  python knn_variants.py <ref>   -> pred_<ref>_knn{k}_{agg}.npy for the test split
aggs: cmed (coordinate-wise median, lower), trim (mean of the k/2 points closest to cmed),
      geomed (Weiszfeld geometric median, 10 iters)
"""
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import WORK, split
from mappers import unpack_gpu


def aggregate(nb, agg):
    """nb: (n,k,2) torch tensor of neighbour positions."""
    if agg == "cmed":
        return nb.median(dim=1).values
    m = nb.median(dim=1).values
    if agg == "trim":
        d = (nb - m[:, None]).norm(dim=2)
        idx = d.topk(nb.shape[1] // 2, dim=1, largest=False).indices
        return nb.gather(1, idx[..., None].expand(-1, -1, 2)).mean(1)
    if agg == "geomed":
        for _ in range(10):
            w = 1 / (nb - m[:, None]).norm(dim=2).clamp_min(1e-3)
            m = (nb * w[..., None]).sum(1) / w.sum(1, keepdim=True)
        return m


def main(ref):
    train, test, _, _ = split()
    codes = np.load(f"{WORK}/sample_codes.npy")
    Y = torch.from_numpy(np.load(f"{WORK}/ref_{ref}.npy")).cuda()
    R = unpack_gpu(torch.from_numpy(codes[train]).cuda()); YR = Y[torch.from_numpy(train).cuda()]
    I = []
    for s in range(0, len(test), 4096):
        q = unpack_gpu(torch.from_numpy(codes[test[s:s + 4096]]).cuda())
        I.append((q @ R.T).topk(20, dim=1).indices)
    I = torch.cat(I)
    for k in (10, 20):
        nb = YR[I[:, :k]]
        for agg in ("cmed", "trim", "geomed"):
            P = aggregate(nb, agg).cpu().numpy().astype(np.float32)
            np.save(f"{WORK}/pred_{ref}_knn{k}_{agg}.npy", P)
            print(ref, k, agg, "saved", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
