"""Exact cosine kNN graph (k=90) of the 1.52M float sample on the GPU (fp16 matmul, chunked).
Also the same graph in Hamming space of the stored codes (for the binary variants)."""
import sys, time
import numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import WORK
kind = sys.argv[1] if len(sys.argv) > 1 else "float"
K = 90
if kind == "float":
    X = torch.from_numpy(np.load(f"{WORK}/sample_emb_f16.npy")).cuda()
else:
    from mappers import unpack_gpu
    X = unpack_gpu(torch.from_numpy(np.load(f"{WORK}/sample_codes.npy")).cuda()) / np.sqrt(384)
    X = X.half()
n = len(X); I = np.zeros((n, K), np.int32); D = np.zeros((n, K), np.float32)
t0 = time.time()
for s in range(0, n, 4096):
    q = X[s:s + 4096]
    best_v = None
    for c in range(0, n, 400_000):
        sim = (q @ X[c:c + 400_000].T).float()
        if kind != "float":
            sim += torch.rand_like(sim) * 1e-4          # tie-break equal Hamming distances
        v, i = sim.topk(K + 1, dim=1); i += c
        if best_v is None: best_v, best_i = v, i
        else:
            v2 = torch.cat([best_v, v], 1); i2 = torch.cat([best_i, i], 1)
            best_v, o = v2.topk(K + 1, dim=1); best_i = i2.gather(1, o)
    # drop self (first column normally)
    I[s:s + 4096] = best_i[:, 1:].cpu().numpy(); D[s:s + 4096] = np.sqrt(np.clip(2 - 2 * best_v[:, 1:].cpu().numpy(), 0, None))
np.save(f"{WORK}/knn_{kind}_I.npy", I); np.save(f"{WORK}/knn_{kind}_D.npy", D)
print(f"exact kNN {kind} k={K}: {time.time()-t0:.0f}s; self-in-col0 check:", flush=True)
