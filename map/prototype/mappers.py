"""Step 4: learn code -> 2D mappers for a reference layout; predict held-out test points.
  python mappers.py <ref_name> [mlp|mlp_small|mlp_noaug|knn|linear|pca ...]
Writes WORK/pred_<ref>_<method>.npy  (len(test), 2) aligned with common.split() test idx,
and WORK/mapper_<ref>_<method>.pt for MLPs.
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import WORK, codes_pm1, split

DEV = "cuda"
SHIFTS = torch.arange(7, -1, -1, dtype=torch.uint8)


def unpack_gpu(packed_u8):
    """(n,48) uint8 tensor on GPU -> (n,384) ±1 half, same bit order as np.unpackbits."""
    bits = (packed_u8.unsqueeze(2) >> SHIFTS.to(packed_u8.device)) & 1
    return bits.reshape(len(packed_u8), -1).to(torch.float16 if packed_u8.is_cuda else torch.float32) * 2 - 1


class Mapper(nn.Module):
    def __init__(self, d=384, h=1024, depth=3):
        super().__init__()
        layers = [nn.Linear(d, h), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.LayerNorm(h), nn.Linear(h, h), nn.GELU()]
        layers += [nn.LayerNorm(h), nn.Linear(h, 2)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("mu", torch.zeros(2))
        self.register_buffer("sd", torch.ones(2))

    def forward(self, x):            # x: ±1 floats
        return self.net(x) * self.sd + self.mu


def train_mlp(codes_tr, y_tr, h=1024, depth=3, epochs=60, bs=4096, lr=2e-3, flip_p=0.02, seed=0):
    torch.manual_seed(seed)
    m = Mapper(h=h, depth=depth).to(DEV)
    mu, sd = y_tr.mean(0), y_tr.std(0)
    m.mu.copy_(torch.tensor(mu)); m.sd.copy_(torch.tensor(sd))
    C = torch.from_numpy(codes_tr).to(DEV)
    Y = torch.from_numpy(((y_tr - mu) / sd).astype(np.float32)).to(DEV)
    n = len(C)
    steps = epochs * (n // bs)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for s in range(0, n - bs + 1, bs):
            idx = perm[s:s + bs]
            x = unpack_gpu(C[idx]).float()
            if flip_p > 0:   # augmentation: flip near-random bits, mimics sign noise of near-zero dims
                x = torch.where(torch.rand_like(x) < flip_p, -x, x)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = m.net(x)
            loss = nn.functional.smooth_l1_loss(pred.float(), Y[idx], beta=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); step += 1
            tot += loss.item()
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep {ep} loss {tot / (n // bs):.4f} {time.time() - t0:.0f}s", flush=True)
    return m


@torch.no_grad()
def predict(m, codes, bs=65536):
    m.eval()
    out = []
    for s in range(0, len(codes), bs):
        x = unpack_gpu(torch.from_numpy(codes[s:s + bs]).to(DEV)).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(m(x).float().cpu().numpy())
    return np.concatenate(out)


def knn_interp_gpu(codes_tr, y_tr, codes_te, k=10):
    """Hamming kNN via ±1 fp16 matmul on the GPU (dot = 384 - 2*hamming)."""
    R = unpack_gpu(torch.from_numpy(codes_tr).cuda())
    I = np.zeros((len(codes_te), k), np.int64); D = np.zeros((len(codes_te), k), np.float32)
    t0 = time.time()
    for s in range(0, len(codes_te), 2048):
        q = unpack_gpu(torch.from_numpy(codes_te[s:s + 2048]).cuda())
        v, i = (q @ R.T).topk(k, dim=1)
        I[s:s + 2048] = i.cpu().numpy(); D[s:s + 2048] = ((384 - v.float()) / 2).cpu().numpy()
    torch.cuda.synchronize(); dt = time.time() - t0
    nb = y_tr[I]
    w = np.exp(-(D - D[:, :1]) / 8.0)[..., None]
    return (nb * w).sum(1) / w.sum(1), np.median(nb, axis=1), dt


def knn_interp(codes_tr, y_tr, codes_te, k=10):
    import faiss
    faiss.omp_set_num_threads(24)
    index = faiss.IndexBinaryFlat(384)
    index.add(codes_tr)
    t0 = time.time()
    D, I = index.search(codes_te, k)
    dt = time.time() - t0
    nb = y_tr[I]                                     # (n,k,2)
    w = np.exp(-(D - D[:, :1]) / 8.0)[..., None]      # soft weights on Hamming distance
    mean = (nb * w).sum(1) / w.sum(1)
    med = np.median(nb, axis=1)
    return mean, med, dt


def main():
    ref = sys.argv[1]
    methods = sys.argv[2:]
    train, test, pool, _ = split()
    codes = np.load(f"{WORK}/sample_codes.npy")
    Y = np.load(f"{WORK}/ref_{ref}.npy")
    for meth in methods:
        t0 = time.time()
        if meth.startswith("mlp"):
            cfg = dict(mlp=dict(h=1024, depth=3), mlp_small=dict(h=256, depth=2),
                       mlp_noaug=dict(h=1024, depth=3, flip_p=0.0),
                       mlp_big=dict(h=2048, depth=4, epochs=120, lr=1e-3))[meth]
            m = train_mlp(codes[train], Y[train], **cfg)
            ttrain = time.time() - t0
            torch.cuda.synchronize(); t1 = time.time()
            P = predict(m, codes[test])
            torch.cuda.synchronize()
            print(f"{meth}: train {ttrain:.0f}s, infer {len(test) / (time.time() - t1):,.0f} docs/s (GPU)")
            torch.save(m.state_dict(), f"{WORK}/mapper_{ref}_{meth}.pt")
        elif meth == "knn":
            mean, med, dt = knn_interp_gpu(codes[train], Y[train], codes[test])
            np.save(f"{WORK}/pred_{ref}_knn_median.npy", med.astype(np.float32))
            print(f"knn: GPU Hamming search {len(test) / dt:,.0f} docs/s (1.37M refs)")
            P = mean
            meth = "knn_mean"
        elif meth == "linear":
            Xtr = codes_pm1(codes[train]); Xte = codes_pm1(codes[test])
            A = np.c_[Xtr, np.ones(len(Xtr))]
            W, *_ = np.linalg.lstsq(A, Y[train], rcond=None)
            P = np.c_[Xte, np.ones(len(Xte))] @ W
        elif meth == "pca":   # target-free linear baseline
            Xtr = codes_pm1(codes[train[::5]])
            mu = Xtr.mean(0)
            _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
            P = (codes_pm1(codes[test]) - mu) @ Vt[:2].T
        np.save(f"{WORK}/pred_{ref}_{meth}.npy", P.astype(np.float32))
        if meth != "pca":
            err = np.linalg.norm(P - Y[test], axis=1)
            ext = np.linalg.norm(np.percentile(Y, 99, 0) - np.percentile(Y, 1, 0))
            r2 = 1 - ((P - Y[test]) ** 2).sum() / ((Y[test] - Y[test].mean(0)) ** 2).sum()
            print(f"{ref}/{meth}: R2={r2:.4f} median err={np.median(err):.3f} "
                  f"({100 * np.median(err) / ext:.2f}% of map diagonal) p90={np.percentile(err, 90):.3f} "
                  f"t={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
