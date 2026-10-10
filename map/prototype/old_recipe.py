"""Reproduce the OLD recipe (umap_final.py) on today's 384-bit codes, same held-out test set:
100k sample, 0/1 unpacked bits, umap-learn cosine n_neighbors=150 min_dist=0.1 init=pca,
MLP [1024,512,256,128] ReLU->BN->Dropout, Adam 1e-3, MSE, 500 epochs, bs 4096.
Writes WORK/ref_old100k.npy (coords for the 100k fit points, indices in WORK/old100k_idx.npy)
and WORK/pred_old100k_oldmlp.npy for the test set."""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import umap

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import WORK, split

DEV = "cuda"


class ParametricUMAP(nn.Module):  # verbatim architecture from umap_final.py:149-164
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128]):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers += [nn.Linear(in_d, h_d), nn.ReLU(), nn.BatchNorm1d(h_d), nn.Dropout(0.1)]
            in_d = h_d
        layers.append(nn.Linear(in_d, 2))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


def main():
    train, test, pool, _ = split()
    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(train, 100_000, replace=False))
    codes = np.load(f"{WORK}/sample_codes.npy")
    X = np.unpackbits(codes[idx], axis=1).astype(np.float32)          # 0/1 like the old code
    t0 = time.time()
    y = umap.UMAP(n_components=2, metric="cosine", n_neighbors=150, min_dist=0.1,
                  init="pca", n_jobs=-1).fit_transform(X)
    print(f"umap-learn 100k nn=150: {time.time() - t0:.0f}s", flush=True)
    np.save(f"{WORK}/old100k_idx.npy", idx); np.save(f"{WORK}/ref_old100k.npy", y.astype(np.float32))

    m = ParametricUMAP(384).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt = torch.from_numpy(X).to(DEV); Yt = torch.from_numpy(y.astype(np.float32)).to(DEV)
    t0 = time.time()
    m.train()
    for ep in range(500):
        perm = torch.randperm(len(Xt), device=DEV)
        tot = 0
        for s in range(0, len(Xt), 4096):
            b = perm[s:s + 4096]
            loss = nn.functional.mse_loss(m(Xt[b]), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 100 == 0 or ep == 499:
            print(f"ep {ep} train mse {tot / (len(Xt) // 4096 + 1):.4f}", flush=True)
    m.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(np.unpackbits(codes[test], axis=1).astype(np.float32)).to(DEV)
        P = torch.cat([m(Xte[s:s + 65536]) for s in range(0, len(Xte), 65536)]).cpu().numpy()
    np.save(f"{WORK}/pred_old100k_oldmlp.npy", P)
    print(f"old MLP train {time.time() - t0:.0f}s; test pred range {P.min(0)} {P.max(0)}; fit-ref range {y.min(0)} {y.max(0)}")


if __name__ == "__main__":
    main()
