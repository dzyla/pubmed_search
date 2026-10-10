"""Shared paths, split and quality metrics (torch/GPU)."""
import numpy as np

WORK = "/mnt/h/pubmed_semantic_search/umap_work"
SCRATCH = "/tmp/claude-1000/-mnt-h-pubmed-semantic-search/1f9b73a6-18aa-4d42-81e2-9f8f4fde4940/scratchpad/umap_study"
N = 1_520_000


def split():
    """Fixed 90/10 train/test split of the 1.52M sample + an eval pool inside test."""
    rng = np.random.default_rng(123)
    perm = rng.permutation(N)
    test = np.sort(perm[:152_000])
    train = np.sort(perm[152_000:])
    pool = np.sort(rng.choice(test, 100_000, replace=False))   # metrics computed inside this pool
    queries = rng.choice(len(pool), 5_000, replace=False)       # positions within pool
    return train, test, pool, queries


def codes_pm1(packed):
    """packed uint8 (n,48) -> float32 (n,384) in {-1,+1}."""
    return np.unpackbits(packed, axis=1).astype(np.float32) * 2 - 1


def hd_neighbors(X, queries, kmax=50, device="cuda"):
    """Exact cosine kNN (within X) for the given query rows. Returns (q, kmax) indices and
    full rank function helper. X: (n,d) float, L2-normalised rows."""
    import torch
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
    Xt = torch.nn.functional.normalize(Xt, dim=1)
    out = []
    for s in range(0, len(queries), 1000):
        q = Xt[queries[s:s + 1000]]
        sim = q @ Xt.T
        sim[torch.arange(len(q)), torch.as_tensor(queries[s:s + 1000], device=device)] = -9
        out.append(sim.topk(kmax, dim=1).indices.cpu().numpy())
    return np.concatenate(out)


def evaluate(Y, X, queries, hd_nn=None, ks=(10, 50), tw_k=10, n_pairs=200_000, device="cuda"):
    """Quality of a 2D embedding Y of the pool against high-D X (float, normalised).
    - kNN recall@k: overlap of 2D kNN with high-D kNN (within the pool)
    - trustworthiness@tw_k (Venna & Kaski), from exact high-D ranks of 2D neighbours
    - spearman: rank corr of high-D cosine distance vs 2D distance over random pairs
    """
    import torch
    from scipy.stats import spearmanr
    n = len(Y)
    if hd_nn is None:
        hd_nn = hd_neighbors(X, queries, max(ks), device)
    Yt = torch.from_numpy(np.ascontiguousarray(Y, dtype=np.float32)).to(device)
    Xt = torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device), dim=1)
    res = {}
    rec = {k: [] for k in ks}
    tw_pen = 0.0
    for s in range(0, len(queries), 1000):
        qi = torch.as_tensor(queries[s:s + 1000], device=device)
        d2 = torch.cdist(Yt[qi], Yt)
        d2[torch.arange(len(qi)), qi] = float("inf")
        nn2 = d2.topk(max(ks), dim=1, largest=False).indices
        hn = torch.as_tensor(hd_nn[s:s + 1000], device=device)
        for k in ks:
            a = nn2[:, :k].unsqueeze(2) == hn[:, :k].unsqueeze(1)
            rec[k].append(a.any(2).float().mean(1).cpu())
        # trustworthiness: high-D rank of each 2D neighbour
        sim = Xt[qi] @ Xt.T
        sim[torch.arange(len(qi)), qi] = -9
        nb = nn2[:, :tw_k]
        s_nb = sim.gather(1, nb)                       # (q,k)
        ranks = (sim.unsqueeze(1) > s_nb.unsqueeze(2)).sum(2) + 1 if len(qi) * n * tw_k < 2e9 else None
        if ranks is None:
            ranks = torch.stack([(sim > s_nb[:, j:j + 1]).sum(1) + 1 for j in range(tw_k)], 1)
        tw_pen += torch.clamp(ranks - tw_k, min=0).sum().item()
    nq = len(queries)
    for k in ks:
        res[f"knn@{k}"] = float(torch.cat(rec[k]).mean())
    res[f"trust@{tw_k}"] = 1 - 2.0 / (nq * tw_k * (2 * n - 3 * tw_k - 1)) * tw_pen
    rng = np.random.default_rng(0)
    a = rng.integers(0, n, n_pairs); b = rng.integers(0, n, n_pairs)
    Xn = Xt.cpu().numpy()
    dh = 1 - np.einsum("ij,ij->i", Xn[a], Xn[b])
    dl = np.linalg.norm(Y[a] - Y[b], axis=1)
    res["spearman"] = float(spearmanr(dh, dl).correlation)
    return res
