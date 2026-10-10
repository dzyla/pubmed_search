"""Exact Hamming kNN graph (k=90) of the stored codes with faiss IndexBinaryFlat on CPU."""
import time, numpy as np, faiss
W = "/mnt/h/pubmed_semantic_search/umap_work"
C = np.load(f"{W}/sample_codes.npy"); faiss.omp_set_num_threads(22)
ix = faiss.IndexBinaryFlat(384); ix.add(C)
t = time.time(); D, I = ix.search(C, 91); dt = time.time() - t
# drop self: remove the query's own index wherever it appears (ties may push it off column 0)
keep = I != np.arange(len(C))[:, None]
rows = np.argsort(~keep, axis=1, kind="stable")[:, :90]
I = np.take_along_axis(I, rows, 1).astype(np.int32); D = np.take_along_axis(D, rows, 1)
np.save(f"{W}/knn_binary_I.npy", I); np.save(f"{W}/knn_binary_D.npy", np.sqrt(4 * D / 384).astype(np.float32))
print(f"faiss CPU Hamming kNN 1.52M x 1.52M: {dt:.0f}s")
