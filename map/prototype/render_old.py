"""Render the OLD parametric-UMAP projection (Jan 2026, 768-bit snowflake codes) for comparison."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A="/home/dzyla/pubmed_search/snowflake_code/visualization_assets"
S="/tmp/claude-1000/-mnt-h-pubmed-semantic-search/1f9b73a6-18aa-4d42-81e2-9f8f4fde4940/scratchpad/umap_study"
ps={k:np.load(f"{A}/projection_{k}_config.npy",mmap_mode="r") for k in ["pubmed","arxiv","biorxiv","medrxiv"]}
allp=np.concatenate([np.asarray(v) for v in ps.values()])
lo=np.percentile(allp,0.05,axis=0); hi=np.percentile(allp,99.95,axis=0)
print("n",len(allp),"range",lo,hi)
H,xe,ye=np.histogram2d(allp[:,0],allp[:,1],bins=1000,range=[[lo[0],hi[0]],[lo[1],hi[1]]])
print("occupied pixel frac",(H>0).mean(),"top1% pixels hold", np.sort(H.ravel())[::-1][:H.size//100].sum()/H.sum())
fig,ax=plt.subplots(figsize=(12,12*(hi[1]-lo[1])/(hi[0]-lo[0])+1),dpi=120)
ax.imshow(np.log1p(H.T),origin="lower",cmap="inferno",extent=[lo[0],hi[0],lo[1],hi[1]],aspect="equal")
ax.set_title(f"OLD approach: MLP regressed on 100k-pt UMAP, {len(allp)/1e6:.1f}M docs (log density)"); ax.axis("off")
fig.savefig(f"{S}/old_projection_density.png",bbox_inches="tight",facecolor="black"); print("ok")
