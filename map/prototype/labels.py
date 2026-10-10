"""Topic labels for map regions: k-means on 2D coords, then the most *distinctive* terms per region
(MeSH descriptors for PubMed, arXiv categories, bioRxiv/medRxiv subject category, CT conditions).
score(t, c) = p(t|c) * log(p(t|c) / p(t)),  requiring p(t|c) >= 4 %.
  python labels.py <coords.npy> <k> <out.json>    (coords row-aligned with sample_meta.parquet)
"""
import json
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

WORK = "/mnt/h/pubmed_semantic_search/umap_work"
STOP = {"Humans", "Male", "Female", "Animals", "Adult", "Middle Aged", "Aged", "Aged, 80 and over",
        "Young Adult", "Adolescent", "Child", "Retrospective Studies", "Prospective Studies",
        "Treatment Outcome", "Time Factors", "Risk Factors", "Follow-Up Studies", "Cross-Sectional Studies",
        "Surveys and Questionnaires", "Cohort Studies", "Reproducibility of Results", "Molecular Sequence Data",
        "Cells, Cultured", "Cell Line", "Kinetics", "Rats", "Mice", "Infant", "Child, Preschool",
        "Sensitivity and Specificity", "Incidence", "Prevalence", "Severity of Illness Index", "Healthy Volunteers"}
ARXIV = {"cs.CV": "Computer vision", "cs.LG": "Machine learning", "cs.CL": "NLP", "cs.AI": "AI",
         "cs.RO": "Robotics", "cs.IT": "Information theory", "cs.CR": "Cryptography/security",
         "cs.DS": "Algorithms", "cs.NI": "Networking", "cs.SE": "Software eng.", "cs.DC": "Distributed computing",
         "cs.IR": "Information retrieval", "cs.HC": "HCI", "cs.SY": "Control systems", "eess.SY": "Control systems",
         "eess.SP": "Signal processing", "eess.IV": "Image processing", "eess.AS": "Audio/speech",
         "hep-ph": "Particle phenomenology", "hep-th": "High-energy theory", "hep-ex": "Particle experiment",
         "hep-lat": "Lattice QCD", "gr-qc": "Gravitation/cosmology", "quant-ph": "Quantum physics",
         "astro-ph": "Astrophysics", "astro-ph.GA": "Galaxies", "astro-ph.SR": "Stellar astrophysics",
         "astro-ph.HE": "High-energy astrophysics", "astro-ph.CO": "Cosmology", "astro-ph.EP": "Exoplanets",
         "astro-ph.IM": "Astro instrumentation", "nucl-th": "Nuclear theory", "nucl-ex": "Nuclear experiment",
         "cond-mat.mtrl-sci": "Materials science", "cond-mat.mes-hall": "Mesoscale/nano physics",
         "cond-mat.str-el": "Strongly correlated electrons", "cond-mat.stat-mech": "Statistical mechanics",
         "cond-mat.supr-con": "Superconductivity", "cond-mat.soft": "Soft matter",
         "cond-mat.quant-gas": "Quantum gases", "cond-mat.dis-nn": "Disordered systems",
         "physics.optics": "Optics", "physics.flu-dyn": "Fluid dynamics", "physics.plasm-ph": "Plasma physics",
         "physics.soc-ph": "Social physics", "math.AP": "PDEs", "math.CO": "Combinatorics", "math.PR": "Probability",
         "math.AG": "Algebraic geometry", "math.OC": "Optimization", "math.NT": "Number theory",
         "math.NA": "Numerical analysis", "math.DG": "Differential geometry", "math.FA": "Functional analysis",
         "math.GT": "Geometric topology", "math.RT": "Representation theory", "math.DS": "Dynamical systems",
         "math.ST": "Statistics theory", "stat.ME": "Statistical methods", "stat.ML": "Machine learning",
         "q-bio.NC": "Neuroscience (q-bio)", "q-bio.PE": "Population biology", "q-fin": "Quant. finance",
         "econ.EM": "Econometrics", "math.GR": "Group theory", "math.QA": "Quantum algebra",
         "math-ph": "Mathematical physics", "math.MP": "Mathematical physics", "nlin.CD": "Chaos"}


def terms(row):
    s = row.source
    if s == "pubmed" and row.mesh_terms:
        return {re.sub(r"\s*\[.*?\]", "", t).strip() for t in row.mesh_terms.split(";")} - STOP
    if s == "arxiv" and row.categories:
        c = row.categories.split()[0]
        return {ARXIV.get(c, ARXIV.get(c.split(".")[0], c))}
    if s in ("biorxiv", "medrxiv") and row.category:
        return {row.category.strip().title()}
    if s == "clinicaltrials" and row.conditions:
        return {t.strip() for t in row.conditions.split(";")[:3]}
    return set()


def main(coords_path, k, out):
    Y = np.load(coords_path)
    meta = pd.read_parquet(f"{WORK}/sample_meta.parquet",
                           columns=["source", "mesh_terms", "categories", "category", "conditions"])
    lo = np.percentile(Y, 0.3, 0); hi = np.percentile(Y, 99.7, 0)
    inside = np.all((Y >= lo) & (Y <= hi), 1)
    km = KMeans(k, n_init=3, random_state=0).fit(Y[inside][::3])
    lab = np.full(len(Y), -1); lab[inside] = km.predict(Y[inside])
    T = [terms(r) for r in meta.itertuples()]
    has = np.array([len(t) > 0 for t in T])
    glob = Counter(t for ts in T for t in ts); ng = has.sum()
    labels = []
    for c in range(k):
        idx = np.where((lab == c) & has)[0]
        if len(idx) < 200:
            continue
        cnt = Counter(t for i in idx for t in T[i])
        best = []
        for t, n in cnt.items():
            p = n / len(idx)
            if p < 0.04:
                continue
            best.append((p * np.log(p / (glob[t] / ng)), t))
        best.sort(reverse=True)
        if not best:
            continue
        txt = "\n".join(t for _, t in best[:2])
        pts = Y[lab == c]
        # place label at the densest point near the centroid (median is robust)
        x, y = np.median(pts, 0)
        src = meta.source.values[lab == c]
        labels.append(dict(x=float(x), y=float(y), text=txt, n=int((lab == c).sum()),
                           top_source=pd.Series(src).value_counts().index[0]))
    json.dump(labels, open(out, "w"), indent=1)
    for l in labels:
        print(l["n"], l["top_source"], l["text"].replace("\n", " | "))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
