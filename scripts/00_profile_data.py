"""Profile raw inputs: HuMAP pair files, ID mapping to DepMap genes, DepMap matrices.
Writes results/profile/data_profile.json. Usage: python scripts/00_profile_data.py"""
import json, os, sys
import numpy as np, pandas as pd
D = "datasets"
out = {}
pos = pd.read_csv(f"{D}/HuMAP/Train_positive_pairs.txt", sep="\t", header=None, names=["a", "b"])
neg = pd.read_csv(f"{D}/HuMAP/Train_negative_pairs.txt", sep="\t", header=None, names=["a", "b"])
for name, df in [("pos", pos), ("neg", neg)]:
    key = df.apply(lambda r: tuple(sorted((r.a, r.b))), axis=1)
    out[name] = dict(rows=len(df), unique_pairs=int(key.nunique()), self_pairs=int((df.a == df.b).sum()),
                     unique_ids=int(pd.unique(df[["a", "b"]].values.ravel()).size),
                     id_examples=list(pd.unique(df[["a", "b"]].values.ravel())[:5]))
    df["key"] = key
out["pos_neg_pair_overlap"] = int(len(set(pos.key) & set(neg.key)))
pid, nid = set(pos[["a","b"]].values.ravel()), set(neg[["a","b"]].values.ravel())
out["ids_pos_only"], out["ids_neg_only"], out["ids_both"] = len(pid-nid), len(nid-pid), len(pid&nid)
# isoform / non-standard IDs
allids = pid | nid
out["ids_with_dash"] = sum("-" in i for i in allids)
# HGNC mapping
h = pd.read_csv(f"{D}/HGNC/hgnc_complete_set.txt", sep="\t", low_memory=False, dtype=str)
h = h[h.status == "Approved"]
m = h[["symbol", "entrez_id", "uniprot_ids", "location"]].dropna(subset=["uniprot_ids"]).copy()
m["uniprot_ids"] = m.uniprot_ids.str.split("|"); m = m.explode("uniprot_ids")
u2e = m.groupby("uniprot_ids").entrez_id.apply(lambda s: sorted(set(s.dropna())))
out["ids_mapped_hgnc"] = int(sum(i in u2e.index for i in allids))
out["ids_multi_entrez"] = int(sum(len(u2e[i]) > 1 for i in allids if i in u2e.index))
ge_cols = pd.read_csv(f"{D}/DepMAP/CRISPRGeneEffect.csv", nrows=0).columns[1:]
dep_entrez = {c.split("(")[-1].rstrip(")") for c in ge_cols}
out["depmap_genes"] = len(ge_cols)
inmap = {i for i in allids if i in u2e.index and any(e in dep_entrez for e in u2e[i])}
out["ids_in_depmap"] = len(inmap)
for name, df in [("pos", pos), ("neg", neg)]:
    ok = df.a.isin(inmap) & df.b.isin(inmap)
    out[name]["pairs_both_in_depmap"] = int(ok.sum())
# degree in positives
deg = pd.Series(list(pos.a) + list(pos.b)).value_counts()
out["pos_degree"] = dict(max=int(deg.max()), median=float(deg.median()), n_deg1=int((deg == 1).sum()))
negdeg = pd.Series(list(neg.a) + list(neg.b)).value_counts()
out["neg_degree"] = dict(max=int(negdeg.max()), median=float(negdeg.median()))
# connected components of positive graph
import scipy.sparse as sp, scipy.sparse.csgraph as csg
ids = sorted(pid); ix = {g: k for k, g in enumerate(ids)}
A = sp.coo_matrix((np.ones(len(pos)), ([ix[a] for a in pos.a], [ix[b] for b in pos.b])), shape=(len(ids),)*2)
nc, lab = csg.connected_components(A, directed=False)
cs = pd.Series(lab).value_counts()
out["pos_components"] = dict(n=int(nc), largest=int(cs.max()), sizes_top10=cs.head(10).tolist())
# DepMap matrices
ge = pd.read_csv(f"{D}/DepMAP/CRISPRGeneEffect.csv", index_col=0)
out["gene_effect"] = dict(shape=list(ge.shape), nan_frac=float(ge.isna().values.mean()),
                          genes_any_nan=int(ge.isna().any().sum()), index_example=list(ge.index[:3]))
sg = pd.read_csv(f"{D}/DepMAP/ScreenGeneEffect.csv", index_col=0)
out["screen_effect"] = dict(shape=list(sg.shape), nan_frac=float(sg.isna().values.mean()),
                            suffixes=pd.Series([i.split(".")[-1] for i in sg.index]).value_counts().to_dict())
cf = pd.read_csv(f"{D}/DepMAP/CRISPRConfounders.csv", index_col=0)
out["confounders"] = dict(shape=list(cf.shape), in_ge=int(cf.index.isin(ge.index).sum()),
                          lib_counts={c: int(cf[c].sum()) for c in cf.columns if c.startswith("Library")})
mdl = pd.read_csv(f"{D}/DepMAP/Model.csv")
out["lineages_in_ge"] = mdl[mdl.ModelID.isin(ge.index)].OncotreeLineage.value_counts().head(30).to_dict()
os.makedirs("results/profile", exist_ok=True)
json.dump(out, open("results/profile/data_profile.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
