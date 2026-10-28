"""Worked example for the explainer: trace two small complexes through every step of zavg_BEF.

Runs the published implementation (mvp/src/coessential) on the raw DepMap file and writes every intermediate
value used in docs/explainer/ to docs/explainer/data/worked_example.json. Uses no pair labels: the two gene
sets (SMC5/6 complex, BBSome) are named by hand as well-known complexes.

Usage: ./.env/bin/python docs/explainer/worked_example.py      (about 4 min, ~6 GB RAM)
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mvp" / "src"))
from coessential.baselines import Baselines  # noqa: E402
from coessential.data import load_depmap  # noqa: E402
from coessential.method import CoessentialityModel  # noqa: E402

SMC56 = ["SMC5", "SMC6", "NSMCE1", "NSMCE2", "NSMCE3", "NSMCE4A", "SLF1", "SLF2"]
BBSOME = ["BBS1", "BBS2", "BBS4", "BBS5", "BBS7", "BBS9", "TTC8", "BBIP1"]
OUT = ROOT / "docs" / "explainer" / "data" / "worked_example.json"


def rank_in_row(row, j, self_idx):
    """1-based rank of column j in a genome-wide score row (higher = better), self excluded."""
    r = np.asarray(row, dtype=np.float64).copy()
    r[self_idx] = -np.inf
    r[~np.isfinite(r)] = -np.inf
    return int((r > r[j]).sum() + 1)


def main():
    t0 = time.time()
    X, genes = load_depmap(ROOT / "datasets" / "DepMAP" / "CRISPRGeneEffect.csv")
    sym = np.array([c.split(" (")[0] for c in genes])
    gi = {s: i for i, s in enumerate(sym)}
    m = CoessentialityModel(X)
    base = Baselines(X)
    G = len(sym)
    out = {"input": {"cell_lines": int(X.shape[0]), "genes": int(G),
                     "nan_fraction_overall": float(np.isnan(X).mean()),
                     "background_genes": int(m.n_bg), "K_neighbours": int(m.K)}}

    # ---- step 1: standardise + covariance + filter
    lam, U, Q, null = m.eig()
    w = m.filter_weights()
    out["filter"] = {
        "eigenvalues_top25": lam[:25].round(4).tolist(),
        "variance_share_top20": float(lam[:20].sum() / lam.sum()),
        "variance_share_top1": float(lam[0] / lam.sum()),
        "n_eigen": int(len(lam)), "n_null_eigen": int(null.sum()),
        "mean_eigenvalue": float(lam.mean()),
        "weights_example": {str(i + 1): float(w[i]) for i in [0, 19, 20, 50, 200, 1000, 1350]},
        "unfiltered_weight_example": {str(i + 1): float(lam[i] ** -0.5) if lam[i] > 0 else None
                                      for i in [20, 50, 200, 1000, 1350]},
    }
    sd = np.nanstd(X, 0)
    out["standardise"] = {s: {"sd": float(sd[gi[s]]), "mean": float(np.nanmean(X[:, gi[s]])),
                              "nan_fraction": float(np.isnan(X[:, gi[s]]).mean())}
                          for s in SMC56 + BBSOME + ["RPL3", "OR1A1"]}
    sds = sd[m.bg_idx]
    out["standardise"]["_genome_sd_quantiles"] = {q: float(np.quantile(sds, float(q))) for q in ["0.1", "0.5", "0.9", "0.99"]}

    # profile excerpt for SMC5 / SMC6: first 12 lines, raw and filtered (first 12 filtered coordinates)
    Z = m.profiles()[0]
    out["profiles"] = {s: {"raw_first12": [None if np.isnan(v) else round(float(v), 3) for v in X[:12, gi[s]]],
                           "filtered_first12": Z[gi[s], :12].round(4).tolist()}
                       for s in ["SMC5", "SMC6", "NSMCE2", "SLF1"]}
    out["profiles"]["_filtered_dim"] = int(Z.shape[1])

    # ---- pairwise tables for both complexes
    nb_idx, nb_val = m.neighbours()
    so_m, so_s = m.so_row_stats()
    f_m, f_s = m.f_row_stats()
    mom = {c: m.null_moments(c) for c in ("B", "E", "F")}
    out["null_moments"] = {c: {"mean": v[0], "sd": v[1]} for c, v in mom.items()}

    def rows_for(gene):
        """Genome-wide score rows of one gene for each method."""
        a = gi[gene]
        others = np.arange(G)
        others = others[others != a]
        ia = np.full(len(others), a)
        r = {}
        for meth in ("pearson", "gls"):
            v = np.full(G, np.nan)
            v[others] = base.score(meth, ia, others)
            r[meth] = v
        zb = m.z_block(np.array([a]))
        for c in ("B", "E", "F"):
            r[c] = zb[c][0]
        r["zavg"] = (zb["B"][0] + zb["E"][0] + zb["F"][0]) / 3
        return r

    def table(members):
        rows_cache = {s: rows_for(s) for s in members}
        recs = []
        for a, b in combinations(members, 2):
            ia, ib = np.array([gi[a]]), np.array([gi[b]])
            rec = {"a": a, "b": b,
                   "pearson": float(base.score("pearson", ia, ib)[0]),
                   "gls_t": float(base.score("gls", ia, ib)[0]),
                   "F": float(m.F(ia, ib)[0]), "SO": float(m.second_order(ia, ib)[0]),
                   "B": float(m.B(ia, ib)[0]), "E": float(m.E(ia, ib)[0])}
            zz, Zc = m.zavg(ia, ib)
            rec.update({f"z_{c}": float(Zc[c][0]) for c in ("B", "E", "F")})
            rec["zavg"] = float(zz[0])
            ranks = {}
            for meth in ("pearson", "gls", "F", "B", "E", "zavg"):
                ra = rank_in_row(rows_cache[a][meth], gi[b], gi[a])
                rb = rank_in_row(rows_cache[b][meth], gi[a], gi[b])
                ranks[meth] = min(ra, rb)
            rec["best_rank_in_row"] = ranks
            recs.append(rec)
        return recs, rows_cache

    smc_tab, smc_rows = table(SMC56)
    bbs_tab, _ = table(BBSOME)
    out["smc56_pairs"] = smc_tab
    out["bbsome_pairs"] = bbs_tab

    # ---- neighbour lists (step 3) for SMC5 and SMC6
    def nlist(s):
        return [{"gene": str(sym[j]), "V": round(float(v), 3)} for j, v in zip(nb_idx[gi[s]], nb_val[gi[s]])]
    out["neighbours"] = {s: nlist(s) for s in ["SMC5", "SMC6", "NSMCE2", "SLF1", "BBS1", "BBS2"]}
    a, b = gi["SMC5"], gi["SMC6"]
    sa, sb = set(nb_idx[a]), set(nb_idx[b])
    shared = sorted(sa & sb - {a, b}, key=lambda j: -min(nb_val[a][list(nb_idx[a]).index(j)], nb_val[b][list(nb_idx[b]).index(j)]))
    out["smc5_smc6_shared"] = {"n_shared": len(shared), "shared": [str(sym[j]) for j in shared],
                               "b_in_a": bool(b in sa), "a_in_b": bool(a in sb)}
    # expected overlap of two random 44-gene lists drawn from the background
    out["random_overlap_expected"] = float(m.K * m.K / m.n_bg)
    out["row_stats"] = {s: {"F_row_mean": float(f_m[gi[s]]), "F_row_sd": float(f_s[gi[s]]),
                            "SO_row_mean": float(so_m[gi[s]]), "SO_row_sd": float(so_s[gi[s]])}
                        for s in ["SMC5", "SMC6", "BBS1", "BBS2"]}

    # ---- a decoy: SMC5's top Pearson partner outside the complex, and where it lands under zavg
    pr = smc_rows["SMC5"]["pearson"].copy()
    pr[[gi[s] for s in SMC56]] = -np.inf
    pr[~np.isfinite(pr)] = -np.inf
    top_p = np.argsort(-pr)[:5]
    dec = []
    for j in top_p:
        rec = {"gene": str(sym[j]), "pearson": float(smc_rows["SMC5"]["pearson"][j]),
               "gls_t": float(smc_rows["SMC5"]["gls"][j]), "zavg": float(smc_rows["SMC5"]["zavg"][j])}
        rec["rank_pearson"] = rank_in_row(smc_rows["SMC5"]["pearson"], j, gi["SMC5"])
        rec["rank_zavg"] = rank_in_row(smc_rows["SMC5"]["zavg"], j, gi["SMC5"])
        dec.append(rec)
    out["smc5_top_pearson_outside_complex"] = dec
    # SMC5's top 15 partners under each score
    for meth in ("pearson", "gls", "zavg"):
        r = smc_rows["SMC5"][meth].copy()
        r[gi["SMC5"]] = -np.inf
        r[~np.isfinite(r)] = -np.inf
        out.setdefault("smc5_top15", {})[meth] = [str(sym[j]) for j in np.argsort(-r)[:15]]
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT} in {out['runtime_s']} s")


if __name__ == "__main__":
    main()
