"""Checks of the all-pairs matrices (outputs/all_pairs/, from mvp/scripts/predict_all_pairs.py).

(a) Values: 2,000 random protein pairs (numpy default_rng(0)) plus 500 pairs with >= 1 and 200 with 2 low-coverage
    genes (rng 1, 2) against the research predictor (src/predict.py, the code behind scripts/predict_pairs.py: raw
    CSV, hash-checked, UniProt input where a gene has an accession that maps uniquely to it, gene indices otherwise) and
    against the mvp per-pair scorer (CoessentialityModel.zavg, behind mvp/scripts/predict_pairs.py); and all pairs of
    results/predictions/genome_wide_top_pairs.tsv.gz whose two genes are protein genes.
    Tolerances: zavg_BEF (float32) 1e-4; z_B/z_E/z_F (float16) 1e-2, and, because float16 cannot hold |z| >= 32 to
    1e-2 (spacing 2^-5), also the float16-aware bound 1e-2 + |z| * 2^-11 (half a float16 ULP) is reported.
(b) Symmetry (lower == upper, bitwise), NaN diagonal, finite off-diagonal, zavg_BEF == mean of the components (within
    float16 rounding).
(c) Top-N of the zavg_BEF matrix restricted to background (low_coverage = 0) genes, N = number of protein x protein
    pairs in the existing genome-wide file, equals that file's protein x protein pair set.

Writes the results into outputs/all_pairs/manifest.json and results/predictions/all_pairs_manifest.json ("checks").
Usage: ./.env/bin/python scripts/verify/check_all_pairs.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mvp" / "src"))
from src import data as rdata  # noqa: E402
from src import predict  # noqa: E402

OUT = ROOT / "outputs" / "all_pairs"
MANIFESTS = [OUT / "manifest.json", ROOT / "results" / "predictions" / "all_pairs_manifest.json"]
EXISTING = ROOT / "results" / "predictions" / "genome_wide_top_pairs.tsv.gz"
TOL32, TOL16, F16_HALF_ULP = 1e-4, 1e-2, 2.0 ** -11
COMP = ("B", "E", "F")


def sample_pairs(n, low):
    """2,000 uniform random pairs (seed 0) + 500 with >= 1 low-coverage gene (seed 1) + 200 with 2 (seed 2)."""
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, n, 4000), rng.integers(0, n, 4000)
    k = np.unique(np.minimum(a, b)[a != b] * n + np.maximum(a, b)[a != b], return_index=True)[1]
    keep = np.sort(k)[:2000]
    lo, hi = np.minimum(a, b)[a != b][keep], np.maximum(a, b)[a != b][keep]
    sets = [("random", lo, hi)]
    L, H = np.flatnonzero(low), np.flatnonzero(~low)
    rng = np.random.default_rng(1)
    x, y = rng.choice(L, 500), rng.choice(np.r_[L, H], 500)
    ok = x != y
    sets.append(("low_cov_any", np.minimum(x, y)[ok], np.maximum(x, y)[ok]))
    rng = np.random.default_rng(2)
    x, y = rng.choice(L, 200), rng.choice(L, 200)
    ok = x != y
    sets.append(("low_cov_both", np.minimum(x, y)[ok], np.maximum(x, y)[ok]))
    return sets


def compare(mats, ia, ib, ref_zavg, ref_z):
    r = {}
    v = mats["zavg_BEF"][ia, ib].astype(np.float64)
    d = np.abs(v - ref_zavg)
    r["zavg_BEF"] = dict(max_abs_diff=float(d.max()), n_fail=int((~(d <= TOL32)).sum()), tol=TOL32)
    for c in COMP:
        v = mats[f"z_{c}"][ia, ib].astype(np.float64)
        d = np.abs(v - ref_z[c])
        f16 = TOL16 + np.abs(ref_z[c]) * F16_HALF_ULP
        r[f"z_{c}"] = dict(max_abs_diff=float(d.max()), n_fail_abs_1e2=int((~(d <= TOL16)).sum()),
                           n_fail_float16_bound=int((~(d <= f16)).sum()),
                           max_abs_ref=float(np.abs(ref_z[c]).max()))
    r["pass"] = (r["zavg_BEF"]["n_fail"] == 0 and all(r[f"z_{c}"]["n_fail_float16_bound"] == 0 for c in COMP))
    r["pass_strict_abs_1e2"] = r["pass"] and all(r[f"z_{c}"]["n_fail_abs_1e2"] == 0 for c in COMP)
    r["n"] = len(ia)
    return r


def main():
    t0 = time.perf_counter()
    mats = {k: np.load(OUT / f"{k}.npy", mmap_mode="r") for k in ("zavg_BEF", "z_B", "z_E", "z_F")}
    tab = pd.read_csv(OUT / "genes.tsv", sep="\t", dtype={"entrez": str})
    n = len(tab)
    low = tab.low_coverage.values.astype(bool)
    gidx = tab.depmap_index.values.astype(np.int64)
    checks = {}

    # ---------------------------------------------------------------- (a) sample pairs vs predict_pairs (research + mvp)
    X, genes, msrc = predict.load_matrix(cache=False)
    scorer = predict.PairScorer(X, genes)
    method = predict.parse_method("zavg:B,E,F")
    sets = sample_pairs(n, low)
    allp = np.unique(np.concatenate([lo * n + hi for _, lo, hi in sets]))
    pa, pb = allp // n, allp % n
    # UniProt input where a gene has an accession mapping uniquely to it (the full predict_pairs path)
    accs = {i: str(u).split("|") for i, u in enumerate(tab.uniprot_ids)}
    umap = rdata.build_uniprot_map(sorted({a for v in accs.values() for a in v}), genes)
    uni_of = {}
    for i, v in accs.items():
        for acc in v:
            if acc in umap.index and umap.gene_idx[acc] == gidx[i]:
                uni_of[i] = acc
                break
    via_uni = np.array([(a in uni_of) and (b in uni_of) for a, b in zip(pa, pb)])
    ref = np.full(len(pa), np.nan)
    refz = {c: np.full(len(pa), np.nan) for c in COMP}
    df = pd.DataFrame(dict(uniprot_a=[uni_of[a] for a in pa[via_uni]], uniprot_b=[uni_of[b] for b in pb[via_uni]]))
    out, rep = predict.score_pairs(df, method, scorer, components=True)
    assert (out.mapped == 1).all(), "unexpected unmapped UniProt pairs"
    ref[via_uni] = out.score.values
    for c in COMP:
        refz[c][via_uni] = out[f"z_{c}"].values
    s, Z, _ = predict.score_genes(scorer, method, gidx[pa[~via_uni]], gidx[pb[~via_uni]], components=True)
    ref[~via_uni] = s
    for c in COMP:
        refz[c][~via_uni] = Z[c]
    research_null = {c: dict(mean=scorer.moments[c]["null_mean"], sd=scorer.moments[c]["null_sd"]) for c in COMP}
    del scorer, X
    from coessential.method import CoessentialityModel
    from coessential import data as mdata
    Xm, _ = mdata.load_depmap(ROOT / "datasets/DepMAP/CRISPRGeneEffect.csv")
    model = CoessentialityModel(Xm)
    del Xm
    mz, mZ = model.zavg(gidx[pa], gidx[pb])
    pos = {k: i for i, k in enumerate(allp)}
    res = {}
    for name, lo, hi in sets:
        ix = np.array([pos[k] for k in lo * n + hi])
        res[name] = dict(
            n_low_coverage_pairs=int((low[lo] | low[hi]).sum()),
            vs_research_predict_pairs=compare(mats, lo, hi, ref[ix], {c: refz[c][ix] for c in COMP}),
            vs_mvp_predict_pairs=compare(mats, lo, hi, mz[ix], {c: mZ[c][ix] for c in COMP}),
            research_vs_mvp_max_abs_diff_zavg=float(np.abs(ref[ix] - mz[ix]).max()))
    res["n_unique_pairs"] = len(allp)
    res["n_scored_via_uniprot_input"] = int(via_uni.sum())
    res["n_scored_via_gene_index"] = int((~via_uni).sum())
    res["research_null_moments"] = research_null
    checks["a_sample_pairs"] = res
    del model

    # ---------------------------------------------------------------- (a) existing genome-wide file
    ex = pd.read_csv(EXISTING, sep="\t", dtype={"entrez_a": str, "entrez_b": str})
    e2i = pd.Series(np.arange(n), index=tab.entrez.values)
    ea, eb = e2i.reindex(ex.entrez_a).values, e2i.reindex(ex.entrez_b).values
    okp = ~(np.isnan(ea) | np.isnan(eb))
    ea, eb = ea[okp].astype(np.int64), eb[okp].astype(np.int64)
    lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
    exp = ex[okp]
    c_ex = compare(mats, lo, hi, exp.zavg_BEF.values, {c: exp[f"z_{c}"].values for c in COMP})
    c_ex.update(n_file_pairs=len(ex), n_non_protein_pairs_skipped=int((~okp).sum()),
                note="file values are rounded to 5 decimals")
    checks["a_existing_genome_wide_file"] = c_ex

    # ---------------------------------------------------------------- (b) symmetry / diagonal / consistency
    sym, diag_nan, offdiag_nonfinite, cons = {}, {}, {}, 0.0
    step = 1024
    for k, M in mats.items():
        ok, nf = True, 0
        for i0 in range(0, n, step):
            i1 = min(i0 + step, n)
            R = np.asarray(M[i0:i1])
            C = np.asarray(M[:, i0:i1]).T
            ok &= bool(np.array_equal(R, C, equal_nan=True))
            sub = R.copy()
            sub[np.arange(i1 - i0), np.arange(i0, i1)] = 0
            nf += int((~np.isfinite(sub)).sum())
        sym[k], offdiag_nonfinite[k] = ok, nf
        diag_nan[k] = bool(np.isnan(np.diagonal(M)).all())
    for i0 in range(0, n, step):
        i1 = min(i0 + step, n)
        za = np.asarray(mats["zavg_BEF"][i0:i1], np.float64)
        zs = [np.asarray(mats[f"z_{c}"][i0:i1], np.float64) for c in COMP]
        zc = sum(zs) / 3.0
        d = np.abs(za - zc)
        # float16 rounding of each component (half ULP; subnormal spacing 2^-24) + float32 rounding of zavg
        bound = (sum(F16_HALF_ULP * np.abs(z) + 2.0 ** -25 for z in zs) / 3.0 + 2.0 ** -24 * np.abs(za)) * (1 + 1e-9)
        cons = max(cons, float(np.nanmax(d / bound)))
    checks["b_symmetry_diagonal"] = dict(
        exact_symmetry=sym, diagonal_all_nan=diag_nan, offdiag_nonfinite=offdiag_nonfinite,
        zavg_vs_mean_of_float16_components_max_ratio_to_rounding_bound=cons,
        **{"pass": all(sym.values()) and all(diag_nan.values()) and not any(offdiag_nonfinite.values())
           and cons <= 1.0})

    # ---------------------------------------------------------------- (c) top-N (background genes) == existing set
    N = len(lo)
    floor = float(exp.zavg_BEF.min()) - 0.01
    ca, cb, cv = [], [], []
    Z = mats["zavg_BEF"]
    bgi = np.flatnonzero(~low)
    for i0 in range(0, n, step):
        i1 = min(i0 + step, n)
        R = np.asarray(Z[i0:i1])
        rows = np.arange(i0, i1)
        m = (np.arange(n)[None, :] > rows[:, None]) & ~low[None, :] & ~low[rows, None] & (R >= floor)
        r, c = np.nonzero(m)
        ca.append(rows[r]); cb.append(c); cv.append(R[r, c])
    ca, cb, cv = map(np.concatenate, (ca, cb, cv))
    o = np.lexsort((cb, ca, -cv.astype(np.float64)))[:N]
    top = set((ca[o] * n + cb[o]).tolist())
    old = set((lo * n + hi).tolist())
    only_new, only_old = top - old, old - top
    cutoff = float(cv[o].min())
    checks["c_top_pairs_reproduced"] = dict(
        N=N, n_background_protein_genes=len(bgi), n_existing_pairs_all_background=int((~low[lo] & ~low[hi]).sum()),
        n_common=len(top & old), n_only_new=len(only_new), n_only_existing=len(only_old),
        cutoff_float32=cutoff, n_candidates_at_floor=len(cv),
        boundary_values_only_new=sorted(float(Z[k // n, k % n]) for k in only_new)[:20],
        boundary_values_only_existing=sorted(float(Z[k // n, k % n]) for k in only_old)[:20],
        **{"pass": len(only_new) == 0 and len(only_old) == 0})
    checks["runtime_s"] = round(time.perf_counter() - t0, 1)
    checks["all_pass"] = bool(all(checks[k]["pass"] for k in ("a_existing_genome_wide_file", "b_symmetry_diagonal",
                                                              "c_top_pairs_reproduced"))
                              and all(v["vs_research_predict_pairs"]["pass"] and v["vs_mvp_predict_pairs"]["pass"]
                                      for k, v in checks["a_sample_pairs"].items() if isinstance(v, dict) and "vs_mvp_predict_pairs" in v))
    for p in MANIFESTS:
        man = json.loads(p.read_text())
        man["checks"] = dict(tool="scripts/verify/check_all_pairs.py", **checks)
        p.write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
