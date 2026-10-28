"""Independent re-derivation of IFT look #1 (results/ift/look01_2026-10-27.json). Same candidates, no selection.

Logged in results/ift_looks.jsonl as look "1-verify" BEFORE this script was first run.

Does NOT import src/ift.py, src/metrics.py, src/evaluate.py, src/scorers.py, src/universe.py or scripts/final_eval.py.
Project imports: src.data (UniProt -> DepMap mapping only: depmap_columns, build_uniprot_map) and the earlier
verifiers' independent layer code (scripts/verify/verify_phaseA_best.py: F, B, Pearson; verify_zavg_BEF.py: E).

Steps
 1. IFT pair set from the raw label files + committed results/splits/protein_assignment.csv (side, cluster,
    component). IFT pair = both proteins side 'ift'; scorable = both map (src.data) to different DepMap genes.
 2. HT weights: 1 within cluster, 1/f between clusters, f = mapped proteins on IFT side / all mapped proteins.
 3. Scores, two independent routes:
      (a) black-box producer predictor scripts/predict_pairs.py (raw CSV) on a label-free, shuffled IFT pair-ID file
          for zavg:B,E,F, B, gls, pearson, spearman (unmapped -> min - 1 per DESIGN §3.10);
      (b) verifier re-implementation of B, E, F (+ zavg with frozen manifest null moments and with own null moments)
          and pairwise-complete Pearson.
 4. Own weighted AP (ties grouped), HT-AP, raw AP, macro per-cluster AP, AP without the largest component, hard-neg
    AP, essentiality strata, AP including unmapped pairs at the minimum score.
 5. Own paired cluster bootstrap (1,000 resamples, seed BOOT_SEED != 0), within weight m_c, between m_c1*m_c2.

Usage: ./.env/bin/python scripts/verify/verify_ift_look1.py [--skip-predict] [--skip-own]
Output: results/verify/ift_look1.json  (pair-level files stay in data_cache/verify/ift_look1/, not committed)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "verify"))
from src import data as D  # noqa: E402  (mapping only)

POS = ROOT / "datasets/HuMAP/Train_positive_pairs.txt"
NEG = ROOT / "datasets/HuMAP/Train_negative_pairs.txt"
ASSIGN = ROOT / "results/splits/protein_assignment.csv"
ASSIGN_SHA = ROOT / "results/splits/protein_assignment.sha256"
LOOK = ROOT / "results/ift/look01_2026-10-27.json"
OUT = ROOT / "results/verify/ift_look1.json"
WORK = ROOT / "data_cache/verify/ift_look1"
CSV = ROOT / "datasets/DepMAP/CRISPRGeneEffect.csv"
BOOT_SEED = 20261027
N_BOOT = 1000
PRED_METHODS = {"zavg_BEF": "zavg:B,E,F", "B": "B", "gls_signed": "gls", "pearson_signed": "pearson",
                "spearman_signed": "spearman"}
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# ------------------------------------------------------------------------------------------------ metrics (own)
def ap_w(y, s, w=None):
    """Weighted average precision, step-wise, tied scores grouped into one threshold."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="stable")
    s, y, w = s[o], y[o], w[o]
    cut = np.flatnonzero(np.diff(s) != 0)
    ends = np.r_[cut, len(s) - 1]
    ctp = np.cumsum(w * y)[ends]
    cfp = np.cumsum(w * (~y))[ends]
    rec = ctp / ctp[-1]
    prec = ctp / (ctp + cfp)
    return float(np.dot(np.diff(np.r_[0.0, rec]), prec))


def sklearn_check():
    """Toy cross-check of ap_w against sklearn (unweighted and weighted)."""
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(1)
    y = rng.random(5000) < 0.05
    s = np.round(rng.normal(size=5000) + y, 1)       # ties on purpose
    w = rng.choice([1.0, 3.7], 5000)
    return dict(unweighted=abs(ap_w(y, s) - average_precision_score(y, s)),
                weighted=abs(ap_w(y, s, w) - average_precision_score(y, s, sample_weight=w)))


def macro_cluster_ap(y, s, ca, cb):
    """Per cluster: within-cluster pairs of that cluster (weight 1); clusters with >=1 pos and >=1 neg."""
    out = {}
    for c in np.unique(ca):
        m = (ca == c) & (cb == c)
        if y[m].sum() > 0 and (~y[m]).sum() > 0:
            out[int(c)] = ap_w(y[m], s[m])
    return float(np.mean(list(out.values()))), len(out)


def macro_cluster_ap_touch(y, s, ca, cb, w):
    """Alternative definition: pairs touching cluster c (HT weights), clusters with >=1 pos."""
    out = {}
    for c in np.unique(np.r_[ca, cb]):
        m = (ca == c) | (cb == c)
        if y[m].sum() > 0:
            out[int(c)] = ap_w(y[m], s[m], w[m])
    return float(np.mean(list(out.values()))), len(out)


class ClusterBoot:
    """Paired cluster bootstrap over the IFT clusters (own implementation)."""

    def __init__(self, y, ca, cb, base_w, scores: dict, n_boot=N_BOOT, seed=BOOT_SEED):
        self.y = np.asarray(y, bool)
        u, inv = np.unique(np.r_[ca, cb], return_inverse=True)
        self.ia, self.ib = inv[:len(ca)], inv[len(ca):]
        self.C = len(u)
        self.within = self.ia == self.ib
        rng = np.random.default_rng(seed)
        self.M = np.stack([np.bincount(rng.integers(0, self.C, self.C), minlength=self.C) for _ in range(n_boot)])
        self.base_w = base_w
        self.prep = {}
        for k, s in scores.items():
            o = np.argsort(-s, kind="stable")
            ss = s[o]
            self.prep[k] = (o, np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1])
        self.ap = {k: self._run(k) for k in scores}

    def _run(self, k):
        o, ends = self.prep[k]
        y = self.y[o]
        res = np.empty(len(self.M))
        for r, m in enumerate(self.M.astype(float)):
            w = (self.base_w * np.where(self.within, m[self.ia], m[self.ia] * m[self.ib]))[o]
            tp = np.cumsum(w * y)[ends]
            fp = np.cumsum(w * ~y)[ends]
            P = tp[-1]
            prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
            res[r] = np.dot(np.diff(np.r_[0.0, tp / P]), prec) if P > 0 else np.nan
        return res

    def diff(self, k1, k2, point1, point2):
        d = self.ap[k1] - self.ap[k2]
        rel = self.ap[k1] / self.ap[k2] - 1
        q = lambda a: [float(np.nanquantile(a, 0.025)), float(np.nanquantile(a, 0.975))]  # noqa: E731
        return dict(diff=point1 - point2, ci=q(d), rel_gain=point1 / point2 - 1, rel_ci=q(rel),
                    frac_boot_le0=float(np.mean(d <= 0)), boot_sd=float(np.nanstd(d)))


# ------------------------------------------------------------------------------------------------ data
def load_ift():
    sha = hashlib.sha256(ASSIGN.read_bytes()).hexdigest()
    assert sha == ASSIGN_SHA.read_text().split()[0], "assignment SHA mismatch"
    parts = []
    for f, lab in ((POS, 1), (NEG, 0)):
        d = pd.read_csv(f, sep="\t", header=None, names=["x", "y"], dtype=str)
        parts.append(pd.DataFrame(dict(a=np.minimum(d.x.values, d.y.values), b=np.maximum(d.x.values, d.y.values),
                                       label=lab)))
    p = pd.concat(parts, ignore_index=True)
    assert not p.duplicated(["a", "b"]).any() and not (p.a == p.b).any()
    cols = D.depmap_columns("CRISPRGeneEffect")
    um = D.build_uniprot_map(pd.unique(p[["a", "b"]].values.ravel()), cols)
    gi = um.gene_idx.where(um.status == "mapped", -1).astype(int).to_dict()
    A = pd.read_csv(ASSIGN).set_index("protein")
    assert set(A.index) == set(um.index)
    # consistency of the committed 'mapped' column with my mapping
    mapped_mine = um.status == "mapped"
    map_mismatch = int((A.mapped.reindex(um.index) != mapped_mine).sum())
    p["ia"], p["ib"] = p.a.map(gi).values, p.b.map(gi).values
    p["sa"], p["sb"] = p.a.map(A.side).values, p.b.map(A.side).values
    p["ca"], p["cb"] = p.a.map(A.cluster).values.astype(int), p.b.map(A.cluster).values.astype(int)
    p["compa"], p["compb"] = p.a.map(A.component).values, p.b.map(A.component).values
    p["scorable"] = (p.ia >= 0) & (p.ib >= 0) & (p.ia != p.ib)
    ift_all = p[(p.sa == "ift") & (p.sb == "ift")].reset_index(drop=True)
    n_mapped = int(mapped_mine.sum())
    n_mapped_ift = int((mapped_mine & (A.side.reindex(um.index) == "ift")).sum())
    info = dict(assignment_sha256=sha, n_label_pairs=len(p), n_pos_all=int(p.label.sum()),
                n_proteins=len(um), n_mapped=n_mapped, n_mapped_ift=n_mapped_ift, f=n_mapped_ift / n_mapped,
                assignment_mapped_col_mismatch=map_mismatch,
                n_cross_pairs_dropped=int(((p.sa == "ift") != (p.sb == "ift")).sum()),
                n_dev_scorable=dict(pos=int(((p.sa == "dev") & (p.sb == "dev") & p.scorable & (p.label == 1)).sum()),
                                    neg=int(((p.sa == "dev") & (p.sb == "dev") & p.scorable & (p.label == 0)).sum())))
    return ift_all, info, list(cols)


def common_essential(X):
    obs = (~np.isnan(X)).sum(0)
    return ((X < -0.5).sum(0) / np.maximum(obs, 1)) >= 0.5


# ------------------------------------------------------------------------------------------------ scores
def run_predictor(ift_all, skip):
    WORK.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(BOOT_SEED).permutation(len(ift_all))   # label-free, shuffled
    ids = ift_all.iloc[order][["a", "b"]]
    pf = WORK / "ift_pair_ids.tsv"
    ids.to_csv(pf, sep="\t", header=False, index=False)

    def one(item):
        name, meth = item
        out = WORK / f"pred_{name}.tsv"
        if not (skip and out.exists()):
            t = time.time()
            r = subprocess.run([sys.executable, str(ROOT / "scripts/predict_pairs.py"), "--pairs", str(pf),
                                "--method", meth, "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(r.stderr[-3000:])
            log(f"predictor {name} done in {time.time() - t:.0f}s")
        return name, out

    with ThreadPoolExecutor(3) as ex:
        outs = dict(ex.map(one, PRED_METHODS.items()))
    key = ift_all.a + "|" + ift_all.b
    res = {}
    for name, out in outs.items():
        d = pd.read_csv(out, sep="\t")
        assert len(d) == len(ift_all)
        kk = np.minimum(d.uniprot_a, d.uniprot_b) + "|" + np.maximum(d.uniprot_a, d.uniprot_b)
        d.index = kk
        d = d.reindex(key)
        assert d.score.notna().all()
        res[name] = dict(score=d.score.values.astype(float), mapped=d.mapped.values.astype(bool))
    return res, hashlib.sha256(pf.read_bytes()).hexdigest()


def own_scores(X, ia, ib):
    import verify_phaseA_best as V
    import verify_zavg_BEF as Z
    bg0 = np.isnan(X).mean(0) < V.MAX_NAN
    na, nb = Z.null_pairs(bg0, 0)
    ev, nm, bg, diag = Z.all_layers(X, ia, ib, na, nb)
    man = {k: Z.manifest_col(*Z.MAN[k])[1] for k in Z.MAN}
    zman = {k: (ev[k] - man[k]["null_mean"]) / man[k]["null_sd"] for k in "BEF"}
    zown = {k: (ev[k] - nm[0][k]["null_mean"]) / nm[0][k]["null_sd"] for k in "BEF"}
    return {"zavg_BEF": (zman["B"] + zman["E"] + zman["F"]) / 3,
            "zavg_BEF_own_null": (zown["B"] + zown["E"] + zown["F"]) / 3,
            "B": ev["B"], "E": ev["E"], "F": ev["F"], "pearson_signed": V.pearson_pairwise(X, ia, ib)}, \
        dict(null_moments_mine={k: nm[0][k] for k in "BEF"},
             null_moments_manifest={k: dict(null_mean=man[k]["null_mean"], null_sd=man[k]["null_sd"]) for k in "BEF"},
             eig=diag.get("E_q0.25_eig513"))


# ------------------------------------------------------------------------------------------------ main
def metric_block(y, s, w, ca, cb, compa, compb, ess_n, largest):
    r = dict(ap_ht=ap_w(y, s, w), ap_raw=ap_w(y, s))
    # primary = pairs touching the cluster, HT weights (this is the definition that reproduces look01; my a-priori
    # reading 'within-cluster pairs only' is kept as a sensitivity variant: only 16 IFT clusters have >=1 within neg)
    r["macro_cluster_ap"], r["n_clusters_macro"] = macro_cluster_ap_touch(y, s, ca, cb, w)
    r["macro_cluster_ap_within_only"], r["n_clusters_within_only"] = macro_cluster_ap(y, s, ca, cb)
    keep = (compa != largest) & (compb != largest)
    r["ap_ht_no_largest_component"] = ap_w(y[keep], s[keep], w[keep])
    r["n_pos_no_largest_component"] = int(y[keep].sum())
    hard = y | (compa == compb)
    r["ap_ht_hard_neg"] = ap_w(y[hard], s[hard], w[hard])
    r["n_hard_neg"] = int((hard & ~y).sum())
    for k, nm in enumerate(("both_nonessential", "mixed", "both_essential")):
        m = ess_n == k
        r[f"ap_ht_{nm}"] = ap_w(y[m], s[m], w[m])
        r[f"n_pos_{nm}"], r[f"n_neg_{nm}"] = int(y[m].sum()), int((~y[m]).sum())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-predict", action="store_true", help="reuse existing predictor outputs")
    ap.add_argument("--skip-own", action="store_true", help="skip the own B/E/F re-implementation")
    args = ap.parse_args()
    look = json.loads(LOOK.read_text())
    res = dict(script="scripts/verify/verify_ift_look1.py", look="1-verify", boot_seed=BOOT_SEED, n_boot=N_BOOT,
               ap_vs_sklearn_abs_err=sklearn_check())

    ift_all, info, cols = load_ift()
    sc = ift_all[ift_all.scorable].reset_index(drop=True)
    y = sc.label.values.astype(bool)
    f = info["f"]
    w = np.where(sc.ca.values == sc.cb.values, 1.0, 1.0 / f)
    clusters = np.unique(np.r_[sc.ca.values, sc.cb.values])
    largest = int(pd.read_csv(ASSIGN).query("component >= 0").groupby("component").size().idxmax())
    info.update(ift_pos=int(y.sum()), ift_neg=int((~y).sum()), ift_clusters=len(clusters),
                n_pos_between_clusters=int((y & (sc.ca.values != sc.cb.values)).sum()),
                ht_prevalence=float((w * y).sum() / w.sum()), raw_prevalence=float(y.mean()),
                n_unmapped_ift_pos=int(ift_all[~ift_all.scorable].label.sum()),
                n_unmapped_ift_neg=int((ift_all[~ift_all.scorable].label == 0).sum()),
                largest_component=largest)
    exp = dict(ift_pos=847, ift_neg=40786, ift_clusters=57, f=look["ht_f"],
               n_unmapped_ift_pos=9, n_unmapped_ift_neg=2352)
    info["checks"] = {k: dict(mine=info[k], expected=v, ok=bool(np.isclose(info[k], v, rtol=0, atol=1e-12)))
                      for k, v in exp.items()}
    res["counts"] = info
    log(f"counts: {json.dumps(info['checks'])}")

    X = pd.read_csv(CSV, index_col=0)
    assert list(X.columns) == cols
    X = X.values.astype(np.float64)
    ess = common_essential(X)
    ess_n = ess[sc.ia.values].astype(int) + ess[sc.ib.values].astype(int)
    compa, compb = sc.compa.values, sc.compb.values
    ca, cb = sc.ca.values, sc.cb.values

    # (a) black-box predictor
    pred, pf_sha = run_predictor(ift_all, args.skip_predict)
    res["pair_id_file_sha256"] = pf_sha
    smask = ift_all.scorable.values
    for k, v in pred.items():
        assert np.array_equal(v["mapped"], smask), f"{k}: predictor mapped flag != my scorable flag"
    S_pred = {k: v["score"][smask] for k, v in pred.items()}

    # (b) own re-implementation
    S_own = {}
    if not args.skip_own:
        S_own, odiag = own_scores(X, sc.ia.values, sc.ib.values)
        res["own_layer_diag"] = odiag
        res["own_vs_predictor_scores"] = {}
        for k in ("zavg_BEF", "B", "pearson_signed"):
            a, b = S_own[k], S_pred[k]
            res["own_vs_predictor_scores"][k] = dict(
                pearson_r=float(np.corrcoef(a, b)[0, 1]), max_abs_diff=float(np.max(np.abs(a - b))),
                spearman_r=float(pd.Series(a).rank().corr(pd.Series(b).rank())))
        log(f"own vs predictor: {res['own_vs_predictor_scores']}")

    # metrics
    M = {}
    for src_name, S in (("predictor", S_pred), ("own", S_own)):
        for k, s in S.items():
            M[f"{k}[{src_name}]"] = metric_block(y, s, w, ca, cb, compa, compb, ess_n, largest)
    # with unmapped pairs at the minimum score (predictor rule: min - 1). Unmapped proteins have cluster -1 in the
    # assignment. Primary (reproduces look01): every pair touching an unmapped protein is between-cluster (1/f).
    # Sensitivity: cluster -1 treated as one cluster, so unmapped x unmapped pairs get weight 1.
    ya = ift_all.label.values.astype(bool)
    wa = np.where((ift_all.ca.values == ift_all.cb.values) & (ift_all.ca.values >= 0), 1.0, 1.0 / f)
    wa_one = np.where(ift_all.ca.values == ift_all.cb.values, 1.0, 1.0 / f)
    for k, v in pred.items():
        M[f"{k}[predictor]"]["ap_ht_with_unmapped_min_score"] = ap_w(ya, v["score"], wa)
        M[f"{k}[predictor]"]["ap_ht_with_unmapped_min_score_alt_minus1_one_cluster"] = ap_w(ya, v["score"], wa_one)
    res["metrics"] = M

    # side-by-side with look01
    keys = ("ap_ht", "ap_raw", "macro_cluster_ap", "ap_ht_no_largest_component", "ap_ht_hard_neg",
            "ap_ht_both_nonessential", "ap_ht_mixed", "ap_ht_both_essential", "ap_ht_with_unmapped_min_score",
            "n_clusters_macro", "n_pos_no_largest_component", "n_hard_neg", "n_pos_both_nonessential",
            "n_neg_both_nonessential", "n_pos_mixed", "n_pos_both_essential")
    cmp_ = {}
    for k in PRED_METHODS:
        mine, rep = M[f"{k}[predictor]"], look["metrics"][k]
        cmp_[k] = {m: dict(mine=mine[m], reported=rep[m], abs_diff=abs(mine[m] - rep[m])) for m in keys if m in rep}
    res["compare_vs_look01"] = cmp_
    res["max_abs_diff_vs_look01"] = max(v["abs_diff"] for d in cmp_.values() for v in d.values())
    log(f"max abs diff vs look01: {res['max_abs_diff_vs_look01']:.2e}")

    # bootstrap
    t = time.time()
    bsc = {k: S_pred[k] for k in PRED_METHODS}
    if S_own:
        bsc["zavg_BEF_own"] = S_own["zavg_BEF"]
    cb_ = ClusterBoot(y, ca, cb, w, bsc)
    pt = {k: M[f"{k}[predictor]"]["ap_ht"] for k in PRED_METHODS}
    if S_own:
        pt["zavg_BEF_own"] = M["zavg_BEF[own]"]["ap_ht"]
    boot = {}
    pairs = [("zavg_BEF", b) for b in ("gls_signed", "spearman_signed", "pearson_signed")] + \
            [("B", b) for b in ("gls_signed", "spearman_signed", "pearson_signed")] + [("zavg_BEF", "B")]
    if S_own:
        pairs += [("zavg_BEF_own", b) for b in ("gls_signed", "spearman_signed", "pearson_signed")]
    for k1, k2 in pairs:
        d = cb_.diff(k1, k2, pt[k1], pt[k2])
        rk = f"{k1}-{k2}" if f"{k1}-{k2}" in look["bootstrap"]["diff"] else None
        if rk:
            r = look["bootstrap"]["diff"][rk]
            d["reported"] = dict(diff=r["diff"], ci=[r["ci_low"], r["ci_high"]], rel_ci=[r["rel_ci_low"], r["rel_ci_high"]])
        boot[f"{k1}-{k2}"] = d
    res["bootstrap"] = dict(seed=BOOT_SEED, n_boot=N_BOOT, n_clusters=cb_.C, runtime_s=round(time.time() - t, 1),
                            weighting="within m_c, between m_c1*m_c2, times HT base weight", diff=boot,
                            ap_ci={k: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]
                                   for k, v in cb_.ap.items()})
    log("bootstrap: " + json.dumps({k: [round(v["diff"], 4)] + [round(x, 4) for x in v["ci"]] for k, v in boot.items()}))

    # verdict recomputation (§3.7 IFT parts: rel gain >= 10% point estimate; non-inferiority -5%)
    ver = {}
    for c in ("zavg_BEF", "B"):
        for b in ("pearson_signed", "spearman_signed", "gls_signed"):
            mc, mb = M[f"{c}[predictor]"], M[f"{b}[predictor]"]
            ok_ii = mc["ap_ht"] / mb["ap_ht"] - 1 >= 0.10
            ok_iii = all(mc[m] >= 0.95 * mb[m] for m in ("macro_cluster_ap", "ap_ht_no_largest_component"))
            ok_iii_w = all(mc[m] >= 0.95 * mb[m] for m in ("macro_cluster_ap_within_only", "ap_ht_no_largest_component"))
            ver[f"{c}-{b}"] = dict(rel_gain=mc["ap_ht"] / mb["ap_ht"] - 1, gate_ii_ift=bool(ok_ii),
                                   non_inferiority=bool(ok_iii), passes=bool(ok_ii and ok_iii),
                                   macro_rel=mc["macro_cluster_ap"] / mb["macro_cluster_ap"] - 1,
                                   passes_if_macro_within_only=bool(ok_ii and ok_iii_w),
                                   reported_passes=look["verdict"]["pairs"][f"{c}-{b}"]["passes"])
    res["verdict_recomputed"] = ver
    res["verdict_matches_look01"] = all(v["passes"] == v["reported_passes"] for v in ver.values())
    res["runtime_s"] = round(time.time() - T0, 1)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
