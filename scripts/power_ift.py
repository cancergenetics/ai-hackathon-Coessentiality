"""DESIGN v2 §3.2 power check for the IFT, using DEV pairs only (no IFT labels or scores are touched).

Question: with an IFT of the realised size and cluster structure, how often does a candidate whose true
HT-weighted AP is (1+g) x the baseline's pass criterion §3.7(ii) (observed relative gain >= 10% AND 95% paired
cluster-bootstrap CI of dAP excluding 0)? g in {0 (type-I check), 0.10, 0.15, 0.20}.

Simulation (semi-parametric, documented):
  * Baseline score s0 = GLS signed t (Wainberg default variant, label-free) on all dev pairs, rank-normalised
    to z0 (AP is rank-invariant).
  * Candidate z1 = rho*z0 + sqrt(1-rho^2)*eps + delta*y, eps ~ N(0,1) per pair (fixed draw). rho = Pearson
    correlation of rank-normalised GLS and Pearson-r scores over all dev pairs (label-free; a realistic
    between-method agreement); sensitivity rho = 0.9. delta is calibrated by bisection so that the dev-universe
    HT-weighted AP of z1 equals (1+g) x that of z0.
  * Each replicate draws a pseudo-IFT from the DEV clusters with the same stratified procedure as the real IFT
    (src.clusters strata + Kish rejection) and a positive fraction chosen so the expected positive count equals the
    real IFT's (847); pairs = both proteins in drawn clusters. HT weights: between-cluster pairs 1/f with f = drawn
    proteins / all scorable proteins (as on the real IFT).
  * Within each replicate, a paired cluster bootstrap (B resamples) gives the dAP CI exactly as final_eval does.
Output: results/splits/power_ift.json (+ printed table).

Usage: ./.env/bin/python scripts/power_ift.py [--reps 200] [--boot 300] [--jobs 8]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.stats import norm, rankdata  # noqa: E402

from src import clusters as C, metrics, scorers  # noqa: E402
from src.config import N_POS_SCORABLE, RESULTS, SEEDS  # noqa: E402
from src.evaluate import run_record  # noqa: E402
from src.universe import load_dev_universe  # noqa: E402

GAINS = (0.0, 0.10, 0.15, 0.20)
MARGIN = 0.10


def znorm(s):
    return norm.ppf(rankdata(s) / (len(s) + 1))


def calibrate(y, z0, eps, rho, w, gain):
    target = (1 + gain) * metrics.weighted_ap(y, z0, w)
    base = rho * z0 + np.sqrt(1 - rho ** 2) * eps
    lo, hi = -2.0, 6.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if metrics.weighted_ap(y, base + mid * y, w) < target:
            lo = mid
        else:
            hi = mid
    d = (lo + hi) / 2
    return d, metrics.weighted_ap(y, base + d * y, w), target


def one_rep(r, ctx, z1_by_gain, n_boot):
    st, wp, csize, frac, pc_a, pc_b, prot_cluster_size_total, n_scorable = ctx["draw"]
    rng = np.random.default_rng([SEEDS[0], 1000 + r])
    for _ in range(1000):
        side = C._draw(st, wp, csize, rng, frac)
        if C.kish_neff(wp[side == "ift"].values) >= C.MIN_NEFF_RATIO * C.kish_neff(wp.values):
            break
    drawn = set(side.index[side == "ift"])
    m = np.isin(pc_a, list(drawn)) & np.isin(pc_b, list(drawn))
    f = csize[list(drawn)].sum() / n_scorable
    y = ctx["y"][m]
    ca, cb = pc_a[m], pc_b[m]
    out = dict(rep=r, n_pos=int(y.sum()), n_neg=int((~y.astype(bool)).sum()), n_clusters=len(drawn), f=float(f))
    for g, z1 in z1_by_gain.items():
        b = metrics.cluster_bootstrap(y, {"cand": z1[m], "base": ctx["z0"][m]}, ca, cb, f=f, n_boot=n_boot,
                                      seed=SEEDS[0] + r % 5, diffs=[("cand", "base")])
        d = b["diff"]["cand-base"]
        out[f"g{g}"] = dict(rel=d["rel_gain"], ci_low=d["ci_low"], ci_excl0=d["ci_low"] > 0,
                            passed=(d["ci_low"] > 0) and (d["rel_gain"] >= MARGIN))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    u = load_dev_universe()
    y = u.y.astype(np.int8)
    mat = scorers.real_matrix()
    s_gls = scorers.make_baseline("gls_signed", mat).bind(u).score(np.arange(len(u)))
    s_p = scorers.make_baseline("pearson_signed", mat).bind(u).score(np.arange(len(u)))
    z0, zp = znorm(s_gls), znorm(s_p)
    rho_obs = float(np.corrcoef(z0, zp)[0, 1])
    eps = np.random.default_rng(SEEDS[0]).normal(size=len(u))
    w_dev = metrics.ht_weights(u.col("cluster_a"), u.col("cluster_b"), u.f)

    summ = json.loads(C.SUMMARY_FILE.read_text())
    n_ift_pos = summ["pairs"]["ift"]["pos"]
    assign = C.load_assignment()
    dev = assign[assign.mapped & (assign.side == "dev")]
    st = C.strata(u.pairs, dev)
    wp = C._within_pos(u.pairs, dev)
    csize = dev.groupby("cluster").size().reindex(wp.index)
    frac = n_ift_pos / y.sum()
    ctx = dict(y=y, z0=z0, draw=(st, wp, csize, frac, u.col("cluster_a"), u.col("cluster_b"), None,
                                 int(assign.mapped.sum())))
    res = dict(method=__doc__.split("Output:")[0].strip(), rho_obs=rho_obs, real_ift=summ["pairs"]["ift"],
               real_ift_f=summ["protein_fraction_ift"], dev_ap_ht_gls=metrics.weighted_ap(y, z0, w_dev),
               reps=a.reps, boot=a.boot, margin=MARGIN, settings={})
    for rho in (rho_obs, 0.9):
        z1s, cal = {}, {}
        for g in GAINS:
            d, ap1, target = calibrate(y, z0, eps, rho, w_dev, g)
            z1s[g] = rho * z0 + np.sqrt(1 - rho ** 2) * eps + d * y
            cal[str(g)] = dict(delta=d, ap_candidate=ap1, target=target)
        reps = Parallel(n_jobs=a.jobs)(delayed(one_rep)(r, ctx, z1s, a.boot) for r in range(a.reps))
        table = {}
        for g in GAINS:
            k = f"g{g}"
            table[str(g)] = dict(p_pass=float(np.mean([r[k]["passed"] for r in reps])),
                                 p_ci_excludes_0=float(np.mean([r[k]["ci_excl0"] for r in reps])),
                                 median_obs_rel_gain=float(np.median([r[k]["rel"] for r in reps])),
                                 q05_q95_obs_rel_gain=[float(np.quantile([r[k]["rel"] for r in reps], q)) for q in (.05, .95)])
        res["settings"][f"rho={rho:.3f}"] = dict(calibration=cal, power=table,
                                                 pseudo_ift_pos=[int(np.quantile([r["n_pos"] for r in reps], q)) for q in (.05, .5, .95)],
                                                 pseudo_ift_neg_median=int(np.median([r["n_neg"] for r in reps])),
                                                 pseudo_ift_clusters_median=int(np.median([r["n_clusters"] for r in reps])))
        print(f"rho={rho:.3f}", json.dumps(table, indent=1))
    res["run_record"] = run_record(seed=SEEDS[0], counts=dict(dev_pairs=len(u), dev_pos=int(y.sum()),
                                                              n_pos_scorable=N_POS_SCORABLE),
                                   runtime_s=round(time.time() - t0, 1))
    out = RESULTS / "splits" / "power_ift.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
