"""Revision F13: coverage of the paired cluster bootstrap (src.metrics.cluster_bootstrap, HT weights) on toy data with a
known truth. No real data is read.

Generative model (one simulated dataset ~ the pooled out-of-fold dev design: 190 clusters, ~820 proteins, ~66k pairs)
  * 190 clusters, size m_c = 2 + min(28, floor(LogNormal(0.5, 0.9)))  (mean ~4.3, max 30, a few large clusters).
  * Clusters randomly assigned to K = 5 folds. Pairs: all within-cluster pairs (weight 1); between-cluster pairs
    only when both clusters are in the same fold (inclusion probability 1/K, HT weight K) -- the pooled-OOF design.
  * Within-cluster pairs are positive with probability p_c ~ Beta(3, 2); between-cluster pairs are negative.
  * Scores of two methods A, B (paired): protein effect h_p (SD 0.3, shared by A and B), cluster signal
    a_c ~ N(mu, 0.7^2) with mu = 1.5 (low AP, ~0.07 / 0.16) and mu = 2.5 (AP in the range of the real dev
    results, ~0.3-0.45), both reported; B's gain delta_c = 0.3 + 0.4 [m_c >= 8] + N(0, 0.3^2) (heterogeneous, larger in big clusters);
    pair noise eps_A ~ N(0, 1), eps_B = 0.8 eps_A + 0.6 eta. Positive: a_c (+ delta_c for B) + h_a + h_b + eps;
    within-cluster negative: 0.3 a_c + h_a + h_b + eps; between-cluster: h_a + h_b + eps.
Truth: the population HT-AP of A and B = weighted AP on the concatenation of N_TRUTH independent datasets from the same
process (separate seeds; concatenation keeps the within/between ratio of one dataset, so this converges to the
superpopulation value the per-dataset estimate targets). Δ_true = AP_B − AP_A.
Coverage: N_SIM datasets, each with the paired cluster bootstrap exactly as used in the project (1,000 resamples,
base_w = HT weights, percentile 95% CI); coverage = fraction of CIs that contain the truth (Monte-Carlo SE reported).

Output: results/revision/bootstrap_coverage.json
Usage: ./.env/bin/python scripts/63_bootstrap_coverage.py [--n-sim 400] [--n-truth 400] [--workers 6]
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from src import metrics  # noqa: E402
from src.config import RESULTS, SEEDS  # noqa: E402

K = 5
N_CLUSTERS = 190
N_BOOT = 1000
OUT = RESULTS / "revision" / "bootstrap_coverage.json"
MUS = (1.5, 2.5)


def simulate(rng, mu=1.5, n_clusters=N_CLUSTERS):
    m = 2 + np.minimum(28, np.floor(rng.lognormal(0.5, 0.9, n_clusters)).astype(int))
    fold = rng.integers(0, K, n_clusters)
    a_c = rng.normal(mu, 0.7, n_clusters)
    d_c = 0.3 + 0.4 * (m >= 8) + rng.normal(0, 0.3, n_clusters)
    p_c = rng.beta(3, 2, n_clusters)
    prot_cl = np.repeat(np.arange(n_clusters), m)
    h = rng.normal(0, 0.3, len(prot_cl))
    ia, ib = [], []
    for f in range(K):
        prots = np.flatnonzero(fold[prot_cl] == f)
        i, j = np.triu_indices(len(prots), 1)
        ia.append(prots[i])
        ib.append(prots[j])
    ia, ib = np.concatenate(ia), np.concatenate(ib)
    ca, cb = prot_cl[ia], prot_cl[ib]
    within = ca == cb
    y = within & (rng.random(len(ia)) < p_c[ca])
    eps = rng.normal(0, 1, len(ia))
    eps_b = 0.8 * eps + 0.6 * rng.normal(0, 1, len(ia))
    base = h[ia] + h[ib]
    sig = np.where(y, a_c[ca], np.where(within, 0.3 * a_c[ca], 0.0))
    sA = sig + base + eps
    sB = sig + np.where(y, d_c[ca], 0.0) + base + eps_b
    w = metrics.ht_weights(ca, cb, 1.0 / K)
    return y, dict(A=sA, B=sB), ca, cb, w


def truth(n_truth, seed, mu):
    ys, sa, sb, ws = [], [], [], []
    for j in range(n_truth):
        y, S, _, _, w = simulate(np.random.default_rng([seed, j]), mu)
        ys.append(y), sa.append(S["A"]), sb.append(S["B"]), ws.append(w)
    y, w = np.concatenate(ys), np.concatenate(ws)
    apA = metrics.weighted_ap(y, np.concatenate(sa), w)
    apB = metrics.weighted_ap(y, np.concatenate(sb), w)
    return dict(A=apA, B=apB, delta=apB - apA, n_pairs=int(len(y)), n_datasets=n_truth)


def one(args):
    i, mu = args
    y, S, ca, cb, w = simulate(np.random.default_rng([SEEDS[0], i]), mu)
    b = metrics.cluster_bootstrap(y, S, ca, cb, n_boot=N_BOOT, seed=SEEDS[0] + i, diffs=[("B", "A")], base_w=w)
    d = b["diff"]["B-A"]
    return dict(i=i, n_pairs=int(len(y)), n_pos=int(y.sum()), n_clusters=b["n_clusters"],
                A=[b["ap"]["A"]["ap"], b["ap"]["A"]["ci_low"], b["ap"]["A"]["ci_high"]],
                B=[b["ap"]["B"]["ap"], b["ap"]["B"]["ci_low"], b["ap"]["B"]["ci_high"]],
                delta=[d["diff"], d["ci_low"], d["ci_high"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sim", type=int, default=400)
    ap.add_argument("--n-truth", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.perf_counter()
    out = dict(what="F13: coverage of the paired cluster bootstrap (HT weights, 1,000 resamples, percentile 95% CI) "
                    "on toy data with known truth", nominal=0.95, n_sim=a.n_sim, n_boot=N_BOOT, scenarios={})
    allsims = {}
    for mu in MUS:
        tr = truth(a.n_truth, SEEDS[1], mu)
        print("truth", mu, tr, flush=True)
        with Pool(a.workers) as p:
            sims = p.map(one, [(i, mu) for i in range(a.n_sim)], chunksize=4)
        sc = dict(mu=mu, truth=tr, coverage={})
        for k, t in (("A", tr["A"]), ("B", tr["B"]), ("delta", tr["delta"])):
            est = np.array([s[k] for s in sims])
            cov = (est[:, 1] <= t) & (t <= est[:, 2])
            sc["coverage"][k] = dict(
                coverage=float(cov.mean()), mc_se=float(np.sqrt(cov.mean() * (1 - cov.mean()) / len(cov))),
                miss_below=float((est[:, 2] < t).mean()), miss_above=float((est[:, 1] > t).mean()),
                mean_estimate=float(est[:, 0].mean()), bias=float(est[:, 0].mean() - t),
                sd_estimate=float(est[:, 0].std(ddof=1)), mean_ci_width=float((est[:, 2] - est[:, 1]).mean()))
        sc["dataset_size"] = dict(mean_pairs=float(np.mean([s["n_pairs"] for s in sims])),
                                  mean_pos=float(np.mean([s["n_pos"] for s in sims])))
        out["scenarios"][f"mu{mu}"] = sc
        allsims[f"mu{mu}"] = sims
    out["model"] = __doc__.split("Truth:")[0].strip()
    out["sims"] = allsims
    out["runtime_s"] = round(time.perf_counter() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v["coverage"] for k, v in out["scenarios"].items()}, indent=1))


if __name__ == "__main__":
    main()
