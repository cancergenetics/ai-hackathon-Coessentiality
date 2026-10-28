"""v2 metrics: uniform reweighting (hand + replication), HT weighting unbiasedness, weighted P@R, cluster bootstrap
coverage and pairing, Holm."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src import metrics, scorers, splits
from src.config import N_NEG_SCORABLE, N_POS_SCORABLE, NEG_POS_RATIO, SEEDS
from src.metrics import (cluster_bootstrap, holm, ht_weights, uniform_reweight, weighted_ap,
                         weighted_precision_at_recall)


def test_ratio_constant():
    assert NEG_POS_RATIO == pytest.approx(593406 / 3191)
    assert (N_POS_SCORABLE, N_NEG_SCORABLE) == (3191, 593406)


def test_reweighted_ap_hand_computed():
    # ranking: + - + -, 2 pos 2 neg; target ratio 3 -> each negative weight 3 (sum neg 6 = 3 * sum pos 2)
    y, s = np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1])
    w = uniform_reweight(y, ratio=3.0)
    assert list(w) == [1, 3, 1, 3]
    # thresholds: .9 -> TP1 FP0 P=1 R=.5 ; .8 -> P=1/4 R=.5 ; .7 -> TP2 FP3 P=2/5 R=1 ; AP = .5*1 + .5*2/5
    assert weighted_ap(y, s, w) == pytest.approx(0.5 + 0.5 * 0.4)


def test_reweighted_equals_replication():
    rng = np.random.default_rng(SEEDS[0])
    y = (rng.random(4000) < 0.03).astype(int)
    s = np.round(rng.normal(size=4000) + 1.2 * y, 1)          # ties
    k = 7
    ratio = k * (y == 0).sum() / y.sum()                     # = replicate every negative k times
    w = uniform_reweight(y, ratio=ratio)
    assert np.allclose(w[y == 0], k)
    rep = np.r_[np.arange(len(y))[y == 1], np.repeat(np.arange(len(y))[y == 0], k)]
    ap_rep = average_precision_score(y[rep], s[rep])
    assert weighted_ap(y, s, w) == pytest.approx(ap_rep, abs=1e-12)
    assert metrics.PresortedScores(y, s).ap(w) == pytest.approx(ap_rep, abs=1e-12)
    # weighted P@R equals P@R on the replicated data
    a = weighted_precision_at_recall(y, s, w)
    b = metrics.precision_at_recall(y[rep], s[rep])
    for key in a:
        assert a[key] == pytest.approx(b[key], abs=1e-12)


def test_ht_weights():
    w = ht_weights([1, 1, 2], [1, 2, 2], 0.25)
    assert list(w) == [1, 4, 1]


def test_ht_ap_unbiased_over_fold_draws():
    """HT-weighted fold AP averaged over many cluster-grouped fold draws ~ the full-universe AP, while the raw fold
    AP is biased upward (test folds over-represent within-cluster pairs)."""
    from src.toy import make_toy
    u, X, _ = make_toy(seed=SEEDS[0], giant=0)
    s = scorers.make_baseline("gls_signed", scorers._Matrix(X)).bind(u).score(np.arange(len(u)))
    y = u.y
    full = weighted_ap(y, s)
    ht, raw = [], []
    for sd in np.random.SeedSequence(SEEDS[0]).generate_state(30):
        sp = splits.make_split(u.pairs, "cluster", 5, int(sd), clusters=u.cluster_map())
        for k in range(5):
            te = np.flatnonzero(sp.test_mask(k))
            w = ht_weights(u.col("cluster_a")[te], u.col("cluster_b")[te], 0.2)
            ht.append(weighted_ap(y[te], s[te], w))
            raw.append(weighted_ap(y[te], s[te]))
    assert abs(np.mean(ht) - full) / full < 0.03, (np.mean(ht), full)
    assert abs(np.mean(ht) - full) < 0.25 * abs(np.mean(raw) - full)


def _clustered(rng, n_clusters, size=5, tau=0.8):
    """All pairs among n_clusters*size proteins; positives = within-cluster pairs with prob .5; score has a
    cluster random effect on within-cluster pairs (clustered dependence) and protein effects."""
    n = n_clusters * size
    cl = np.repeat(np.arange(n_clusters), size)
    ia, ib = np.triu_indices(n, 1)
    ca, cb = cl[ia], cl[ib]
    within = ca == cb
    y = within & (rng.random(len(ia)) < 0.5)
    u_c = rng.normal(0, tau, n_clusters)
    b = rng.normal(0, 0.5, n)
    s = 1.8 * y + np.where(within, u_c[ca], 0) + b[ia] + b[ib] + rng.normal(size=len(ia))
    return y, s, ca, cb


def test_cluster_bootstrap_coverage_toy():
    """Nominal 95% cluster-bootstrap CI covers the population AP in >= 85% of 60 toy datasets. Truth = AP on a
    10x larger universe with between-cluster pairs down-weighted (HT) to the small universe's between:within mix."""
    rng = np.random.default_rng(SEEDS[1])
    C = 40
    yb, sb, cab, cbb = _clustered(rng, 10 * C)
    truth = weighted_ap(yb, sb, ht_weights(cab, cbb, 10.0))     # between weight 1/10
    cover = 0
    for i in range(60):
        y, s, ca, cb = _clustered(rng, C)
        b = cluster_bootstrap(y, {"s": s}, ca, cb, weighting="raw", n_boot=200, seed=SEEDS[0] + i % 5)
        cover += b["ap"]["s"]["ci_low"] <= truth <= b["ap"]["s"]["ci_high"]
    assert cover / 60 >= 0.85, cover / 60


def test_cluster_bootstrap_weights_and_pairing():
    rng = np.random.default_rng(SEEDS[2])
    y, s, ca, cb = _clustered(rng, 30)
    b0 = cluster_bootstrap(y, {"a": s, "b": s.copy()}, ca, cb, f=0.2, n_boot=100, diffs=[("a", "b")])
    d = b0["diff"]["a-b"]
    assert d["diff"] == 0 and d["ci_low"] == 0 and d["ci_high"] == 0
    assert b0["ap"]["a"]["ap"] == pytest.approx(weighted_ap(y, s, ht_weights(ca, cb, 0.2)))
    better = s + 1.0 * y
    b1 = cluster_bootstrap(y, {"a": better, "b": s}, ca, cb, f=0.2, n_boot=200, diffs=[("a", "b")])
    assert b1["diff"]["a-b"]["ci_low"] > 0 and b1["diff"]["a-b"]["rel_gain"] > 0
    # reweighted mode recomputes negative weights per replicate (point estimate = uniform reweighting)
    b2 = cluster_bootstrap(y, {"a": s}, ca, cb, weighting="reweighted", n_boot=20)
    assert b2["ap"]["a"]["ap"] == pytest.approx(weighted_ap(y, s, uniform_reweight(y)))


def test_holm():
    adj = holm({"x": 0.01, "y": 0.04})
    assert adj["x"] == pytest.approx(0.02) and adj["y"] == pytest.approx(0.04)
    adj = holm({"x": 0.03, "y": 0.02})
    assert adj["y"] == pytest.approx(0.04) and adj["x"] == pytest.approx(0.04)


def test_metric_suite_keys():
    rng = np.random.default_rng(SEEDS[3])
    y, s, ca, cb = _clustered(rng, 20)
    ess = rng.random(len(y)) < 0.3
    out = metrics.metric_suite(y, s, ht_weights(ca, cb, 0.2), ca, cb, ca, cb, ess, ~ess, largest_component=0)
    for k in ("ap_ht", "ap_reweighted", "ap_raw", "auroc_ht", "macro_cluster_ap", "ap_ht_no_largest_component",
              "ap_ht_hard_neg", "ap_ht_mixed", "ht_P@R0.1"):
        assert np.isfinite(out[k]), k
    assert out["n_pos_both_nonessential"] + out["n_pos_mixed"] + out["n_pos_both_essential"] == y.sum()
