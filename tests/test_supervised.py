"""Phase-B supervised scorers (src/supervised.py) and gene features (src/gene_features.py) on toys.

* planted signal is recovered (LGBM and logistic regression);
* label-shuffle controls (all labels / held fold) give AP ~ prevalence;
* gene-stat identifiability: with a planted per-protein label propensity and protein-identifying gene features,
  a random pair-level split is inflated while a cluster-disjoint split stays ~ prevalence;
* scores are batch/order invariant and symmetric in (a, b);
* fit reads only training labels (held-out labels NaN -> identical model; NaN in train -> error);
* gene features: genome-wide bins (<= 5), symmetric min/max.
"""
import numpy as np
import pandas as pd
import pytest

from src import evaluate, gene_features as gf, metrics, scorers, supervised as S
from src.config import SEEDS
from src.toy import make_toy
from src.universe import PairUniverse

SMALL = dict(allow_small_leaves=True, lgbm_params=dict(min_child_samples=40, n_jobs=2), n_bags=2, neg_ratio=20)


@pytest.fixture(scope="module")
def toy():
    u, X, _ = make_toy(seed=SEEDS[0], n_complex=40, giant=0)
    return u, X


def planted_arrays(u, seed=SEEDS[1], strength=2.0):
    rng = np.random.default_rng(seed)
    n = int(u.col("pair_id").max()) + 1
    y = np.zeros(n)
    y[u.col("pair_id")] = u.y
    return {"signal": strength * y + rng.normal(size=n), "noise1": rng.normal(size=n), "noise2": rng.normal(size=n)}


def ap_mean(r):
    return r["summary"]["ap_ht"]["mean"]


def prevalence(r):
    return float(np.mean([f["ht_prevalence"] for f in r["folds"]]))


@pytest.mark.parametrize("model", ["lgbm", "logreg"])
def test_recovers_planted_signal(toy, model):
    u, _ = toy
    arr = planted_arrays(u)
    sc = S.make_supervised("planted", [S.ArrayFeatures(arr)], model=model, **SMALL)
    r = evaluate.run_dev_cv(sc, universe=u, write=False)
    oracle = evaluate.run_dev_cv(scorers.PrecomputedScorer("oracle", arr["signal"]), universe=u, write=False)
    assert ap_mean(r) > 0.9 * ap_mean(oracle) and ap_mean(r) > 10 * prevalence(r)
    noise = S.make_supervised("noise", [S.ArrayFeatures({k: arr[k] for k in ("noise1", "noise2")}, monotone=0)],
                              model=model, **SMALL)
    rn = evaluate.run_dev_cv(noise, universe=u, write=False)
    assert ap_mean(rn) < 3 * prevalence(rn)
    if model == "lgbm":
        imp = sc.feature_importance()
        assert imp["signal"] > 0.8


def test_baseline_features_close_to_gls(toy):
    u, X = toy
    mat = scorers._Matrix(X)
    sc = S.make_supervised("lgbm_base", [S.BaselineFeatures(["pearson_signed", "gls_signed"], mat)], **SMALL)
    r = evaluate.run_dev_cv(sc, universe=u, write=False)
    g = evaluate.run_dev_cv(scorers.make_baseline("gls_signed", mat), universe=u, write=False)
    assert ap_mean(r) > 0.8 * ap_mean(g)


@pytest.mark.parametrize("mode", ["train", "held"])
def test_label_shuffle_controls(toy, mode):
    u, _ = toy
    arr = planted_arrays(u)
    # monotone=0: a monotone constraint is itself prior knowledge (direction of similarity), which the real-data
    # control reports separately; the pipeline check must not rely on it
    fac = lambda: S.make_supervised("shuf", [S.ArrayFeatures(arr, monotone=0)], **SMALL)  # noqa: E731
    c = evaluate.label_shuffle_control(fac, u, seed=SEEDS[0], mode=mode)
    assert c["mean_ap_ht"] < 2.5 * c["mean_prevalence"]


def propensity_universe(seed=SEEDS[2], n_prot=500, cl_size=6):
    """Labels from a per-protein propensity theta (no pair similarity); proteins grouped into random clusters.
    Gene 'stats' = random per-protein codes (identifying, but carrying no transferable information)."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(scale=1.3, size=n_prot)
    ia, ib = np.triu_indices(n_prot, 1)
    p = 1 / (1 + np.exp(-(-6.0 + theta[ia] + theta[ib])))
    y = (rng.random(len(ia)) < p).astype(np.int8)
    cl = rng.permutation(np.repeat(np.arange(int(np.ceil(n_prot / cl_size))), cl_size)[:n_prot])
    names = np.array([f"P{i:04d}" for i in range(n_prot)])
    pairs = pd.DataFrame(dict(a=names[ia], b=names[ib], label=y, ia=ia, ib=ib, pair_id=np.arange(len(ia)),
                              cluster_a=cl[ia], cluster_b=cl[ib], comp_a=cl[ia], comp_b=cl[ib], side="dev"))
    u = PairUniverse(pairs, "toy_propensity", f=1.0, largest_component=int(np.bincount(cl).argmax()))
    codes = rng.random(size=(n_prot, 2))
    return u, codes


def test_gene_stat_identifiability():
    u, codes = propensity_universe()
    res = {}
    for scheme in ("cluster", "random"):
        sc = S.make_supervised("genes_only", [S.GeneFeatures("raw", table=codes)], **SMALL)
        r = evaluate.run_dev_cv(sc, scheme=scheme, universe=u, write=False)
        res[scheme] = (ap_mean(r), prevalence(r))
    ap_c, prev_c = res["cluster"]
    ap_r, prev_r = res["random"]
    assert ap_c < 1.5 * prev_c                       # unseen proteins: identity features carry nothing
    assert ap_r > 3 * prev_r and ap_r > 3 * ap_c      # seen proteins: propensity memorised -> inflated


def test_batch_order_invariance_and_symmetry(toy):
    u, X = toy
    mat = scorers._Matrix(X)
    stats = gf.gene_stats(X)
    blocks = [S.BaselineFeatures(["pearson_signed", "gls_signed"], mat), S.GeneFeatures("binned", table=stats),
              S.GeneFeatures("raw", table=stats),
              S.ChromFeatures(np.array(list("12") * (X.shape[1] // 2 + 1))[:X.shape[1]],
                              np.array(list("pqq") * X.shape[1])[:X.shape[1]])]
    for model in ("lgbm", "logreg"):
        sc = S.make_supervised("inv", blocks, model=model, **SMALL).bind(u)
        sp = evaluate.make_dev_split(u, "cluster", SEEDS[0])
        sc.fit(np.flatnonzero(sp.train_mask(0)))
        rng = np.random.default_rng(SEEDS[3])
        full = sc.score(np.arange(len(u)))
        idx = rng.choice(len(u), 3000, replace=False)
        perm = rng.permutation(idx)
        assert np.allclose(sc.score(perm), full[perm], rtol=0, atol=1e-10)
        for ch in np.array_split(idx, 9):
            assert np.allclose(sc.score(ch), full[ch], rtol=0, atol=1e-10)
        # (a, b) swapped universe -> identical features and scores
        sw = u.pairs.copy()
        for c1, c2 in (("a", "b"), ("ia", "ib"), ("cluster_a", "cluster_b"), ("comp_a", "comp_b")):
            sw[c1], sw[c2] = u.pairs[c2].values, u.pairs[c1].values
        F2, _, _ = sc.features(PairUniverse(sw, "toy_swapped"))
        assert np.array_equal(F2, sc.F)
        assert np.allclose(sc.predict_features(F2), full, rtol=0, atol=1e-10)


def test_fit_reads_only_training_labels(toy):
    u, X = toy
    mat = scorers._Matrix(X)
    sp = evaluate.make_dev_split(u, "cluster", SEEDS[0])
    tr, te = np.flatnonzero(sp.train_mask(1)), np.flatnonzero(sp.test_mask(1))
    lab = u.pairs.label.to_numpy(dtype=np.float64)
    lab[~sp.train_mask(1)] = np.nan                                   # every non-training label hidden
    p = u.pairs.copy()
    p["label"] = lab
    u_nan = PairUniverse(p, "toy_nan", f=u.f, largest_component=u.largest_component)
    out = {}
    for model in ("lgbm", "logreg"):
        blocks = [S.BaselineFeatures(["pearson_signed", "gls_signed"], mat), S.GeneFeatures("binned", table=gf.gene_stats(X))]
        a = S.make_supervised("m", blocks, model=model, **SMALL).bind(u).fit(tr).score(te)
        b = S.make_supervised("m", blocks, model=model, **SMALL).bind(u_nan).fit(tr).score(te)
        assert np.array_equal(a, b)
        out[model] = metrics.weighted_ap(u.y[te], a)
        with pytest.raises(ValueError):
            S.make_supervised("m", blocks, model=model, **SMALL).bind(u_nan).fit(np.r_[tr, te[:5]])
    assert all(v > 3 * u.y[te].mean() for v in out.values())


def test_design_constraints_enforced():
    with pytest.raises(ValueError):
        S.make_supervised("x", [], lgbm_params=dict(min_child_samples=100))
    with pytest.raises(ValueError):
        S.make_supervised("x", [], lgbm_params=dict(num_leaves=15))
    with pytest.raises(ValueError):
        S.make_supervised("x", [], seed=12345)


def test_monotone_switch(toy):
    u, _ = toy
    arr = planted_arrays(u)
    on = S.make_supervised("m", [S.ArrayFeatures(arr)], **SMALL).bind(u)
    off = S.make_supervised("m", [S.ArrayFeatures(arr)], monotone=False, **SMALL).bind(u)
    assert on.monotone == [1, 1, 1] and off.monotone == [0, 0, 0]


def test_cluster_balanced_weights():
    y = np.array([1, 1, 1, 1, 0, 0, 1])
    ca = np.array([0, 0, 0, 1, 0, 1, 0])
    cb = np.array([0, 0, 0, 1, 1, 1, 1])
    w = S.cluster_balanced_weights(y, ca, cb)
    assert np.allclose(w[y == 0], 1) and np.isclose(w[y == 1].mean(), 1)
    # cluster 0 touches 4 positives, cluster 1 touches 2 -> a cluster-1 positive outweighs a cluster-0 positive
    assert w[3] > w[0] and np.isclose(w[0] / w[3], np.sqrt(2 / 4))
    assert np.isclose(w[6], 0.5 * (w[0] + w[3]))


def test_gene_features_bins_and_symmetry(toy):
    _, X = toy
    st = gf.gene_stats(X)
    assert st.shape == (X.shape[1], len(gf.STATS))
    j = gf.STATS.index("nan_frac")
    assert np.allclose(st[:, j], np.isnan(X).mean(0))
    assert np.allclose(st[:, gf.STATS.index("mean")], np.nanmean(X, 0))
    ed = gf.quantile_edges(st)
    B = gf.bin_stats(st, ed)
    assert B.min() >= 0 and B.max() <= 4 and all(len(e) <= 4 for e in ed)
    m = gf.STATS.index("mean")                                         # ~ equal-mass genome-wide bins
    assert np.all(np.abs(np.bincount(B[:, m]) / len(B) - 0.2) < 0.02)
    with pytest.raises(ValueError):
        gf.quantile_edges(st, n_bins=6)
    rng = np.random.default_rng(SEEDS[0])
    ia, ib = rng.integers(0, X.shape[1], 500), rng.integers(0, X.shape[1], 500)
    F1, cols = gf.pair_features(B, ia, ib)
    F2, _ = gf.pair_features(B, ib, ia)
    assert np.array_equal(F1, F2) and len(cols) == 2 * len(gf.STATS)
    u = gf.signature_uniqueness(B, np.arange(X.shape[1]))
    assert 0 <= u["frac_unique"] <= 1
