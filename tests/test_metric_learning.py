"""Phase C metric learning (src/metric_learning.py) on toys.

* planted pair structure hidden in a low-variance subspace (plus high-variance nuisance directions): the L2 learned
  metric recovers it and generalises to held-out genes (cluster-grouped CV: test complexes/genes unseen in fit);
* training-label shuffle: the learned metric does not beat its initialisation (identity);
* EmbeddingEngine on a LayerEngine's own profiles reproduces the Phase-A layers (cos / rowz / so / rowz-so);
* identity metric = base cosine; scores are batch/order invariant and symmetric; fit reads training labels only.
"""
import numpy as np
import pandas as pd
import pytest

from src import evaluate, layers, layers_ext
from src import metric_learning as ML
from src.config import SEEDS
from src.universe import PairUniverse

FAST = dict(max_steps=300, eval_every=10, patience=60, lr=1e-2)


def planted(seed=SEEDS[0], n_complex=60, n_single=60, m=60, k=6, amp=2.0, nuis=3.0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(3, 9, n_complex)
    comp = np.r_[np.repeat(np.arange(n_complex), sizes), -1 - np.arange(n_single)]
    G = len(comp)
    S = np.linalg.qr(rng.normal(size=(m, k)))[0].T                  # hidden signal subspace (k x m)
    N = np.linalg.qr(rng.normal(size=(m, 5)))[0].T                  # nuisance directions, large variance
    Z = rng.normal(size=(G, m)) + nuis * rng.normal(size=(G, 5)) @ N
    for c in range(n_complex):
        idx = np.flatnonzero(comp == c)
        Z[idx] += amp * rng.normal(size=(1, k)) @ S
    ia, ib = np.triu_indices(G, 1)
    lab = ((comp[ia] == comp[ib]) & (comp[ia] >= 0)).astype(np.int8)
    names = np.array([f"P{i:04d}" for i in range(G)])
    p = pd.DataFrame(dict(a=names[ia], b=names[ib], label=lab, ia=ia, ib=ib, pair_id=np.arange(len(ia))))
    for s_, col in (("a", ia), ("b", ib)):
        p[f"cluster_{s_}"] = comp[col]
        p[f"comp_{s_}"] = comp[col]
    u = PairUniverse(p, "toy_metric", f=1.0, largest_component=0)
    return u, dict(Z=Z, bg=np.ones(G, bool), name="toy"), S


@pytest.fixture(scope="module")
def toy():
    return planted()


def ap(r):
    return r["summary"]["ap_ht"]["mean"]


def run(u, base, level, **kw):
    sc = ML.MetricScorer(f"t_{level}", level=level, base=base, **{**FAST, **kw})
    return evaluate.run_dev_cv(sc, universe=u, write=False), sc


def test_identity_is_base_cosine(toy):
    u, base, _ = toy
    sc = ML.MetricScorer("id", level="identity", base=base).bind(u)
    sc.fit(np.arange(len(u)))
    idx = np.arange(0, len(u), 7)
    Zn = base["Z"] / np.linalg.norm(base["Z"], axis=1, keepdims=True)
    exp = (Zn[u.col("ia")[idx]] * Zn[u.col("ib")[idx]]).sum(1)
    assert np.allclose(sc.score(idx), exp, atol=1e-10)


def test_l2_recovers_planted_and_generalises(toy):
    u, base, S = toy
    r_id, _ = run(u, base, "identity")
    r_l2, sc = run(u, base, "L2", d=8, lams=(1e-2, 1e-3))
    # cluster-grouped CV: every test pair involves genes of complexes never seen in training
    assert ap(r_l2) > ap(r_id) + 0.15, (ap(r_l2), ap(r_id))
    assert ap(r_l2) > 3 * ap(r_id)
    assert all(f["chosen_steps"] > 0 for f in sc.fit_summary())
    # the learned residual map concentrates on the planted subspace
    B, A = sc.state["B"], sc.state["A"]
    M = np.eye(A.shape[1]) + B @ A
    Mn = M.T @ M
    gain_S = np.trace(S @ Mn @ S.T) / S.shape[0]
    gain_all = np.trace(Mn) / Mn.shape[0]
    assert gain_S > 1.5 * gain_all


def test_l1_runs_and_does_not_collapse(toy):
    u, base, _ = toy
    r_id, _ = run(u, base, "identity")
    r_l1, _ = run(u, base, "L1", lams=(1e-2,))
    assert ap(r_l1) > ap(r_id) - 0.02


def test_infonce_also_learns(toy):
    u, base, _ = toy
    r_id, _ = run(u, base, "identity")
    r, _ = run(u, base, "L2", d=8, loss="infonce", lams=(1e-2,))
    assert ap(r) > ap(r_id) + 0.1


def test_label_shuffle_does_not_beat_init(toy):
    u, base, _ = toy
    r_id, _ = run(u, base, "identity")
    rng = np.random.default_rng(SEEDS[1])
    us = u.with_labels(rng.permutation(u.y))
    sc = ML.MetricScorer("shuf", level="L2", d=8, lams=(1e-2, 1e-3), base=base, **FAST)
    c = evaluate.label_shuffle_control(lambda: ML.MetricScorer("shuf", level="L2", d=8, lams=(1e-2, 1e-3),
                                                               base=base, **FAST), u, mode="train")
    assert c["mean_ap_ht"] <= ap(r_id) + 0.02, (c["mean_ap_ht"], ap(r_id))
    del sc, us


def test_batch_invariance_symmetry_and_label_access(toy):
    u, base, _ = toy
    sp = evaluate.make_dev_split(u, "cluster", SEEDS[0])
    tr, te = np.flatnonzero(sp.train_mask(0)), np.flatnonzero(sp.test_mask(0))
    sc = ML.MetricScorer("b", level="L2", d=8, lams=(1e-2,), base=base, **FAST).bind(u)
    sc.fit(tr)
    s = sc.score(te)
    perm = np.random.default_rng(0).permutation(len(te))
    assert np.allclose(sc.score(te[perm]), s[perm], atol=1e-12)
    assert np.allclose(sc.score(te[:50]), s[:50], atol=1e-12)
    # symmetric: swap a/b
    p2 = u.pairs.copy()
    p2[["a", "b", "ia", "ib", "cluster_a", "cluster_b"]] = p2[["b", "a", "ib", "ia", "cluster_b", "cluster_a"]].values
    sc2 = ML.MetricScorer("b", level="L2", d=8, lams=(1e-2,), base=base, **FAST).bind(PairUniverse(p2, "toy_swap"))
    sc2.state, sc2.fit_log = sc.state, sc.fit_log
    assert np.allclose(sc2.score(te), s, atol=1e-10)
    # held-out labels are never read: NaN outside train -> same fitted state
    lab = u.pairs.label.astype(float).values.copy()
    lab[np.setdiff1d(np.arange(len(u)), tr)] = np.nan
    p3 = u.pairs.copy()
    p3["label"] = lab
    sc3 = ML.MetricScorer("b", level="L2", d=8, lams=(1e-2,), base=base, **FAST).bind(PairUniverse(p3, "toy_nan"))
    sc3.fit(tr)
    assert np.allclose(sc3.score(te), s, atol=1e-6)
    with pytest.raises(ValueError):
        sc3.fit(np.r_[tr, te[:3]])


@pytest.mark.parametrize("q", [2.0, 0.5])
def test_embedding_engine_reproduces_layers(q):
    rng = np.random.default_rng(SEEDS[0])
    X = rng.normal(size=(40, 900)) + rng.normal(size=(40, 3)) @ rng.normal(size=(3, 900))
    X[rng.random(X.shape) < 0.02] = np.nan
    X[:, :20][rng.random((40, 20)) < 0.3] = np.nan                  # some non-background genes
    eng = layers.LayerEngine(X, block=256)
    base = layers.Spectral(2, 0.1, standardize=True)
    Z = eng.spectral(base)[0]
    emb = ML.EmbeddingEngine(Z, eng.bg, block=256)
    ia, ib = np.triu_indices(60, 1)
    so = layers.SecondOrder(base, q=q, measure="cos")
    layers_ext.release()
    ref = {"cos": layers.compute_layer(base, ia, ib, eng), "rowz": layers.compute_layer(layers.RowZ(base), ia, ib, eng),
           "so": layers.compute_layer(so, ia, ib, eng), "rowzso": layers_ext.compute(layers_ext.RowZSO(so), ia, ib, eng)}
    layers_ext.release()
    for out, r in ref.items():
        got = ML.engine_scores(emb, out, ia, ib, q=q)
        assert np.allclose(got, r, atol=1e-5), out
