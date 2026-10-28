"""Label-free similarity layers (src/layers.py): known-answer toys, controls, symmetry, batch invariance."""
import numpy as np
import pytest
from scipy import stats

from src import baselines
from src.config import SEEDS
from src.layers import (INF, LayerEngine, MutualRank, RowZ, SecondOrder, Spectral, as_config, compute_layer,
                        null_moments, null_pairs, row_z, screening_grid, zscore_layer)


def _toy(seed=SEEDS[0], n=60, G=400, n_conf=3, conf_scale=2.0, modules=20, msize=4, mscale=1.0, nan_genes=0):
    """X lines x genes = heterogeneous gene means + rank-n_conf line confounder + planted modules + noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, G)) + rng.normal(scale=1.0, size=G)
    X += conf_scale * rng.normal(size=(n, n_conf)) @ rng.normal(size=(n_conf, G))
    planted = []
    for m in range(modules):
        g = np.arange(m * msize, (m + 1) * msize)
        X[:, g] += mscale * rng.normal(size=(n, 1))
        planted += [(a, b) for i, a in enumerate(g) for b in g[i + 1:]]
    for j in range(G - nan_genes, G):                       # high-NaN genes (not in background)
        X[rng.random(n) < 0.4, j] = np.nan
    return X, planted


def _pairs(g):
    return np.triu_indices(g, 1)


def _top_frac(score, ia, ib, planted):
    """Fraction of planted pairs among the top-|planted| scored pairs (toy positive control)."""
    P = set(map(tuple, planted))
    top = np.argsort(-score)[: len(P)]
    return np.mean([(ia[t], ib[t]) in P for t in top])


@pytest.fixture(scope="module")
def toy():
    X, planted = _toy()
    return X, planted, LayerEngine(X, block=64)


ALL_KINDS = [Spectral(0, 0.1), Spectral(3, INF), Spectral(0, None, shrink="lw"), Spectral(2, 1.0, standardize=True),
             Spectral(0, 0.0, center="line"), RowZ(Spectral(1, 0.1)), MutualRank(Spectral(1, 0.1)),
             MutualRank(Spectral(1, 0.1), balanced=True), SecondOrder(Spectral(1, 0.1), 2.0, "cos"),
             SecondOrder(Spectral(1, 0.1), 5.0, "jaccard"), SecondOrder(Spectral(1, 0.1), 2.0, "cos", balanced=False)]


# -------------------------------------------------------------------------------- A1 known answers
def test_line_centred_k0_tau0_equals_gls_exactly(toy):
    X, _, eng = toy
    ia, ib = _pairs(80)
    g = baselines.gls_pairs(X, ia, ib)
    r_gls = g["signed"] / np.sqrt(g["df"] + g["signed"] ** 2)
    s = eng.compute(Spectral(0, 0.0, center="line"), ia, ib)
    assert np.allclose(s, r_gls, atol=1e-9)


def test_gene_centred_k0_tau0_ranks_like_gls(toy):
    X, _, eng = toy
    ia, ib = _pairs(80)
    g = baselines.gls_pairs(X, ia, ib)["signed"]
    s = eng.compute(Spectral(0, 0.0), ia, ib)
    assert stats.spearmanr(s, g)[0] > 0.99


def test_tau_inf_k0_equals_pearson(toy):
    X, _, eng = toy
    ia, ib = _pairs(80)
    assert np.allclose(eng.compute(Spectral(0, INF), ia, ib), baselines.pearson_pairs(X, ia, ib)["signed"],
                       atol=1e-10)


def test_tau_inf_topk_equals_pc_removed_pearson(toy):
    X, _, eng = toy
    Xc = X - X.mean(0)
    w, U = np.linalg.eigh(Xc @ Xc.T)
    U3 = U[:, np.argsort(w)[::-1][:3]]
    R = Xc - U3 @ (U3.T @ Xc)
    ia, ib = _pairs(80)
    ref = np.array([np.corrcoef(R[:, a], R[:, b])[0, 1] for a, b in zip(ia, ib)])
    assert np.allclose(eng.compute(Spectral(3, INF), ia, ib), ref, atol=1e-10)


def test_ridge_limits(toy):
    _, _, eng = toy
    ia, ib = _pairs(60)
    assert np.allclose(eng.compute(Spectral(2, 1e8), ia, ib), eng.compute(Spectral(2, INF), ia, ib), atol=1e-6)
    assert np.allclose(eng.compute(Spectral(0, 1e-10), ia, ib), eng.compute(Spectral(0, 0.0), ia, ib), atol=1e-5)


def test_ledoit_wolf_is_ridge_point(toy):
    _, _, eng = toy
    e = eng.eig("gene", False)
    tau = e["lw_alpha"] / (1 - e["lw_alpha"])
    ia, ib = _pairs(50)
    assert 0 < e["lw_alpha"] < 1
    assert np.allclose(eng.compute(Spectral(4, None, shrink="lw"), ia, ib), eng.compute(Spectral(4, tau), ia, ib))


def test_topk_removal_recovers_planted_pairs_under_confounder():
    """Positive control: strong rank-3 line confounder; removing the top-3 PCs recovers planted modules."""
    X, planted = _toy(SEEDS[1], n=120, G=1200, conf_scale=3.0, modules=40, msize=4, mscale=0.8)
    eng = LayerEngine(X, block=256)
    ia, ib = _pairs(300)
    p = _top_frac(eng.compute(Spectral(0, INF), ia, ib), ia, ib, planted)
    k3 = _top_frac(eng.compute(Spectral(3, INF), ia, ib), ia, ib, planted)
    gls = _top_frac(eng.compute(Spectral(0, 0.0), ia, ib), ia, ib, planted)
    assert k3 > p + 0.3 and k3 > 0.8, (p, k3)
    assert gls > p, (p, gls)


# -------------------------------------------------------------------------------- A3 balancing
def test_row_z_invariant_to_row_shift_and_scale():
    rng = np.random.default_rng(SEEDS[2])
    S = rng.normal(size=(5, 50))
    self_pos = np.array([0, 3, -1, 10, 49])
    S2 = S.copy()
    S2[1] = 4.0 * S2[1] + 7.0
    S2[2] -= 3.0
    assert np.allclose(row_z(S, self_pos), row_z(S2, self_pos))
    z = row_z(S, self_pos)
    keep = np.ones(50, bool)
    keep[3] = False
    assert abs(z[1, keep].mean()) < 1e-12 and abs(z[1, keep].std() - 1) < 1e-12


def _full_S(eng, base):
    Z = eng.spectral(base)[0]
    return Z @ Z.T


def test_rowz_matches_bruteforce():
    X, _ = _toy(SEEDS[3], G=300, nan_genes=20)
    eng = LayerEngine(X, block=64)
    base = Spectral(1, 0.1)
    S = _full_S(eng, base)
    bg = eng.bg_idx
    assert len(bg) == 280                                       # high-NaN genes excluded from background
    m = np.array([np.delete(S[a, bg], np.flatnonzero(bg == a)).mean() for a in range(300)])
    s = np.array([np.delete(S[a, bg], np.flatnonzero(bg == a)).std() for a in range(300)])
    ia, ib = _pairs(300)
    ref = 0.5 * ((S[ia, ib] - m[ia]) / s[ia] + (S[ia, ib] - m[ib]) / s[ib])
    out = eng.compute(RowZ(base), ia, ib)
    assert np.isfinite(out).all()                               # high-NaN genes are scored too
    assert np.allclose(out, ref, atol=1e-5)


@pytest.mark.parametrize("balanced", [False, True])
def test_mutual_rank_matches_bruteforce(balanced):
    X, _ = _toy(SEEDS[3], G=300, nan_genes=20)
    eng = LayerEngine(X, block=64)
    base = Spectral(1, 0.1)
    S = _full_S(eng, base)
    if balanced:
        m, s = eng.row_stats(base)
        S = 0.5 * ((S - m[:, None]) / s[:, None] + (S - m[None, :]) / s[None, :])
    bg = eng.bg_idx

    def rank(a, b):
        js = bg[(bg != a) & (bg != b)]
        return 1 + (S[a, js] > S[a, b]).sum()

    ia, ib = _pairs(300)
    sel = np.random.default_rng(0).choice(len(ia), 3000, replace=False)
    ia, ib = ia[sel], ib[sel]
    ref = -np.sqrt([rank(a, b) * rank(b, a) for a, b in zip(ia, ib)])
    out = eng.compute(MutualRank(base, balanced), ia, ib)
    assert np.mean(out == ref) > 0.99 and np.allclose(out, ref, rtol=0.02)   # float32 rows: rare near-ties


# -------------------------------------------------------------------------------- A4 second order
def _shared_neighbour_toy(seed=SEEDS[4], n=100, G=2000):
    """Genes 0 and 1 each share a factor with neighbour genes 2..21 but not with each other."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, G))
    f1, f2 = rng.normal(size=(n, 1)), rng.normal(size=(n, 1))
    X[:, [0]] += 1.2 * f1
    X[:, [1]] += 1.2 * f2
    X[:, 2:22] += f1 + f2
    return X


@pytest.mark.parametrize("measure", ["cos", "jaccard"])
def test_second_order_detects_shared_neighbours(measure):
    X = _shared_neighbour_toy()
    eng = LayerEngine(X, block=256)
    base = Spectral(0, INF)
    direct = eng.compute(base, [0], [1])[0]
    na, nb = null_pairs(eng, n=20000)
    null = eng.compute(SecondOrder(base, 1.0, measure), na, nb)
    s01 = eng.compute(SecondOrder(base, 1.0, measure), [0], [1])[0]
    assert abs(direct) < 0.2                                    # little direct correlation
    assert s01 > np.quantile(null, 0.999), (s01, np.quantile(null, 0.999))


def test_second_order_partner_exclusion_bruteforce(toy):
    _, _, eng = toy
    cfg_c, cfg_j = SecondOrder(Spectral(0, 0.1), 5.0, "cos"), SecondOrder(Spectral(0, 0.1), 5.0, "jaccard")
    K = eng.n_neighbours(5.0)
    idx, val = (x.copy() for x in eng.neighbours(cfg_c.base, True, K))
    ia, ib = _pairs(40)
    oc, oj = eng.compute(cfg_c, ia, ib), eng.compute(cfg_j, ia, ib)
    assert all(np.array_equal(x, y) for x, y in zip((idx, val), eng.neighbours(cfg_c.base, True, K)))  # cache intact
    for t, (a, b) in enumerate(zip(ia, ib)):
        va = {j: max(v, 0) for j, v in zip(idx[a], val[a]) if j != b}
        vb = {j: max(v, 0) for j, v in zip(idx[b], val[b]) if j != a}
        assert a not in va and b not in vb
        dot = sum(va[j] * vb[j] for j in set(va) & set(vb))
        den = np.sqrt(sum(v * v for v in va.values()) * sum(v * v for v in vb.values()))
        assert oc[t] == pytest.approx(dot / den, abs=1e-9)
        assert oj[t] == pytest.approx(len(set(va) & set(vb)) / len(set(va) | set(vb)), abs=1e-12)


# -------------------------------------------------------------------------------- generic properties
@pytest.mark.parametrize("cfg", ALL_KINDS, ids=lambda c: c.name)
def test_symmetry(toy, cfg):
    _, _, eng = toy
    ia, ib = _pairs(120)
    assert np.array_equal(eng.compute(cfg, ia, ib), eng.compute(cfg, ib, ia))


@pytest.mark.parametrize("cfg", ALL_KINDS, ids=lambda c: c.name)
def test_batch_invariance(cfg):
    """A pair's score is identical whether computed alone, in another order, or among other pairs
    (fresh engines, so no shared cache state)."""
    X, _ = _toy(nan_genes=10)
    ia, ib = _pairs(120)
    rng = np.random.default_rng(SEEDS[0])
    full = LayerEngine(X, block=64).compute(cfg, ia, ib)
    sub = rng.choice(len(ia), 500, replace=False)
    extra_a, extra_b = _pairs(400)
    mixed_a, mixed_b = np.r_[extra_a[-3000:], ia[sub]], np.r_[extra_b[-3000:], ib[sub]]
    mixed = LayerEngine(X, block=64).compute(cfg, mixed_a, mixed_b)[-500:]
    single = np.array([LayerEngine(X, block=64).compute(cfg, ia[s:s + 1], ib[s:s + 1])[0] for s in sub[:5]])
    assert np.array_equal(mixed, full[sub])
    assert np.array_equal(single, full[sub[:5]])


def test_config_roundtrip_and_grid():
    layers, bases = screening_grid()
    names = [c.name for c in layers]
    assert len(names) == len(set(names))
    for c in layers:
        assert as_config(c.to_dict()) == c
    assert Spectral(0, INF) in bases


def test_null_pairs_and_zscore(toy):
    _, _, eng = toy
    a, b = null_pairs(eng, n=5000)
    a2, b2 = null_pairs(eng, n=5000)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)
    assert (a < b).all() and eng.bg[a].all() and eng.bg[b].all()
    mo = null_moments(Spectral(0, 0.1), eng, pairs=(a, b))
    z = zscore_layer("x", eng.compute(Spectral(0, 0.1), a, b), mo)
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9


def test_compute_layer_accepts_engine_and_dict(toy):
    _, _, eng = toy
    cfg = RowZ(Spectral(2, 1.0))
    ia, ib = _pairs(30)
    assert np.array_equal(compute_layer(cfg.to_dict(), ia, ib, engine=eng), eng.compute(cfg, ia, ib))
    with pytest.raises(ValueError):
        eng.compute(cfg, [1], [1])
