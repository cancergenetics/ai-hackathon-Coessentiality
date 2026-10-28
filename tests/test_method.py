"""Known-answer tests of the method and baselines on toy data."""
import dataclasses

import numpy as np
import pytest
from scipy import stats

from coessential.baselines import Baselines
from coessential.cli import top_pairs
from coessential.method import CoessentialityModel
from toy import INF, TOY, all_pairs, auroc, modules, shared_neighbour_toy


@pytest.fixture(scope="module")
def mod():
    X, lab = modules()
    return X, lab, CoessentialityModel(X, TOY)


def test_filter_equals_pearson_when_unfiltered():
    """k = 0, tau = inf: no PC removal, no whitening -> F is the Pearson correlation."""
    X, _ = modules(seed=3)
    ia, ib = all_pairs(X.shape[1])
    for std in (True, False):
        m = CoessentialityModel(X, dataclasses.replace(TOY, k=0, tau=INF, standardize=std))
        C = np.corrcoef(X.T)
        assert np.allclose(m.F(ia, ib), C[ia, ib], atol=1e-10)
    # pairwise-complete Pearson baseline agrees too (no NaN)
    assert np.allclose(Baselines(X).score("pearson", ia, ib), C[ia, ib], atol=1e-10)


def gls_reference(screens):
    """Line-by-line port of Wainberg et al.'s GLS loop (kundajelab/coessentiality); screens: genes x lines."""
    Sigma = np.cov(screens.T)
    L = np.linalg.cholesky(np.linalg.pinv(Sigma))
    warped, w0 = screens @ L, L.sum(axis=0)
    G, df = len(warped), warped.shape[1] - 2
    t = np.empty((G, G))
    for g in range(G):
        Xd = np.stack((w0, warped[g]), axis=1)
        coef, res = np.linalg.lstsq(Xd, warped.T, rcond=None)[:2]
        t[g] = coef[1] / np.sqrt(np.linalg.pinv(Xd.T @ Xd)[1, 1] * res / df)
    return t


def test_gls_baseline_matches_wainberg_loop():
    X, _ = modules(n_lines=40, n_modules=2, size=10, n_noise=80, seed=4)
    ia, ib = all_pairs(X.shape[1])
    t = Baselines(X).score("gls", ia, ib)
    ref = gls_reference(X.T)
    assert np.allclose(t, ref[ia, ib], rtol=1e-6, atol=1e-8)


def test_filter_equals_gls_at_k0_tau0():
    """k = 0, tau = 0 with Wainberg's covariance convention: F = GLS correlation r, hence the same ranking as GLS t."""
    X, _ = modules(n_lines=50, n_modules=3, size=10, n_noise=150, seed=5)
    ia, ib = all_pairs(X.shape[1])
    m = CoessentialityModel(X, dataclasses.replace(TOY, k=0, tau=0.0, center="line", standardize=False))
    r = m.F(ia, ib)
    t = Baselines(X).score("gls", ia, ib)
    df = X.shape[0] - 2
    assert np.allclose(r, t / np.sqrt(df + t * t), atol=1e-8)
    assert stats.spearmanr(r, t).statistic > 1 - 1e-12


def test_second_order_detects_shared_neighbours():
    """a-b share a module of neighbours, c-d only a private factor; both pairs have the same direct correlation."""
    X = shared_neighbour_toy()
    m = CoessentialityModel(X, dataclasses.replace(TOY, tau=INF, q=5))      # K = 21 neighbours
    a, b, c, d = 40, 41, 42, 43
    F = m.F([a, c], [b, d])
    assert abs(F[0] - F[1]) < 0.05                                         # first order cannot tell them apart
    so = m.second_order([a, c], [b, d])
    na, nb = m.null_pairs()
    noise = (na >= 44) & (nb >= 44)
    q99 = np.quantile(m.second_order(na[noise], nb[noise]), 0.99)
    assert so[0] > q99 + 0.3 and so[0] > so[1] + 0.3                      # second order can
    assert so[1] < q99
    # partner exclusion: c and d are each other's neighbours, but that link never enters their second-order score
    idx, _ = m.neighbours()
    assert d in idx[c] and c in idx[d]
    W = m.knn_weights()[0].toarray()
    vc, vd = W[c].copy(), W[d].copy()
    vc[d] = vd[c] = 0.0
    assert np.isclose(so[1], vc @ vd / np.linalg.norm(vc) / np.linalg.norm(vd))


def test_eigenmap_recovers_planted_modules(mod):
    X, lab, m = mod
    ia, ib = all_pairs(X.shape[1])
    inmod = (lab[ia] >= 0) & (lab[ia] == lab[ib])
    e = m.E(ia, ib)
    assert auroc(e[inmod], e[~inmod]) > 0.99
    z, _ = m.zavg(ia, ib)
    assert auroc(z[inmod], z[~inmod]) > 0.99


def test_symmetry(mod):
    X, _, m = mod
    rng = np.random.default_rng(0)
    ia, ib = rng.choice(X.shape[1], 500), rng.choice(X.shape[1], 500)
    ok = ia != ib
    ia, ib = ia[ok], ib[ok]
    for c in "BEF":
        assert np.array_equal(m.component(c, ia, ib), m.component(c, ib, ia))
    bl = Baselines(X)
    for meth in ("pearson", "spearman", "gls"):
        assert np.array_equal(bl.score(meth, ia, ib), bl.score(meth, ib, ia))


def test_batch_and_order_invariance(mod):
    """A pair's score is bit-identical whatever else is requested, in whatever order, from a fresh model."""
    X, _, m = mod
    ia, ib = all_pairs(X.shape[1])
    sel = np.arange(0, len(ia), 37)
    z1, Z1 = m.zavg(ia[sel], ib[sel])
    m2 = CoessentialityModel(X, TOY)
    perm = np.random.default_rng(1).permutation(len(ia))[:20000]
    big = np.r_[perm, sel][::-1]
    z2, Z2 = m2.zavg(ib[big], ia[big])                  # other batch, reversed order, swapped orientation
    z2, Z2 = z2[:len(sel)][::-1], {c: v[:len(sel)][::-1] for c, v in Z2.items()}
    assert np.array_equal(z1, z2)
    for c in "BEF":
        assert np.array_equal(Z1[c], Z2[c])
    bl1, bl2 = Baselines(X), Baselines(X)
    for meth in ("pearson", "spearman", "gls"):
        assert np.array_equal(bl1.score(meth, ia[sel], ib[sel]), bl2.score(meth, ia[big], ib[big])[:len(sel)][::-1])


def test_null_z_moments(mod):
    _, _, m = mod
    na, nb = m.null_pairs()
    assert len(na) == TOY.n_null and (na < nb).all() and m.bg[na].all() and m.bg[nb].all()
    for c in "BEF":
        z = m.z(c, na, nb)
        assert abs(z.mean()) < 1e-10 and abs(z.std() - 1) < 1e-10
    na2, _ = CoessentialityModel(m.Xc.T, TOY).null_pairs()
    assert np.array_equal(na, na2)                      # fixed seed


def test_genome_wide_block_and_top_pairs(mod):
    X, _, m = mod
    G = X.shape[1]
    ia, ib = all_pairs(G)
    z, _ = m.zavg(ia, ib)
    M = np.vstack([m.zavg_block(rows) for rows in m._blocks()])
    assert np.allclose(M[ia, ib], z, atol=1e-9) and np.allclose(M[ib, ia], z, atol=1e-9)
    ta, tb = top_pairs(m, 300)
    ref = np.argsort(-z, kind="stable")[:300]
    assert set(zip(ta.tolist(), tb.tolist())) == set(zip(ia[ref].tolist(), ib[ref].tolist()))


def test_missing_values():
    """Genes with >= 10% NaN: excluded from Sigma and neighbour sets, still scored; baselines are pairwise complete."""
    X, _ = modules(seed=6)
    rng = np.random.default_rng(6)
    X[rng.random(X.shape) < 0.02] = np.nan               # sparse missingness
    X[:30, 5] = np.nan                                   # gene 5: 37.5% NaN (block, like a missing library)
    X[:30, 300] = np.nan
    m = CoessentialityModel(X, TOY)
    assert not m.bg[5] and not m.bg[300] and m.bg.sum() == X.shape[1] - 2
    idx, _ = m.neighbours()
    assert not np.isin(idx, [5, 300]).any()
    ia, ib = all_pairs(X.shape[1])
    hit = (ia == 5) | (ib == 5)
    z, _ = m.zavg(ia[hit], ib[hit])
    assert np.isfinite(z).all()
    bl = Baselines(X)
    for j in (6, 300):
        ok = ~np.isnan(X[:, 5]) & ~np.isnan(X[:, j])
        assert np.isclose(bl.score("pearson", [5], [j])[0], stats.pearsonr(X[ok, 5], X[ok, j]).statistic)
        assert np.isclose(bl.score("spearman", [5], [j])[0], stats.spearmanr(X[ok, 5], X[ok, j]).statistic)


def test_self_pairs_rejected(mod):
    _, _, m = mod
    with pytest.raises(ValueError):
        m.B([1, 2], [1, 3])
