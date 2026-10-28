"""Baselines vs scipy / reference implementations; GLS positive control; negative control."""
import numpy as np
import pytest
from scipy import stats

from src.baselines import (gls_pairs, gls_reference, olfactory_correction, pearson_pairs, sample_covariance,
                           spearman_pairs)
from src.config import SEEDS
from src.metrics import auprc


def _nan_blocks(rng, n=80, g=40):
    X = rng.normal(size=(n, g))
    X[:, :5] = np.round(X[:, :5], 1)                  # ties for Spearman
    X[:20, 10:15] = np.nan                            # block missingness patterns (library-like)
    X[50:70, 15:18] = np.nan
    X[5:25, 18:20] = np.nan                           # overlaps the first block
    return X


def _all_pairs(g):
    ia, ib = np.triu_indices(g, 1)
    return ia, ib


def test_pearson_spearman_match_scipy_with_nan():
    rng = np.random.default_rng(SEEDS[0])
    X = _nan_blocks(rng)
    ia, ib = _all_pairs(X.shape[1])
    P, S = pearson_pairs(X, ia, ib), spearman_pairs(X, ia, ib)
    for k, (i, j) in enumerate(zip(ia, ib)):
        ok = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
        assert P["signed"][k] == pytest.approx(stats.pearsonr(X[ok, i], X[ok, j])[0], abs=1e-10)
        assert S["signed"][k] == pytest.approx(stats.spearmanr(X[:, i], X[:, j], nan_policy="omit")[0], abs=1e-10)
        assert P["n_obs"][k] == ok.sum()
    assert np.allclose(P["abs"], np.abs(P["signed"]))


def test_gls_identity_cov_equals_ols():
    rng = np.random.default_rng(SEEDS[1])
    X = rng.normal(size=(50, 12)) + 3.0
    ia, ib = _all_pairs(12)
    out = gls_pairs(X, ia, ib, Sigma=np.eye(50))
    for k, (i, j) in enumerate(zip(ia, ib)):
        lr = stats.linregress(X[:, i], X[:, j])
        assert out["signed"][k] == pytest.approx(lr.slope / lr.stderr, rel=1e-8)
        assert out["p"][k] == pytest.approx(lr.pvalue, rel=1e-6, abs=1e-300)


def test_gls_vectorised_equals_wainberg_reference():
    rng = np.random.default_rng(SEEDS[2])
    n, g = 30, 300
    A = rng.normal(size=(n, n)) * 0.3 + np.eye(n)
    X = (A @ rng.normal(size=(n, g))) + rng.normal(size=(1, g))   # correlated cells
    _, _, t_ref, p_ref = gls_reference(X.T)
    ia, ib = _all_pairs(25)
    out = gls_pairs(X, ia, ib)                                       # Sigma estimated from all 300 genes
    assert np.allclose(out["signed"], t_ref[ia, ib], rtol=1e-6, atol=1e-8)
    assert np.allclose(out["p"], p_ref[ia, ib], rtol=1e-5, atol=1e-12)
    assert np.allclose(t_ref, t_ref.T, rtol=1e-6, atol=1e-8)        # symmetry used by vectorisation


def test_gls_nan_marginalisation():
    rng = np.random.default_rng(SEEDS[3])
    n, g = 30, 300
    X = rng.normal(size=(n, g))
    Sigma, _ = sample_covariance(X)                                  # from complete genes
    Xn = X.copy(); Xn[:8, 0] = np.nan                                 # gene 0 missing in 8 samples
    out = gls_pairs(Xn, np.array([0]), np.array([1]), Sigma=Sigma)
    S = ~np.isnan(Xn[:, 0])
    _, _, t_ref, _ = gls_reference(X[S][:, :2].T, Sigma=Sigma[np.ix_(S, S)])
    assert out["signed"][0] == pytest.approx(t_ref[0, 1], rel=1e-8)
    assert out["n_obs"][0] == S.sum()


def _planted(rng, n=60, g=2000, n_pos=100, rho=0.5):
    """Cells with block (lineage) covariance; independent genes ~ N(0, Sigma); planted pairs share signal."""
    blocks = np.repeat(np.arange(6), n // 6)
    Sigma = 0.15 * np.eye(n) + 0.85 * (blocks[:, None] == blocks[None, :])
    C = np.linalg.cholesky(Sigma)
    X = C @ rng.normal(size=(n, g))
    ia = np.arange(0, 2 * n_pos, 2); ib = ia + 1
    X[:, ib] = rho * X[:, ia] + np.sqrt(1 - rho ** 2) * X[:, ib]
    neg_a = rng.integers(2 * n_pos, g, 20000); neg_b = rng.integers(2 * n_pos, g, 20000)
    keep = neg_a != neg_b
    pa = np.r_[ia, neg_a[keep]]; pb = np.r_[ib, neg_b[keep]]
    y = np.r_[np.ones(n_pos), np.zeros(keep.sum())]
    return X, pa, pb, y


def test_gls_positive_control_beats_pearson():
    rng = np.random.default_rng(SEEDS[4])
    X, pa, pb, y = _planted(rng)
    ap_p = auprc(y, pearson_pairs(X, pa, pb)["abs"])
    ap_g = auprc(y, gls_pairs(X, pa, pb)["abs"])
    assert ap_g > ap_p + 0.2, (ap_g, ap_p)
    assert ap_g > 0.8


def test_negative_control_shuffled():
    rng = np.random.default_rng(SEEDS[0])
    X, pa, pb, y = _planted(rng)
    s = gls_pairs(X, pa, pb)["abs"]
    aps = [auprc(np.random.default_rng(sd).permutation(y), s) for sd in range(20)]
    assert np.mean(aps) == pytest.approx(y.mean(), abs=0.003)


def test_olfactory_correction_shapes_and_rank():
    rng = np.random.default_rng(SEEDS[1])
    X = rng.normal(size=(40, 500)) + rng.normal(size=(40, 1)) * 2
    X[:3, 450] = np.nan
    X[10, :100] = np.nan                                   # a sample lacking all olfactory genes -> excluded
    Xc, keep = olfactory_correction(X, np.arange(100), n_pc=4)
    assert Xc.shape == (35, 500) and keep.sum() == 35 and not keep[10] and not keep[-4:].any()
    assert np.isnan(Xc[:3, 450]).all() and np.isnan(Xc).sum() == 3
    S, _ = sample_covariance(Xc)
    assert np.linalg.matrix_rank(S) == 35


def test_whitening_rank_deficiency():
    from src.baselines import _gls_t_block, whitening
    rng = np.random.default_rng(SEEDS[2])
    n, g = 30, 400
    X = rng.normal(size=(n, g))
    # full rank: pinv-eig whitening and Cholesky whitening give identical t (t depends only on x' Sigma^-1 y)
    S, _ = sample_covariance(X)
    A1, k1 = whitening(S, "drop"); A2, k2 = whitening(S, "pinv")
    assert k1.all() and k2.all()
    assert np.allclose(_gls_t_block(X[:, :20], A1), _gls_t_block(X[:, :20], A2), atol=1e-8)
    # make 2 exact linear dependencies among samples (spread over many samples, not the last ones)
    X[3] = X[14:24].mean(0); X[7] = 0.5 * X[11] - 0.5 * X[12]
    S, _ = sample_covariance(X)
    assert np.linalg.matrix_rank(S, hermitian=True) == n - 2
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(np.linalg.pinv(S))
    A, keep = whitening(S, "drop")
    assert (~keep).sum() == 2 and A.shape == (n - 2, n - 2)
    out = gls_pairs(X, np.array([0]), np.array([1]), Sigma=S)
    _, _, t_ref, _ = gls_reference(X[keep][:, :2].T, Sigma=S[np.ix_(keep, keep)])
    assert out["signed"][0] == pytest.approx(t_ref[0, 1], rel=1e-6)
    Ap, kp = whitening(S, "pinv")
    assert kp.all() and Ap.shape == (n - 2, n)
