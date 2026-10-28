"""Scorer contract: batch/order invariance, (a,b) symmetry, fast path; real-data GLS invariance (skipped w/o cache)."""
import numpy as np
import pytest

from src import scorers
from src.config import CACHE, SEEDS
from src.toy import make_toy


@pytest.fixture(scope="module")
def toy():
    u, X, _ = make_toy(seed=SEEDS[0], n_complex=30, giant=0)
    return u, X


TOY_VARIANTS = [v for v in scorers.BASELINE_VARIANTS if "_or" not in v]     # OR correction needs real gene names


@pytest.mark.parametrize("variant", TOY_VARIANTS)
def test_batch_and_order_invariance(toy, variant):
    u, X = toy
    sc = scorers.make_baseline(variant, scorers._Matrix(X)).bind(u)
    rng = np.random.default_rng(SEEDS[0])
    full = sc.score(np.arange(len(u)))
    idx = rng.choice(len(u), 3000, replace=False)
    perm = rng.permutation(idx)
    a = sc.score(idx)
    b = sc.score(perm)
    assert np.allclose(a, full[idx], rtol=1e-10, atol=1e-12)
    assert np.allclose(b, full[perm], rtol=1e-10, atol=1e-12)
    for chunk in np.array_split(idx, 7):                     # small batches
        assert np.allclose(sc.score(chunk), full[chunk], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("variant", TOY_VARIANTS)
def test_symmetry(toy, variant):
    u, X = toy
    sc = scorers.make_baseline(variant, scorers._Matrix(X))
    sc.bind(u)
    ia, ib = u.col("ia")[:2000], u.col("ib")[:2000]
    assert np.allclose(sc.fn(ia, ib), sc.fn(ib, ia), rtol=1e-10, atol=1e-12)


def test_precomputed_fast_path(toy):
    u, X = toy
    arr = np.random.default_rng(SEEDS[1]).normal(size=len(u) + 10)
    sub = u.subset(np.arange(100, 900))
    sc = scorers.PrecomputedScorer("pre", arr).bind(sub)
    assert np.array_equal(sc.score(np.arange(5)), arr[sub.col("pair_id")[:5]])


@pytest.mark.skipif(not (CACHE / "CRISPRGeneEffect.f32.npy").exists(), reason="cache absent")
def test_real_gls_batch_invariance():
    import pandas as pd
    allp = pd.read_pickle(CACHE / "pairs.pkl")
    sc = allp[allp.scorable].reset_index(drop=True)
    rng = np.random.default_rng(SEEDS[2])
    idx = rng.choice(len(sc), 1500, replace=False)
    mat = scorers.real_matrix()
    X, Sigma = mat.get(False, False)
    from src.baselines import gls_pairs
    ia, ib = sc.ia.values[idx], sc.ib.values[idx]
    full = gls_pairs(X, ia, ib, Sigma=Sigma)["signed"]
    part = gls_pairs(X, ia[::-1][:300], ib[::-1][:300], Sigma=Sigma)["signed"]
    swap = gls_pairs(X, ib[:200], ia[:200], Sigma=Sigma)["signed"]
    assert np.allclose(part, full[::-1][:300], rtol=1e-9)
    assert np.allclose(swap, full[:200], rtol=1e-9)
