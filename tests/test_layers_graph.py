"""kNN-graph / diffusion / embedding layers (src/layers_graph.py): planted-module controls, known answers vs
src/layers + brute force, symmetry, batch invariance."""
import numpy as np
import pytest

from src.config import SEEDS
from src.layers import LayerEngine, SecondOrder, Spectral
from src.layers_ext import RowZSO
from src.layers_ext import compute as ext_compute
from src.layers_graph import GraphEngine, GraphLayer, KNNGraph, SoftGraph, as_config, series_coeffs, parse_op

BASE = Spectral(1, 0.1)


def _toy(seed=SEEDS[0], n=60, G=400, modules=12, msize=8, mscale=1.0, nan_genes=0):
    """Planted modules of msize genes (shared latent factor) + rank-2 line confounder + noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, G)) + rng.normal(size=G)
    X += 2.0 * rng.normal(size=(n, 2)) @ rng.normal(size=(2, G))
    planted = []
    for m in range(modules):
        g = np.arange(m * msize, (m + 1) * msize)
        X[:, g] += mscale * rng.normal(size=(n, 1))
        planted += [(a, b) for i, a in enumerate(g) for b in g[i + 1:]]
    for j in range(G - nan_genes, G):
        X[rng.random(n) < 0.4, j] = np.nan
    return X, planted


@pytest.fixture(scope="module")
def toy():
    X, planted = _toy()
    eng = LayerEngine(X, block=64)
    return X, planted, eng, GraphEngine(eng)


@pytest.fixture(scope="module")
def toy_nan():
    X, planted = _toy(nan_genes=12)
    eng = LayerEngine(X, block=64)
    return eng, GraphEngine(eng)


U = KNNGraph(BASE, 2.0, "union", "lin")
LAYERS = [GraphLayer(KNNGraph(BASE, 2.0, "dir", "lin")), GraphLayer(KNNGraph(BASE, 2.0, "mutual", "p4")),
          GraphLayer(KNNGraph(BASE, 2.0, "avg", "bin"), rowz=True), GraphLayer(U, "adj", "tom"),
          GraphLayer(SoftGraph(BASE, 6.0), "adj", "tom"), GraphLayer(SoftGraph(BASE, 6.0), "adj", "cos"),
          GraphLayer(U, "pow2", "kern"), GraphLayer(U, "pow3", "cos", rowz=True), GraphLayer(U, "ppr0.5", "cos"),
          GraphLayer(U, "ppr0.5", "kern", rowz=True), GraphLayer(U, "heat1", "kern"), GraphLayer(U, "lap16", "emb"),
          GraphLayer(U, "dmap16t4", "emb", rowz=True)]


def _top_frac(score, ia, ib, planted):
    P = set(map(tuple, planted))
    top = np.argsort(-score)[: len(P)]
    return np.mean([(ia[t], ib[t]) in P for t in top])


# ------------------------------------------------------------------------------ known answers
def test_dir_lin_cos_equals_second_order(toy):
    _, _, eng, ge = toy
    ia, ib = np.triu_indices(eng.G, 1)
    ref = eng.compute(SecondOrder(BASE, 2.0, "cos"), ia, ib)
    out = ge.compute(GraphLayer(KNNGraph(BASE, 2.0, "dir", "lin")), ia, ib)
    assert np.allclose(out, ref, atol=2e-5)


def test_rowz_dir_lin_equals_rowz_second_order(toy):
    _, _, eng, ge = toy
    ia, ib = np.triu_indices(eng.G, 1)
    ref = ext_compute(RowZSO(SecondOrder(BASE, 2.0, "cos")), ia, ib, eng)
    out = ge.compute(GraphLayer(KNNGraph(BASE, 2.0, "dir", "lin"), rowz=True), ia, ib)
    assert np.allclose(out, ref, atol=1e-3)
    assert np.corrcoef(out, ref)[0, 1] > 0.99999


def test_partner_excluded_cosine_bruteforce(toy):
    _, _, eng, ge = toy
    L = GraphLayer(U, "ppr0.5", "cos")
    P = ge.profiles(L).astype(np.float64)
    rng = np.random.default_rng(0)
    ia, ib = rng.choice(eng.G, 300), rng.choice(eng.G, 300)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    ref = []
    for a, b in zip(ia, ib):
        m = np.ones(eng.G, bool)
        m[[a, b]] = False
        x, y = P[a, m], P[b, m]
        ref.append(x @ y / np.sqrt((x @ x) * (y @ y)))
    assert np.allclose(ge.compute(L, ia, ib), ref, atol=1e-5)


def test_tom_bruteforce(toy):
    _, _, eng, ge = toy
    L = GraphLayer(KNNGraph(BASE, 2.0, "union", "bin"), "adj", "tom")
    A = ge.adjacency(L.graph).toarray()
    assert set(np.unique(A)) <= {0.0, 1.0} and np.allclose(A, A.T) and not np.diag(A).any()
    k = A.sum(1)
    ia, ib = np.triu_indices(eng.G, 1)
    ref = (np.einsum("ij,ij->i", A[ia], A[ib]) + A[ia, ib]) / (np.minimum(k[ia], k[ib]) + 1 - A[ia, ib])
    assert np.allclose(ge.compute(L, ia, ib), ref, atol=1e-5)


def test_kernel_pow2_is_two_step_walk(toy):
    _, _, eng, ge = toy
    N = ge.operator(U).toarray().astype(np.float64)
    assert np.allclose(N, N.T, atol=1e-7)
    ia, ib = np.triu_indices(eng.G, 1)
    assert np.allclose(ge.compute(GraphLayer(U, "pow2", "kern"), ia, ib), (N @ N)[ia, ib], atol=1e-6)


def test_series_coeffs():
    for op in ("ppr0.5", "ppr0.2", "heat1", "heat3"):
        c = series_coeffs(parse_op(op))
        assert 1 - c.sum() < 1e-3 and (c > 0).all()
    assert np.array_equal(series_coeffs(parse_op("pow3")), [0, 0, 0, 1])


def test_graph_symmetrisation(toy):
    _, _, eng, ge = toy
    d = ge.adjacency(KNNGraph(BASE, 2.0, "dir", "bin")).toarray()
    u = ge.adjacency(KNNGraph(BASE, 2.0, "union", "bin")).toarray()
    m = ge.adjacency(KNNGraph(BASE, 2.0, "mutual", "bin")).toarray()
    assert np.array_equal(u, u.T) and np.array_equal(m, m.T)
    assert (m <= d).all() and (d <= u).all() and m.sum() < d.sum() < u.sum()
    assert (d.sum(1) == eng.n_neighbours(2.0)).all()


# ------------------------------------------------------------------------------ planted-module controls
@pytest.mark.parametrize("L", [GraphLayer(U, "ppr0.5", "cos"), GraphLayer(U, "pow2", "kern"),
                               GraphLayer(U, "heat1", "kern"), GraphLayer(U, "adj", "tom"),
                               GraphLayer(U, "lap16", "emb"), GraphLayer(U, "adj", "cos")], ids=lambda L: L.name)
def test_diffusion_recovers_planted_modules(toy, L):
    _, planted, eng, ge = toy
    ia, ib = np.triu_indices(eng.G, 1)
    s = ge.compute(L, ia, ib)
    frac = _top_frac(s, ia, ib, planted)
    base = _top_frac(eng.compute(BASE, ia, ib), ia, ib, planted)          # first-order input: 0.54
    if L.op != "heat1":       # heat kernel value keeps a large 1-step (direct-link) term: ranking test only
        assert frac > 0.75 and frac > base + 0.2, (L.name, frac, base)
    # module co-membership: within-module pairs outrank between-module pairs (AUROC)
    y = np.zeros(len(ia), bool)
    lut = {p: i for i, p in enumerate(zip(ia, ib))}
    y[[lut[p] for p in planted]] = True
    r = np.argsort(np.argsort(s))
    auc = (r[y].mean() - (y.sum() - 1) / 2) / (~y).sum()
    assert auc > 0.9, (L.name, auc)


# ------------------------------------------------------------------------------ symmetry / invariance
@pytest.mark.parametrize("L", LAYERS, ids=lambda L: L.name)
def test_symmetry_and_batch_invariance(toy, L):
    X, _, eng, ge = toy
    rng = np.random.default_rng(1)
    ia, ib = rng.choice(eng.G, 2000), rng.choice(eng.G, 2000)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    s = ge.compute(L, ia, ib)
    assert np.isfinite(s).all()
    assert np.array_equal(s, ge.compute(L, ib, ia))                    # exactly symmetric
    sub = rng.permutation(len(ia))[:137]
    assert np.array_equal(s[sub], ge.compute(L, ia[sub], ib[sub]))       # batch / order invariance
    ge2 = GraphEngine(LayerEngine(X, block=64))                          # fresh engine, different request
    assert np.array_equal(s[sub], ge2.compute(L, ia[sub][::-1], ib[sub][::-1])[::-1])


@pytest.mark.parametrize("L", [LAYERS[0], LAYERS[3], LAYERS[6], LAYERS[8], LAYERS[11]], ids=lambda L: L.name)
def test_non_background_genes_scored(toy_nan, L):
    eng, ge = toy_nan
    nb = np.flatnonzero(~eng.bg)
    assert len(nb) == 12
    ia = np.repeat(nb, 50)
    ib = np.tile(np.arange(50), len(nb))
    s = ge.compute(L, ia, ib)
    assert np.isfinite(s).all() and np.array_equal(s, ge.compute(L, ib, ia))
    # non-BG genes are never anyone's neighbour / intermediate
    A = ge.adjacency(L.graph)
    A = A.toarray() if hasattr(A, "toarray") else A
    assert not A[:, nb].any()


def test_config_roundtrip():
    for L in LAYERS:
        assert as_config(L.to_dict()) == L and as_config(L.to_dict()).name == L.name
    with pytest.raises(ValueError):
        GraphLayer(KNNGraph(BASE, 1.0, "dir"), "pow2", "kern")               # diffusion needs a symmetric graph
    with pytest.raises(ValueError):
        GraphLayer(U, "pow2", "tom")
