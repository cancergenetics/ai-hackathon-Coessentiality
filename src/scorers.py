"""Scorer interface used by src/evaluate.py and scripts/final_eval.py.

A scorer is bound to a PairUniverse (``bind``; label-free, genome-wide precomputation goes here and is timed as
setup), then ``fit(train_pair_idx)`` (no-op for label-free scorers; supervised scorers may read
``universe.y[train_pair_idx]`` and nothing else label-related) and ``score(pair_idx)`` -> float array. Indices are
row positions in the bound universe. Scores must depend only on the pair itself (batch/order invariance and (a,b)
symmetry are tested).

Fast path (PI addendum 6): ``PrecomputedScorer`` reads a per-pair score array aligned to the canonical scorable
``pair_id`` (Phase-A label-free layers cached as f(X, ia, ib) on all scorable pairs before any split).

Baseline variants (DESIGN §3.8; NaN protocol for GLS):
  pearson_{signed,abs}, spearman_{signed,abs}: pairwise-complete (exact; src/baselines.py).
  gls[_impute][_or]_{signed,abs}:
     (default) exact marginalisation: Sigma from genes with no NaN ("drop NaN genes from Sigma estimation"), each pair
               scored on the models observed for both genes with the marginal Sigma[S,S] (= per-pair literal
               cell-line drop); _impute: NaN -> gene mean, Sigma from all genes, one whitening.
     _or: Wainberg olfactory-receptor 4-PC removal (src.baselines.olfactory_correction) before Sigma.
     signed = t, abs = two-sided -log10 p.
  The literal Wainberg protocol (drop every model with any NaN) leaves 6 of 1,361 models for the scorable genes and
  is not run (recorded in results/baselines_v2/README.md).
"""
from __future__ import annotations

import numpy as np

from src import baselines


class Scorer:
    name = "scorer"
    label_free = True

    def __init__(self):
        self.params = {}
        self.u = None

    def bind(self, universe):
        self.u = universe
        self._ia = universe.col("ia")
        self._ib = universe.col("ib")
        return self

    def fit(self, train_idx):
        return self

    def score(self, idx) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        return dict(name=self.name, label_free=self.label_free, params=self.params, cls=type(self).__name__)


class GeneFunctionScorer(Scorer):
    """Label-free scorer from fn(ia, ib) -> scores over gene column indices."""

    def __init__(self, name, fn, params=None, setup=None):
        super().__init__()
        self.name, self.fn, self.params, self._setup = name, fn, dict(params or {}), setup

    def bind(self, universe):
        super().bind(universe)
        if self._setup is not None:
            self._setup()
            self._setup = None
        return self

    def score(self, idx):
        idx = np.asarray(idx)
        return np.asarray(self.fn(self._ia[idx], self._ib[idx]), dtype=np.float64)


class PrecomputedScorer(Scorer):
    """Fast path: per-pair score array indexed by the universe's pair_id."""

    def __init__(self, name, scores_by_pair_id, params=None):
        super().__init__()
        self.name, self.arr, self.params = name, np.asarray(scores_by_pair_id, dtype=np.float64), dict(params or {})

    def bind(self, universe):
        super().bind(universe)
        self._pid = universe.col("pair_id")
        return self

    def score(self, idx):
        return self.arr[self._pid[np.asarray(idx)]]


# ------------------------------------------------------------------------------------------- baselines
BASELINE_VARIANTS = ["pearson_signed", "pearson_abs", "spearman_signed", "spearman_abs"] + [
    f"gls{nan}{orr}_{out}" for nan in ("", "_impute") for orr in ("", "_or") for out in ("signed", "abs")]


class _Matrix:
    """Lazily prepared (genome-wide, label-free) matrices shared by baseline scorers built on the same X."""

    def __init__(self, X, genes=None):
        self.X = X
        self.genes = genes
        self._cache = {}

    def get(self, impute: bool, olf: bool):
        key = (impute, olf)
        if key in self._cache:
            return self._cache[key]
        X = np.asarray(self.X, dtype=np.float64)
        if impute:
            mu = np.nanmean(X, axis=0)
            X = np.where(np.isnan(X), mu[None, :], X)
        if olf:
            X, _ = baselines.olfactory_correction(X, self.olfactory_idx(), n_pc=4)
        Sigma, _ = baselines.sample_covariance(X)
        self._cache[key] = (X, Sigma)
        return self._cache[key]

    def olfactory_idx(self):
        if self.genes is None:
            raise ValueError("olfactory correction needs gene names")
        from src.data import gene_annotation
        ann = gene_annotation(self.genes)
        return np.flatnonzero(ann.gene_group.fillna("").str.contains("Olfactory receptors").values)


def make_baseline(variant: str, mat: _Matrix) -> GeneFunctionScorer:
    method, _, out = variant.rpartition("_")
    if method in ("pearson", "spearman"):
        f = baselines.pearson_pairs if method == "pearson" else baselines.spearman_pairs
        return GeneFunctionScorer(variant, lambda ia, ib: f(mat.X, ia, ib)[out],
                                  params=dict(method=method, output=out, nan="pairwise_complete"))
    if not method.startswith("gls"):
        raise ValueError(variant)
    impute, olf = "_impute" in method, "_or" in method
    state = {}

    def setup():
        state["X"], state["Sigma"] = mat.get(impute, olf)

    def fn(ia, ib):
        return baselines.gls_pairs(state["X"], ia, ib, Sigma=state["Sigma"])[out]

    return GeneFunctionScorer(variant, fn, setup=setup,
                              params=dict(method="gls", output=out, nan="mean_impute" if impute else "marginalise",
                                          olfactory_correction=olf, rank_deficiency="drop"))


def real_matrix() -> _Matrix:
    from src.data import load_matrix
    X, _, genes = load_matrix("CRISPRGeneEffect", mmap=True)
    return _Matrix(X, genes)


# ------------------------------------------------------------------------------------------ frozen IFT candidates
# RESEARCH_TREE D3 / results/phaseB_v2/decision.json: fixed sequence 1. zavg_BEF (primary), 2. B (= ref_B, C_single).
# Label-free layers cached genome-wide on all scorable pairs (row = pair_id) before any split, read with the same
# loader and the same manifest-null z-scoring as scripts/21_phaseB_v2.py (``zlayer``); the null moments are the ones
# src/predict.py recomputes and asserts equal to the manifests (scripts/predict_pairs.py --method zavg:B,E,F).
CANDIDATE_LAYERS = {
    "B": ("screen_v2", "rowz[socos_q0.25[spec_k20_t0.1_std]]"),
    "E": ("graph_v1", "emb_lap512[knn_union_lin_q0.25[spec_k20_t0.1_std]]"),
    "F": ("screen_v1", "spec_k20_t0.1_std"),
}
CANDIDATES = {"zavg_BEF": ("B", "E", "F"), "B": ("B",)}


def cached_layer(key: str) -> tuple[np.ndarray, dict]:
    """(raw float64 scores indexed by scorable pair_id, manifest column record) of a frozen candidate layer."""
    from src.supervised import load_layer_cache          # lazy: src.supervised imports this module
    tag, name = CANDIDATE_LAYERS[key]
    M, _, cols = load_layer_cache(tag)
    c = cols[name]
    return np.asarray(M[:, c["col"]], dtype=np.float64), c


def null_z(key: str) -> np.ndarray:
    """Layer z-scored with its manifest genome-wide null moments (frozen; not estimated on evaluated pairs)."""
    raw, c = cached_layer(key)
    return (raw - c["null_mean"]) / c["null_sd"]


def make_candidate(name: str) -> PrecomputedScorer:
    """``zavg_BEF``: equal-weight mean of the null-z-scored B, E, F; ``B``: the raw cached B layer."""
    comps = CANDIDATES[name]
    info = {}
    for k in comps:
        _, c = cached_layer(k)
        info[k] = dict(layer=CANDIDATE_LAYERS[k][1], cache=CANDIDATE_LAYERS[k][0], col=c["col"],
                       null_mean=c["null_mean"], null_sd=c["null_sd"])
    if name == "B":
        arr, _ = cached_layer("B")
        params = dict(members=info, combine="raw cached layer (monotone in its null z)")
    else:
        arr = np.mean([null_z(k) for k in comps], axis=0)
        params = dict(members=info, combine="equal-weight mean of manifest-null z")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: non-finite cached scores")
    return PrecomputedScorer(name, arr, params=params)


def make_scorer(name: str, mat: _Matrix | None = None) -> Scorer:
    """Registry used by scripts/final_eval.py: baseline variants and the frozen IFT candidates by name."""
    if name in BASELINE_VARIANTS:
        return make_baseline(name, mat if mat is not None else real_matrix())
    if name in CANDIDATES:
        return make_candidate(name)
    raise KeyError(f"unknown scorer {name!r}")
