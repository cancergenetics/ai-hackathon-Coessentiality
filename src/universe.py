"""Evaluation universes: a pair table plus everything metrics and splits need, with a stable ``pair_id``.

``pair_id`` = row position in the canonical *scorable* pair table (``data.load_pairs()`` rows with scorable == True,
file order). Precomputed per-pair score arrays (Phase-A label-free layers, computed as f(X, ia, ib) on all scorable
pairs before any split) are aligned to it, so a scorer can index them directly (``scorers.PrecomputedScorer``).

``load_dev_universe()`` returns DEV pairs only (both proteins on the dev side of the §3.2 assignment). The IFT
loader lives in ``src/ift.py`` and may be imported only by ``scripts/final_eval.py`` (AST-tested).

Common-essential genes (label-free, from CRISPRGeneEffect, all genes): fraction of models with gene effect < -0.5
(among non-missing) >= 0.5. This reproduces the DESIGN §2 figures (9.1% of genes genome-wide; 30.8% of the
labelled scorable genes). The stricter >= 0.9 threshold would mark 5.5% / 20.2%.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from src.config import CACHE, N_NEG_SCORABLE, N_POS_SCORABLE

ESS_EFFECT = -0.5
ESS_FRACTION = 0.5


def common_essential(X: np.ndarray, effect: float = ESS_EFFECT, frac: float = ESS_FRACTION) -> np.ndarray:
    """Boolean per gene (column of X, samples x genes)."""
    X = np.asarray(X, dtype=np.float32)
    obs = (~np.isnan(X)).sum(0)
    dep = (X < effect).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (dep / np.maximum(obs, 1)) >= frac


@dataclass
class PairUniverse:
    """pairs columns: a, b, label, ia, ib, pair_id, cluster_a/b, comp_a/b, ess_a/b, side (dev/ift).
    f: fraction of scorable proteins in this universe (HT inclusion fraction for pooled metrics).
    folds_f: HT fraction for a K-fold test fold is 1/K (set by the evaluator)."""
    pairs: pd.DataFrame
    name: str
    f: float = 1.0
    largest_component: int | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.pairs)

    @property
    def y(self) -> np.ndarray:
        return self.pairs.label.values.astype(np.int8)

    def col(self, c) -> np.ndarray:
        return self.pairs[c].values

    def cluster_map(self) -> dict:
        p = self.pairs
        return dict(zip(np.r_[p.a.values, p.b.values], np.r_[p.cluster_a.values, p.cluster_b.values]))

    def component_map(self) -> dict:
        p = self.pairs
        return dict(zip(np.r_[p.a.values, p.b.values], np.r_[p.comp_a.values, p.comp_b.values]))

    def with_labels(self, y) -> "PairUniverse":
        p = self.pairs.copy()
        p["label"] = np.asarray(y).astype(np.int8)
        return replace(self, pairs=p)

    def subset(self, idx, name=None) -> "PairUniverse":
        return replace(self, pairs=self.pairs.iloc[np.asarray(idx)].reset_index(drop=True), name=name or self.name)


def annotate(pairs: pd.DataFrame, assign: pd.DataFrame, ess: np.ndarray | None) -> pd.DataFrame:
    """Add cluster/component/side/essentiality columns from a protein assignment table."""
    a = assign.set_index("protein")
    p = pairs.copy()
    for side_col, prot_col in (("a", "a"), ("b", "b")):
        p[f"cluster_{side_col}"] = a.cluster.reindex(p[prot_col]).values.astype(np.int64)
        p[f"comp_{side_col}"] = a.component.reindex(p[prot_col]).values.astype(np.int64)
        p[f"side_{side_col}"] = a.side.reindex(p[prot_col]).values
    p["side"] = np.where(p.side_a == p.side_b, p.side_a, "cross")
    if ess is not None:
        p["ess_a"] = ess[p.ia.values]
        p["ess_b"] = ess[p.ib.values]
    return p


def scorable_table() -> pd.DataFrame:
    """Canonical scorable pair table with pair_id; asserts the universe size used for reweighting."""
    allp = pd.read_pickle(CACHE / "pairs.pkl")
    sc = allp[allp.scorable].reset_index(drop=True)
    sc["pair_id"] = np.arange(len(sc), dtype=np.int64)
    n_pos = int(sc.label.sum())
    if (n_pos, len(sc) - n_pos) != (N_POS_SCORABLE, N_NEG_SCORABLE):
        raise RuntimeError(f"scorable universe changed: {n_pos}/{len(sc) - n_pos}")
    return sc


def _ess_flags():
    from src.data import load_matrix
    X, _, _ = load_matrix("CRISPRGeneEffect", mmap=True)
    return common_essential(X)


def _build(sides: tuple[str, ...], name: str) -> PairUniverse:
    from src.clusters import load_assignment
    assign = load_assignment()
    sc = scorable_table()
    p = annotate(sc, assign[assign.mapped], _ess_flags())
    keep = p.side.isin(sides).values
    p = p[keep].reset_index(drop=True)
    m = assign[assign.mapped]
    f = float(m.side.isin(sides).mean())
    largest = int(m.loc[m.component_size.idxmax(), "component"])
    meta = dict(n_cross_excluded=int((~keep).sum()), sides=list(sides))
    return PairUniverse(p, name, f=f, largest_component=largest, meta=meta)


def load_dev_universe() -> PairUniverse:
    """Dev pairs only (both proteins dev). f = dev fraction of scorable proteins."""
    return _build(("dev",), "dev")
