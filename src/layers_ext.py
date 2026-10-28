"""Phase-A layer extensions used by the screen_v2 cache (label-free; same contract as src/layers.py).

``RowZSO``: genome-wide row-z balancing of a sparse second-order cosine layer (A3 applied on top of A4).
For a SecondOrder config C (measure='cos') with partner-excluded weighted-cosine score SO(a, b), the row moments
(mean_a, sd_a, ddof=0) of gene a are taken over SO(a, j), j in BG \\ {a}, computed exactly (same partner exclusion
as ``LayerEngine._second_order``) in fixed gene blocks. Score = (z_ab + z_ba) / 2.

``compute(cfg, ia, ib, engine)`` dispatches to ``src.layers`` for the screen_v1 config kinds and handles RowZSO;
``as_config`` accepts the ``to_dict()`` form of either. Nothing here reads labels or the pair table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import scipy.sparse as sp

from src import layers
from src.layers import SecondOrder


@dataclass(frozen=True)
class RowZSO:
    base: SecondOrder
    kind: ClassVar[str] = "rowzso"

    def __post_init__(self):
        if self.base.measure != "cos":
            raise ValueError("RowZSO is implemented for measure='cos' only")

    @property
    def name(self):
        return f"rowz[{self.base.name}]"

    def to_dict(self):
        return dict(kind=self.kind, base=self.base.to_dict())


def as_config(c):
    if isinstance(c, dict) and c.get("kind") == "rowzso":
        return RowZSO(layers.as_config(c["base"]))
    return layers.as_config(c)


def _so_matrix(eng: layers.LayerEngine, cfg: SecondOrder):
    K = eng.n_neighbours(cfg.q)
    Kmax = max(K, eng.n_neighbours(2.0))
    idx, val = eng.neighbours(cfg.base, cfg.balanced, Kmax)
    idx, val = idx[:, :K], val[:, :K]
    w = np.clip(val, 0.0, None)
    W = sp.csr_matrix((w.ravel().copy(), idx.ravel().copy(), np.arange(0, eng.G * K + 1, K)), shape=(eng.G, eng.G))
    W.sort_indices()
    return W


_SO_STATS: dict = {}


def so_row_stats(eng: layers.LayerEngine, cfg: SecondOrder):
    """Exact genome-wide row moments of the second-order cosine layer (rows = all genes, columns = BG \\ self)."""
    key = (id(eng), cfg)
    if key in _SO_STATS:
        return _SO_STATS[key]
    W = _so_matrix(eng, cfg)
    WT = W.T.tocsr()
    sq = np.asarray(W.multiply(W).sum(1)).ravel()
    m, s = np.empty(eng.G), np.empty(eng.G)
    for b in eng._all_blocks():
        rows = np.arange(b * eng.block, min((b + 1) * eng.block, eng.G))
        D = (W[rows] @ W.T).toarray()                 # shared-neighbour dot products
        Wa = W[rows].toarray()                      # weight of j in a's vector  (W[a, j])
        Wt = WT[rows].toarray()                     # weight of a in j's vector  (W[j, a])
        with np.errstate(invalid="ignore", divide="ignore"):
            den = np.sqrt(np.clip(sq[rows, None] - Wa ** 2, 0, None) * np.clip(sq[None, :] - Wt ** 2, 0, None))
            S = np.where(den > 0, D / den, 0.0)
        del D, Wa, Wt, den
        m[rows], s[rows] = layers.row_moments(S[:, eng.bg_idx], eng.bg_pos[rows])
    _SO_STATS[key] = (m, s)
    return m, s


def compute(cfg, ia, ib, engine: layers.LayerEngine | None = None) -> np.ndarray:
    cfg = as_config(cfg)
    if not isinstance(cfg, RowZSO):
        return layers.compute_layer(cfg, ia, ib, engine)
    eng = engine or layers.get_engine(cfg.base.base.matrix)
    ia, ib = np.asarray(ia, dtype=np.int64), np.asarray(ib, dtype=np.int64)
    s = eng.compute(cfg.base, ia, ib)
    m, sd = so_row_stats(eng, cfg.base)
    return 0.5 * ((s - m[ia]) / sd[ia] + (s - m[ib]) / sd[ib])


def release():
    _SO_STATS.clear()
