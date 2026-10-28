"""Label-free pair-similarity layers from a DepMap gene-effect matrix (DESIGN v3 §4 Phase A: A1, A3, A4, A6 null).

Every layer is a deterministic function f(X, a, b) of the gene-effect matrix X (cell lines x genes) and the two
gene column indices. Nothing here reads pair labels, the pair table, or which genes are labelled; scores for a
pair do not depend on which other pairs are requested (tested in tests/test_layers.py).

Preprocessing (``LayerEngine.__init__``)
---------------------------------------
* Background genes ("genome", BG): genes with NaN fraction < ``max_nan`` (10%; 17,525 of 18,434 genes for
  CRISPRGeneEffect 26Q3). Only BG genes are used to estimate the line x line covariance / PCs and to form the
  genome-wide row distributions (row-z moments, ranks, neighbour sets, null pairs).
* Every gene (BG or not, e.g. the 16 labelled genes with >=10% NaN) is centred on its observed mean and its
  missing entries are set to that mean (i.e. 0 after centring) before filtering, and is then scored like any
  other gene. Mean imputation shrinks the filtered profile of a high-NaN gene towards 0 on the imputed lines;
  correlations are computed on the full imputed vector (documented approximation; GLS in src/baselines.py
  marginalises exactly instead).
* Optional per-gene standardisation (``Spectral.standardize``) divides each centred gene by its observed SD
  before Sigma is estimated. Pair scores are scale-invariant per gene, so it only changes Sigma / the PCs.

A1 spectral filter family (``Spectral``)
----------------------------------------
center='gene' (default): profiles are gene-centred and expressed in an orthonormal basis Q of 1-perp (Householder),
so the intercept direction is removed exactly. Sigma = Y'Y / G_bg with Y = Xc_bg Q (second moment over BG genes;
it keeps line-wide mean shifts, which therefore count as confounder directions). Sigma = U diag(lam) U'
(lam descending). Filter weights w(lam_i) = 0 for i <= k (top-k removal), else (lam_i + tau_abs)^(-1/2) with
tau_abs = tau * mean(lam). tau=0: GLS whitening on range(Sigma) (eigenvalues <= n*eps*lam_max dropped, as
pinv); tau=inf: w = 1, i.e. Pearson after removing the top-k PCs (k=0: exactly Pearson). shrink='lw': Ledoit-Wolf
analytic shrinkage (1-a) Sigma + a mean(lam) I, which is the ridge filter with tau = a / (1 - a).
Pair score = cosine of the filtered profiles (= Pearson of filtered profiles, since the intercept is gone).
center='line': Wainberg's convention (Sigma = np.cov over genes, each line centred across genes); filtered profiles
F'x have the filtered intercept F'1 projected out, so k=0, tau=0 equals the GLS correlation r of
src/baselines.py exactly (t = r sqrt(df/(1-r^2)) is monotone in r) up to the NaN protocol.

A3 genome-wide balancing (``RowZ``, ``MutualRank``)
---------------------------------------------------
For a base spectral layer S, row moments (mean_a, sd_a, ddof=0) of every gene a are taken over S_aj, j in BG \\ {a}.
RowZ: (z_ab + z_ba) / 2 with z_ab = (S_ab - mean_a) / sd_a. MutualRank: -sqrt(rank_a(b) rank_b(a)) with
rank_a(b) = 1 + #{j in BG \\ {a, b}: V_aj > V_ab}; V = S (balanced=False) or the balanced matrix
B_aj = (z_aj + z_ja)/2 (balanced=True). Genome-wide rows are always computed in fixed gene blocks of size
``block`` (block = genes [i*block, (i+1)*block)), whatever genes are requested, so a row is bit-identical across
calls.

A4 sparse second-order similarity (``SecondOrder``)
---------------------------------------------------
Neighbour set N_a = top K = round(q/100 * |BG|) genes j in BG \\ {a} by B_aj (balanced=True) or S_aj, weights
max(value, 0). Pair (a, b): partner exclusion - b is removed from a's vector and a from b's vector (each gene
is excluded from its own vector by construction), so the score measures shared third-party neighbours only and
a direct strong link a-b neither helps (no overlap term) nor hurts (no norm penalty). cos = weighted cosine;
jaccard = |N_a ∩ N_b| / |N_a ∪ N_b| on the partner-excluded sets. Neighbour sets are selected before
exclusion (a vector can therefore have K-1 entries).

A6 fixed genome-wide null (``null_pairs``, ``null_moments``, ``zscore_layer``)
-----------------------------------------------------------------------------
200,000 random unordered BG gene pairs (a != b, drawn with replacement, seed SEEDS[0]); each layer's z-score uses
the mean/SD of its scores on these pairs. Moments of the screening grid are stored in results/layers/manifest.json.
"""
from __future__ import annotations

import json
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import scipy.sparse as sp
from sklearn.covariance import ledoit_wolf_shrinkage

from src.config import RESULTS, SEEDS

log = logging.getLogger(__name__)

INF = math.inf
BG_MAX_NAN = 0.10
BLOCK = 1024
N_NULL = 200_000
PAIR_CHUNK = 50_000
MANIFEST = RESULTS / "layers" / "manifest.json"
DEFAULT_MATRIX = "CRISPRGeneEffect"


# ------------------------------------------------------------------------------------------ configs
def _tau_str(t):
    return "inf" if t == INF else f"{t:g}"


@dataclass(frozen=True)
class Spectral:
    k: int = 0
    tau: float | None = INF          # relative ridge (fraction of mean eigenvalue); None when shrink='lw'
    shrink: str = "ridge"            # 'ridge' | 'lw'
    center: str = "gene"             # 'gene' | 'line'
    standardize: bool = False
    matrix: str = DEFAULT_MATRIX
    kind: ClassVar[str] = "spectral"

    def __post_init__(self):
        if self.shrink not in ("ridge", "lw") or self.center not in ("gene", "line"):
            raise ValueError(self)
        if self.shrink == "ridge" and (self.tau is None or self.tau < 0):
            raise ValueError("ridge needs tau >= 0")
        if self.center == "line" and self.standardize:
            raise ValueError("standardize is only defined for center='gene'")

    @property
    def name(self) -> str:
        s = f"spec_k{self.k}_t" + ("lw" if self.shrink == "lw" else _tau_str(self.tau))
        s += "_line" if self.center == "line" else ""
        s += "_std" if self.standardize else ""
        return s + ("" if self.matrix == DEFAULT_MATRIX else f"_{self.matrix}")

    def to_dict(self) -> dict:
        return dict(kind=self.kind, k=self.k, tau=None if self.tau is None else _tau_str(self.tau),
                    shrink=self.shrink, center=self.center, standardize=self.standardize, matrix=self.matrix)


@dataclass(frozen=True)
class RowZ:
    base: Spectral
    kind: ClassVar[str] = "rowz"

    @property
    def name(self):
        return f"rowz[{self.base.name}]"

    def to_dict(self):
        return dict(kind=self.kind, base=self.base.to_dict())


@dataclass(frozen=True)
class MutualRank:
    base: Spectral
    balanced: bool = False
    kind: ClassVar[str] = "mr"

    @property
    def name(self):
        return f"{'mrbal' if self.balanced else 'mr'}[{self.base.name}]"

    def to_dict(self):
        return dict(kind=self.kind, base=self.base.to_dict(), balanced=self.balanced)


@dataclass(frozen=True)
class SecondOrder:
    base: Spectral
    q: float = 1.0                   # percent of BG genes kept as neighbours
    measure: str = "cos"             # 'cos' | 'jaccard'
    balanced: bool = True
    kind: ClassVar[str] = "so"

    def __post_init__(self):
        if self.measure not in ("cos", "jaccard"):
            raise ValueError(self.measure)

    @property
    def name(self):
        return f"so{self.measure}_q{self.q:g}{'' if self.balanced else 'raw'}[{self.base.name}]"

    def to_dict(self):
        return dict(kind=self.kind, base=self.base.to_dict(), q=self.q, measure=self.measure, balanced=self.balanced)


LayerConfig = Spectral | RowZ | MutualRank | SecondOrder


def as_config(c) -> LayerConfig:
    """Accepts a config object or its ``to_dict()`` form."""
    if not isinstance(c, dict):
        return c
    c = dict(c)
    kind = c.pop("kind")
    if kind == "spectral":
        tau = c.get("tau")
        c["tau"] = None if tau is None else float(tau)
        return Spectral(**c)
    c["base"] = as_config(c["base"])
    return {"rowz": RowZ, "mr": MutualRank, "so": SecondOrder}[kind](**c)


def screening_grid() -> tuple[list[LayerConfig], list[Spectral]]:
    """Screening grid frozen before any label is looked at (DESIGN §4 Phase A). Returns (layers, bases).

    Bases for A3/A4 are chosen a priori to span the family (no labels): Pearson, two ridge levels x three k, LW."""
    K, TAU = (0, 1, 2, 3, 5, 10, 20, 40), (0.0, 0.01, 0.1, 1.0, 10.0, INF)
    layers: list[LayerConfig] = [Spectral(k, t) for k in K for t in TAU]
    layers += [Spectral(k, None, shrink="lw") for k in K]
    layers += [Spectral(0, 0.0, center="line")]                                      # Wainberg-centred GLS r
    layers += [Spectral(k, t, standardize=True) for k in (0, 5, 20) for t in (0.1, 1.0, INF) if (k, t) != (0, INF)]
    bases = [Spectral(0, INF), Spectral(0, 0.1), Spectral(0, None, shrink="lw"), Spectral(5, 0.1),
             Spectral(5, INF), Spectral(20, 0.1), Spectral(20, INF)]
    for b in bases:
        layers += [RowZ(b), MutualRank(b, False), MutualRank(b, True)]
        layers += [SecondOrder(b, q, m) for m in ("cos", "jaccard") for q in (0.5, 1.0, 2.0)]
    return layers, bases


# ------------------------------------------------------------------------------------------ helpers
def householder_complement(n: int) -> np.ndarray:
    """n x (n-1) orthonormal basis of the complement of the all-ones vector."""
    u = np.full(n, 1.0 / np.sqrt(n))
    v = u.copy()
    v[0] -= 1.0
    H = np.eye(n) - 2.0 * np.outer(v, v) / (v @ v)            # H e_1 = u, H symmetric orthogonal
    return H[:, 1:]


def row_moments(S: np.ndarray, self_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD (ddof=0) of each row of S (rows x background genes) excluding column self_pos[i] (-1: none)."""
    S = np.asarray(S, dtype=np.float64)
    r = np.flatnonzero(self_pos >= 0)
    cnt = S.shape[1] - (self_pos >= 0).astype(np.float64)
    tot = S.sum(1)
    tot[r] -= S[r, self_pos[r]]
    m = tot / cnt
    D = S - m[:, None]
    D[r, self_pos[r]] = 0.0
    return m, np.sqrt((D * D).sum(1) / cnt)


def row_z(S: np.ndarray, self_pos: np.ndarray) -> np.ndarray:
    """Row-standardised S (self entries are standardised too but were excluded from the moments)."""
    m, s = row_moments(S, self_pos)
    return (np.asarray(S, dtype=np.float64) - m[:, None]) / s[:, None]


def _rowdot(A: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Per-pair dot products of rows of A; each pair's value depends only on its two rows (no BLAS blocking)."""
    out = np.empty(len(ia))
    for s in range(0, len(ia), PAIR_CHUNK):
        sl = slice(s, s + PAIR_CHUNK)
        out[sl] = (A[ia[sl]] * A[ib[sl]]).sum(1)
    return out


# ------------------------------------------------------------------------------------------ engine
class LayerEngine:
    """Holds the preprocessed matrix and per-config caches (spectral profiles LRU, row moments, neighbour sets)."""

    def __init__(self, X: np.ndarray, max_nan: float = BG_MAX_NAN, block: int = BLOCK, cache_size: int = 3):
        X = np.asarray(X, dtype=np.float64)
        self.n, self.G = X.shape
        nan = np.isnan(X)
        self.nan_frac = nan.mean(0)
        if (self.nan_frac == 1).any():
            raise ValueError("genes with no observation")
        self.bg = self.nan_frac < max_nan
        self.bg_idx = np.flatnonzero(self.bg)
        self.n_bg = len(self.bg_idx)
        self.bg_pos = np.full(self.G, -1)
        self.bg_pos[self.bg_idx] = np.arange(self.n_bg)
        self.mean = np.nanmean(X, 0)
        self.sd = np.nanstd(X, 0)
        Xc = X - self.mean
        Xc[nan] = 0.0
        self.Xc = np.ascontiguousarray(Xc.T)                 # genes x lines, gene-centred, mean-imputed
        self.max_nan, self.block, self.cache_size = max_nan, block, cache_size
        self._eig, self._spec, self._stats, self._nbr = {}, OrderedDict(), {}, {}
        self.info: dict = {}

    # -------------------------------------------------------------------------- A1 spectral filter
    def eig(self, center: str, standardize: bool) -> dict:
        key = (center, standardize)
        if key not in self._eig:
            Xb = self.Xc[self.bg_idx]
            if standardize:
                Xb = Xb / self.sd[self.bg_idx, None]
            if center == "gene":
                Q = householder_complement(self.n)
                Y = Xb @ Q
                S = (Y.T @ Y) / len(Y)
            else:
                Q = None
                Y = Xb + self.mean[self.bg_idx, None]           # imputed raw values, genes x lines
                Y = Y - Y.mean(0, keepdims=True)                # each line centred across genes (np.cov)
                S = (Y.T @ Y) / (len(Y) - 1)
            lam, U = np.linalg.eigh(S)
            o = np.argsort(lam)[::-1]
            lam, U = np.clip(lam[o], 0.0, None), U[:, o]
            alpha = float(ledoit_wolf_shrinkage(Y, assume_centered=True))
            null = lam <= len(lam) * np.finfo(float).eps * lam[0]
            self._eig[key] = dict(lam=lam, U=U, Q=Q, lw_alpha=alpha, null=null)
            self.info[f"eig_{center}{'_std' if standardize else ''}"] = dict(
                dim=len(lam), n_null=int(null.sum()), lw_alpha=alpha, lw_tau_rel=alpha / (1 - alpha),
                top_frac=[float(x) for x in lam[:10] / lam.sum()], mean_eig=float(lam.mean()))
            del Y
        return self._eig[key]

    def filter_weights(self, cfg: Spectral) -> np.ndarray:
        e = self.eig(cfg.center, cfg.standardize)
        lam = e["lam"]
        if cfg.shrink == "lw":
            tau_abs = e["lw_alpha"] / (1 - e["lw_alpha"]) * lam.mean()
        elif cfg.tau == INF:
            tau_abs = INF
        else:
            tau_abs = cfg.tau * lam.mean()
        if tau_abs == INF:
            w = np.ones_like(lam)
        elif tau_abs == 0:
            w = np.zeros_like(lam)
            w[~e["null"]] = lam[~e["null"]] ** -0.5
        else:
            w = (lam + tau_abs) ** -0.5
        w[: cfg.k] = 0.0
        return w

    def spectral(self, cfg: Spectral) -> tuple[np.ndarray, np.ndarray]:
        """Unit-norm filtered profiles (genes x m) in float64 and float32."""
        if cfg in self._spec:
            self._spec.move_to_end(cfg)
            return self._spec[cfg]
        e = self.eig(cfg.center, cfg.standardize)
        w = self.filter_weights(cfg)
        sel = w > 0
        M = e["U"][:, sel] * w[sel]
        if e["Q"] is not None:
            M = e["Q"] @ M
        P = self.Xc @ M                                         # per-gene scale irrelevant (cosine)
        if cfg.center == "line":                                # project out the filtered intercept F'1
            c = M.sum(0)
            c /= np.linalg.norm(c)
            P -= np.outer(P @ c, c)
        with np.errstate(invalid="ignore", divide="ignore"):
            Z = P / np.linalg.norm(P, axis=1, keepdims=True)
        out = (np.ascontiguousarray(Z), np.ascontiguousarray(Z, dtype=np.float32))
        self._spec[cfg] = out
        while len(self._spec) > self.cache_size:
            self._spec.popitem(last=False)
        return out

    # -------------------------------------------------------------------------- genome-wide rows
    def _block_S(self, Z32, b):
        """Fixed gene block b (genes [b*block, (b+1)*block)) vs all genes, float32. Always the same matmul, so a
        gene's row is bit-identical whichever genes were requested."""
        rows = np.arange(b * self.block, min((b + 1) * self.block, self.G))
        return rows, Z32[rows] @ Z32.T

    def _all_blocks(self):
        return range(math.ceil(self.G / self.block))

    def row_stats(self, base: Spectral) -> tuple[np.ndarray, np.ndarray]:
        if base not in self._stats:
            _, Z32 = self.spectral(base)
            m, s = np.empty(self.G), np.empty(self.G)
            for b in self._all_blocks():
                rows, S = self._block_S(Z32, b)
                m[rows], s[rows] = row_moments(S[:, self.bg_idx], self.bg_pos[rows])
            self._stats[base] = (m, s)
        return self._stats[base]

    def _values(self, base, rows, S, balanced):
        if not balanced:
            return S.astype(np.float64)
        m, s = self.row_stats(base)
        return 0.5 * ((S - m[rows, None]) / s[rows, None] + (S - m[None, :]) / s[None, :])

    def neighbours(self, base: Spectral, balanced: bool, K: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-K BG neighbours (excluding self) of every gene: (idx G x K int64, value G x K), sorted descending."""
        key = (base, balanced)
        if key not in self._nbr or self._nbr[key][0].shape[1] < K:
            _, Z32 = self.spectral(base)
            idx, val = np.empty((self.G, K), np.int64), np.empty((self.G, K))
            for b in self._all_blocks():
                rows, S = self._block_S(Z32, b)
                V = self._values(base, rows, S, balanced)[:, self.bg_idx]
                r = np.flatnonzero(self.bg_pos[rows] >= 0)
                V[r, self.bg_pos[rows[r]]] = -np.inf
                part = np.argpartition(-V, K - 1, axis=1)[:, :K]
                pv = np.take_along_axis(V, part, 1)
                o = np.argsort(-pv, axis=1, kind="stable")
                idx[rows] = self.bg_idx[np.take_along_axis(part, o, 1)]
                val[rows] = np.take_along_axis(pv, o, 1)
            self._nbr[key] = (idx, val)
        idx, val = self._nbr[key]
        return idx[:, :K], val[:, :K]

    def ranks(self, base: Spectral, balanced: bool, r_genes, c_genes) -> np.ndarray:
        """rank of c in row r: 1 + #{j in BG, j != r: V_rj > V_rc} (j = c never counts)."""
        r_genes, c_genes = np.asarray(r_genes), np.asarray(c_genes)
        out = np.empty(len(r_genes))
        _, Z32 = self.spectral(base)
        blk = r_genes // self.block
        order = np.argsort(r_genes, kind="stable")
        bounds = np.searchsorted(r_genes[order], np.arange(0, self.G + 1))     # per-gene slices of `order`
        for b in np.unique(blk):
            rows, S = self._block_S(Z32, b)
            V = self._values(base, rows, S, balanced)
            for g in np.unique(r_genes[blk == b]):
                p = order[bounds[g]:bounds[g + 1]]
                lr = g - rows[0]
                row = V[lr, self.bg_idx].copy()
                if self.bg_pos[g] >= 0:
                    row[self.bg_pos[g]] = -np.inf
                row.sort()
                out[p] = 1 + self.n_bg - np.searchsorted(row, V[lr, c_genes[p]], side="right")
        return out

    # -------------------------------------------------------------------------- layer dispatch
    def compute(self, cfg, ia, ib) -> np.ndarray:
        cfg = as_config(cfg)
        ia, ib = np.asarray(ia, dtype=np.int64), np.asarray(ib, dtype=np.int64)
        if ia.shape != ib.shape:
            raise ValueError("ia/ib shapes differ")
        if (ia == ib).any():
            raise ValueError("self pairs are undefined")
        if isinstance(cfg, Spectral):
            return _rowdot(self.spectral(cfg)[0], ia, ib)
        base = cfg.base
        if isinstance(cfg, RowZ):
            s = _rowdot(self.spectral(base)[0], ia, ib)
            m, sd = self.row_stats(base)
            return 0.5 * ((s - m[ia]) / sd[ia] + (s - m[ib]) / sd[ib])
        if isinstance(cfg, MutualRank):
            r = self.ranks(base, cfg.balanced, np.r_[ia, ib], np.r_[ib, ia])
            return -np.sqrt(r[: len(ia)] * r[len(ia):])
        if isinstance(cfg, SecondOrder):
            return self._second_order(cfg, ia, ib)
        raise TypeError(cfg)

    def n_neighbours(self, q: float) -> int:
        return max(1, int(round(q / 100.0 * self.n_bg)))

    def _second_order(self, cfg: SecondOrder, ia, ib) -> np.ndarray:
        K = self.n_neighbours(cfg.q)
        Kmax = max(K, self.n_neighbours(2.0))                   # one selection pass serves q <= 2%
        idx, val = self.neighbours(cfg.base, cfg.balanced, Kmax)
        idx, val = idx[:, :K], val[:, :K]
        w = np.ones_like(val) if cfg.measure == "jaccard" else np.clip(val, 0.0, None)
        # copies: csr may alias its inputs and sort_indices() permutes them in place (would corrupt the cache)
        W = sp.csr_matrix((w.ravel().copy(), idx.ravel().copy(), np.arange(0, self.G * K + 1, K)),
                          shape=(self.G, self.G))
        W.sort_indices()
        sq = np.asarray(W.multiply(W).sum(1)).ravel()
        out = np.empty(len(ia))
        for s in range(0, len(ia), PAIR_CHUNK):
            sl = slice(s, s + PAIR_CHUNK)
            a, b = ia[sl], ib[sl]
            dot = np.asarray(W[a].multiply(W[b]).sum(1)).ravel()   # self excluded => partner terms are 0
            wab = (w[a] * (idx[a] == b[:, None])).sum(1)           # b's weight in a's vector
            wba = (w[b] * (idx[b] == a[:, None])).sum(1)
            if cfg.measure == "jaccard":
                union = (K - wab) + (K - wba) - dot
                out[sl] = dot / union
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    den = np.sqrt(np.clip(sq[a] - wab ** 2, 0, None) * np.clip(sq[b] - wba ** 2, 0, None))
                    out[sl] = np.where(den > 0, dot / den, 0.0)
        return out

    def release(self, base: Spectral | None = None):
        """Free caches (all, or those tied to one base)."""
        if base is None:
            self._spec.clear(), self._stats.clear(), self._nbr.clear()
            return
        self._spec.pop(base, None), self._stats.pop(base, None)
        for k in [k for k in self._nbr if k[0] == base]:
            self._nbr.pop(k)


# ------------------------------------------------------------------------------------------ module API
_ENGINES: dict[str, LayerEngine] = {}


def get_engine(matrix: str = DEFAULT_MATRIX) -> LayerEngine:
    if matrix not in _ENGINES:
        from src import data
        X, _, _ = data.load_matrix(matrix)
        _ENGINES[matrix] = LayerEngine(X)
    return _ENGINES[matrix]


def _matrix_of(cfg) -> str:
    return cfg.matrix if isinstance(cfg, Spectral) else cfg.base.matrix


def compute_layer(config, gene_idx_a, gene_idx_b, engine: LayerEngine | None = None) -> np.ndarray:
    """Scores (float64, higher = more similar) of layer ``config`` for pairs of gene column indices."""
    cfg = as_config(config)
    return (engine or get_engine(_matrix_of(cfg))).compute(cfg, gene_idx_a, gene_idx_b)


def null_pairs(engine: LayerEngine | None = None, n: int = N_NULL, seed: int = SEEDS[0]):
    """Fixed random genome-wide (BG x BG) unordered gene pairs, a != b, drawn with replacement."""
    eng = engine or get_engine()
    rng = np.random.default_rng(seed)
    a, b = rng.choice(eng.bg_idx, n), rng.choice(eng.bg_idx, n)
    while (bad := a == b).any():
        b[bad] = rng.choice(eng.bg_idx, int(bad.sum()))
    return np.minimum(a, b), np.maximum(a, b)


def moments(scores: np.ndarray) -> dict:
    s = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(s)
    return dict(null_mean=float(s[ok].mean()), null_sd=float(s[ok].std()), null_n=int(ok.sum()))


def null_moments(config, engine: LayerEngine | None = None, pairs=None) -> dict:
    cfg = as_config(config)
    eng = engine or get_engine(_matrix_of(cfg))
    na, nb = pairs if pairs is not None else null_pairs(eng)
    return moments(eng.compute(cfg, na, nb))


def zscore_layer(name: str, scores: np.ndarray, moments: dict | None = None) -> np.ndarray:
    """z-transform with the layer's frozen genome-wide null moments (from the manifest unless given)."""
    if moments is None:
        cols = {c["name"]: c for c in json.loads(MANIFEST.read_text())["columns"]}
        moments = cols[name]
    return (np.asarray(scores, dtype=np.float64) - moments["null_mean"]) / moments["null_sd"]
