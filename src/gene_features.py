"""Label-free per-gene statistics and symmetric pair features for the Phase-B supervised pair model (DESIGN v3 §4 B).

Per-gene statistics (columns of CRISPRGeneEffect, cell lines x genes; NaN-aware, over the observed lines):
  mean, sd (ddof=1), skew (standardised third moment, population form), frac_dep (fraction of observed lines with
  effect < -0.5), nan_frac (fraction of lines missing), min, p05 (5th percentile).

Coarsening (R2-1: min/max of raw stats are near-unique per gene, so a tree could isolate one protein's label
propensity): each statistic is cut into <= 5 quantile bins whose edges are taken from the GENOME-WIDE distribution
of that statistic over all genes with a finite value (18,434 columns for CRISPRGeneEffect), never from labelled or
training genes. Edges = quantiles 0.2/0.4/0.6/0.8, de-duplicated (statistics with a point mass, e.g. nan_frac = 0
for most genes, get fewer bins). Bin = number of edges <= value (0..n_edges). Non-finite stat -> bin -1 (does not
occur for mapped labelled genes).

Pair features are symmetric in (a, b): for every statistic s, min(s_a, s_b) and max(s_a, s_b); binned by default,
raw values for the 'raw_gene_feats' ablation. Nothing here reads labels, the pair table or which genes are
labelled; values depend only on X and the two gene column indices.
"""
from __future__ import annotations

import numpy as np

STATS = ("mean", "sd", "skew", "frac_dep", "nan_frac", "min", "p05")
DEP_EFFECT = -0.5
N_BINS = 5
_MEMO: dict = {}


def gene_stats(X: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """(n_genes x len(STATS)) float64 array of per-gene statistics of X (samples x genes)."""
    X = np.asarray(X)
    G = X.shape[1]
    out = np.full((G, len(STATS)), np.nan)
    for s in range(0, G, chunk):
        B = np.asarray(X[:, s:s + chunk], dtype=np.float64)
        obs = ~np.isnan(B)
        n = obs.sum(0)
        ok = n >= 3
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.nansum(B, 0) / n
            D = np.where(obs, B - m, 0.0)
            v2 = (D ** 2).sum(0)
            sd = np.sqrt(v2 / (n - 1))
            sd_pop = np.sqrt(v2 / n)
            skew = (D ** 3).sum(0) / n / sd_pop ** 3
            frac_dep = (np.where(obs, B, np.inf) < DEP_EFFECT).sum(0) / n
            mn = np.nanmin(np.where(obs, B, np.inf), 0)
            p05 = np.nanpercentile(B, 5, axis=0) if ok.all() else np.array(
                [np.nanpercentile(B[:, j], 5) if ok[j] else np.nan for j in range(B.shape[1])])
        blk = np.c_[m, sd, skew, frac_dep, 1.0 - n / B.shape[0], mn, p05]
        blk[~ok] = np.nan
        out[s:s + chunk] = blk
    return out


def quantile_edges(stats: np.ndarray, n_bins: int = N_BINS) -> list[np.ndarray]:
    """Per-statistic interior bin edges from the genome-wide distribution (all genes with a finite value)."""
    if n_bins > N_BINS:
        raise ValueError("DESIGN caps gene-feature coarsening at 5 bins")
    q = np.arange(1, n_bins) / n_bins
    edges = []
    for j in range(stats.shape[1]):
        v = stats[:, j]
        v = v[np.isfinite(v)]
        edges.append(np.unique(np.quantile(v, q)))
    return edges


def bin_stats(stats: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    """Integer bins (0..len(edges_j)); -1 for non-finite values."""
    out = np.empty(stats.shape, dtype=np.int64)
    for j, e in enumerate(edges):
        v = stats[:, j]
        b = np.searchsorted(e, np.where(np.isfinite(v), v, 0.0), side="right")
        out[:, j] = np.where(np.isfinite(v), b, -1)
    return out


def pair_features(G: np.ndarray, ia, ib, names=STATS) -> tuple[np.ndarray, list[str]]:
    """Symmetric pair features [min_s, max_s for s in stats] from a per-gene table G (genes x stats)."""
    ia, ib = np.asarray(ia), np.asarray(ib)
    A, B = np.asarray(G[ia], dtype=np.float64), np.asarray(G[ib], dtype=np.float64)
    lo, hi = np.minimum(A, B), np.maximum(A, B)
    F = np.empty((len(ia), 2 * G.shape[1]))
    F[:, 0::2], F[:, 1::2] = lo, hi
    cols = [f"{p}_{s}" for s in names for p in ("min", "max")]
    return F, cols


def real_gene_tables(matrix: str = "CRISPRGeneEffect") -> dict:
    """Genome-wide stats, edges and bins for the real matrix (memoised per process)."""
    if matrix not in _MEMO:
        from src.data import load_matrix
        X, _, genes = load_matrix(matrix, mmap=True)
        st = gene_stats(X)
        ed = quantile_edges(st)
        _MEMO[matrix] = dict(stats=st, edges=ed, bins=bin_stats(st, ed), genes=genes,
                             n_bins=[len(e) + 1 for e in ed])
    return _MEMO[matrix]


def signature_uniqueness(bins: np.ndarray, gene_idx) -> dict:
    """How identifying the binned stat vector is among a gene set (diagnostic for R2-1)."""
    sub = bins[np.unique(np.asarray(gene_idx))]
    _, inv, cnt = np.unique(sub, axis=0, return_inverse=True, return_counts=True)
    per_gene = cnt[inv.ravel()]
    return dict(n_genes=int(len(sub)), n_signatures=int(len(cnt)), frac_unique=float(np.mean(per_gene == 1)),
                median_genes_per_signature=float(np.median(per_gene)))
