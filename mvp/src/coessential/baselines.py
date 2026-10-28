"""Conventional co-essentiality baselines (signed; higher = more similar).

* Pearson r and Spearman rho, pairwise complete (each pair uses the cell lines observed for both genes; Spearman
  re-ranks within those lines, average ranks for ties).
* GLS signed t-statistic (Wainberg et al. 2021, Nat Genet 53:638): cell-line covariance Sigma = np.cov over the
  genes with no missing value (each line centred across genes); for the lines S observed for both genes the
  whitening operator A satisfies A'A = Sigma[S, S]^-1 (A = chol(pinv(Sigma[S, S]))'). The GLS regression of warped
  y on [warped intercept, warped x] gives, by Frisch-Waugh-Lovell, t = r sqrt(df / (1 - r^2)) with r the cosine of
  the warped profiles after projecting out the warped intercept and df = |S| - 2.
  Rank deficiency (DepMap 26Q3 Sigma has rank n - 4): the dependent lines found by pivoted QR are dropped.

Missingness in DepMap is block-structured (a gene is missing in every model of a library that lacks its guides), so
pairs are grouped by the pair of missingness patterns and computed exactly per group. GLS warps genes in fixed
genome-wide blocks, so a pair's value never depends on which other pairs are requested.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.linalg import qr

METHODS = ("pearson", "spearman", "gls")
GENE_BLOCK = 1024
WHITE_CACHE = 16


def sample_covariance(X: np.ndarray) -> np.ndarray:
    """Cell-line x cell-line covariance over genes with no NaN (np.cov(X[:, complete].T) convention)."""
    ok = ~np.isnan(X).any(0)
    Xc = np.asarray(X[:, ok], dtype=np.float64)
    Xc = Xc - Xc.mean(1, keepdims=True)
    return (Xc @ Xc.T) / (Xc.shape[1] - 1)


def whitening(Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(A k x n with A'A = Sigma^-1 on the kept lines, boolean mask of kept lines)."""
    n = Sigma.shape[0]
    r = np.linalg.matrix_rank(Sigma, hermitian=True)
    keep = np.ones(n, bool)
    if r < n:
        _, _, piv = qr(Sigma, pivoting=True, mode="economic")
        keep[piv[r:]] = False
        Sigma = Sigma[np.ix_(keep, keep)]
    return np.linalg.cholesky(np.linalg.pinv(Sigma)).T, keep


def _rowdot(Z, la, lb, chunk: int = 50_000):
    out = np.empty(len(la))
    for s in range(0, len(la), chunk):
        out[s:s + chunk] = np.einsum("ij,ij->i", Z[la[s:s + chunk]], Z[lb[s:s + chunk]])
    return out


class Baselines:
    """Pairwise-complete Pearson / Spearman and GLS signed t for gene-index pairs of X (cell lines x genes)."""

    def __init__(self, X: np.ndarray):
        self.X = X
        self.XT = np.ascontiguousarray(np.asarray(X, dtype=np.float64).T)      # genes x lines, NaN kept
        valid = ~np.isnan(X)
        keys = [valid[:, j].tobytes() for j in range(X.shape[1])]
        uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        self.pid = np.array([uniq[k] for k in keys])
        self.pmask = [np.frombuffer(k, dtype=bool) for k in uniq]
        self._sigma = None
        self._white: dict = {}

    def _groups(self, ia, ib):
        n = len(self.pmask)
        p1, p2 = np.minimum(self.pid[ia], self.pid[ib]), np.maximum(self.pid[ia], self.pid[ib])
        code = p1 * n + p2
        for c in np.unique(code):
            idx = np.flatnonzero(code == c)
            P, Q = divmod(int(c), n)
            yield idx, self.pmask[P] & self.pmask[Q]

    def score(self, method: str, ia, ib) -> np.ndarray:
        if method not in METHODS:
            raise ValueError(method)
        ia, ib = np.asarray(ia, np.int64), np.asarray(ib, np.int64)
        lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
        out = np.full(len(lo), np.nan)
        for idx, S in self._groups(lo, hi):
            genes = np.unique(np.r_[lo[idx], hi[idx]])
            la, lb = np.searchsorted(genes, lo[idx]), np.searchsorted(genes, hi[idx])
            if method == "gls":
                out[idx] = self._gls(S, genes, la, lb)
                continue
            Y = self.XT[genes][:, S]
            if method == "spearman":
                Y = stats.rankdata(Y, axis=1)
            Y = Y - Y.mean(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                Z = Y / np.sqrt((Y * Y).sum(1, keepdims=True))
            out[idx] = np.clip(_rowdot(Z, la, lb), -1.0, 1.0)
        return out

    def _whiten(self, S):
        key = S.tobytes()
        if key not in self._white:
            if self._sigma is None:
                self._sigma = sample_covariance(self.X)
            A, keep = whitening(self._sigma[np.ix_(S, S)])
            w0 = A.sum(1)
            self._white[key] = (A, np.flatnonzero(S)[keep], w0 / np.linalg.norm(w0))
            while len(self._white) > WHITE_CACHE:
                self._white.pop(next(iter(self._white)))
        return self._white[key]

    def _gls(self, S, genes, la, lb):
        A, Sk, w0 = self._whiten(S)
        df = A.shape[0] - 2
        if df < 1:
            return np.full(len(la), np.nan)
        Z = np.empty((len(genes), A.shape[0]))
        G = self.XT.shape[0]
        for b in np.unique(genes // GENE_BLOCK):            # warp whole fixed blocks (batch invariance)
            rows = np.arange(b * GENE_BLOCK, min((b + 1) * GENE_BLOCK, G))
            Y = np.nan_to_num(self.XT[rows][:, Sk])
            W = Y @ A.T
            R = W - np.outer(W @ w0, w0)                    # project out the warped intercept
            with np.errstate(invalid="ignore", divide="ignore"):
                R /= np.linalg.norm(R, axis=1, keepdims=True)
            sel = (genes >= rows[0]) & (genes <= rows[-1])
            Z[sel] = R[genes[sel] - rows[0]]
        r = np.clip(_rowdot(Z, la, lb), -1.0, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return r * np.sqrt(df / (1.0 - r * r))
