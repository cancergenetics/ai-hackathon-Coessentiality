"""zavg_BEF: a label-free co-essentiality score for co-complex membership.

Every score is a deterministic function of the gene-effect matrix X (cell lines x genes) and the two gene column
indices. No pair labels are used anywhere. All genome-wide quantities (filter, row moments, neighbour sets, graph,
embedding, null) are computed over the whole genome, never over the requested pairs, so a pair's score does not
depend on which other pairs are scored, nor on the order (a, b).

1. Preprocessing
   * Background genes (BG): genes with NaN fraction < 10%. Only BG genes are used to estimate the covariance, to
     form genome-wide row distributions and neighbour sets, and as null pairs.
   * Every gene is centred on its observed mean and its missing entries are set to that mean (0 after centring).
2. Filter (F, first order)
   * Each centred gene is divided by its observed SD (ddof=0) for the covariance estimate only.
   * Sigma = Y'Y / |BG| with Y the standardised BG profiles expressed in an orthonormal basis of the complement of
     the all-ones vector (this removes the intercept exactly). Sigma = U diag(lam) U' (lam descending).
   * Filter weights: 0 for the top k = 20 eigencomponents, (lam_i + tau * mean(lam))^(-1/2) otherwise, tau = 0.1
     (a ridge-regularised, PC-filtered GLS whitening).
   * F(a, b) = cosine of the filtered (unstandardised, centred) profiles.
3. Second order (B)
   * Balanced first-order similarity V_aj = (z_aj + z_ja) / 2, where z_aj is F_aj standardised by gene a's
     genome-wide row mean / SD (over BG \\ {a}).
   * Each gene keeps its top K = round(q/100 * |BG|) BG neighbours by V (q = 0.25%, K = 44 for DepMap 26Q3), with
     weights max(V, 0).
   * SO(a, b) = cosine of the two sparse neighbour-weight vectors with partner exclusion (b is removed from a's
     vector and a from b's), so only shared third-party neighbours count.
   * B(a, b) = (z'_ab + z'_ba) / 2 with z' the genome-wide row-z of SO (row moments over BG \\ {a}).
4. Graph embedding (E)
   * Union kNN graph: the directed neighbour-weight matrix W of step 3, symmetrised as max(W, W') on the BG block
     (non-BG genes keep their out-edges and have no in-edges).
   * N = D^-1/2 W D^-1/2 (float32, as in the original run). Laplacian eigenmap: the top d + 1 eigenvectors of N
     restricted to BG genes (ARPACK, seeded start vector), the leading (trivial) one dropped, d = 512. Non-BG genes
     are embedded by the Nystrom extension N_g,BG U / lam.
   * E(a, b) = cosine of the embedding rows.
5. Final score
   * Each component is z-scored with fixed moments (mean, SD ddof=0) of its values on 200,000 random BG gene pairs
     (numpy default_rng(0)).
   * zavg_BEF = (z_B + z_E + z_F) / 3.

Port of the research code; ``tests/test_equivalence.py`` checks it against the research cache.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

COMPONENTS = ("B", "E", "F")
PAIR_CHUNK = 50_000


@dataclass(frozen=True)
class Config:
    """Frozen hyper-parameters of the published method (change only for experiments / toy data)."""
    k: int = 20                  # top eigencomponents removed
    tau: float = 0.1             # ridge, as a fraction of the mean eigenvalue (math.inf = no whitening)
    q: float = 0.25              # neighbours kept, percent of BG genes
    d: int = 512                 # embedding dimension
    max_nan: float = 0.10        # BG genes: NaN fraction < max_nan
    standardize: bool = True     # standardise genes before estimating Sigma
    center: str = "gene"         # 'gene' (method) | 'line' (Wainberg GLS convention; for tests)
    n_null: int = 200_000        # null gene pairs for the z-moments
    seed: int = 0                # null pairs and eigensolver start vector
    block: int = 1024            # fixed gene blocks for genome-wide row computations

    def __post_init__(self):
        if self.center not in ("gene", "line") or (self.center == "line" and self.standardize):
            raise ValueError("center must be 'gene', or 'line' with standardize=False")
        if self.tau < 0:
            raise ValueError("tau >= 0")


# ------------------------------------------------------------------------------------------ helpers
def householder_complement(n: int) -> np.ndarray:
    """n x (n-1) orthonormal basis of the complement of the all-ones vector."""
    u = np.full(n, 1.0 / np.sqrt(n))
    v = u.copy()
    v[0] -= 1.0
    H = np.eye(n) - 2.0 * np.outer(v, v) / (v @ v)
    return H[:, 1:]


def row_moments(S: np.ndarray, self_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD (ddof=0) of each row of S (rows x BG genes), excluding column self_pos[i] (-1: none)."""
    S = np.asarray(S, dtype=np.float64)
    r = np.flatnonzero(self_pos >= 0)
    cnt = S.shape[1] - (self_pos >= 0).astype(np.float64)
    tot = S.sum(1)
    tot[r] -= S[r, self_pos[r]]
    m = tot / cnt
    D = S - m[:, None]
    D[r, self_pos[r]] = 0.0
    return m, np.sqrt((D * D).sum(1) / cnt)


def rowdot(A: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Per-pair dot products of rows of A (each value depends only on its two rows)."""
    out = np.empty(len(ia))
    for s in range(0, len(ia), PAIR_CHUNK):
        sl = slice(s, s + PAIR_CHUNK)
        out[sl] = (A[ia[sl]] * A[ib[sl]]).sum(1)
    return out


def _check_pairs(ia, ib):
    ia, ib = np.asarray(ia, dtype=np.int64), np.asarray(ib, dtype=np.int64)
    if ia.shape != ib.shape:
        raise ValueError("ia/ib shapes differ")
    if (ia == ib).any():
        raise ValueError("self pairs are undefined")
    return np.minimum(ia, ib), np.maximum(ia, ib)          # canonical orientation


# ------------------------------------------------------------------------------------------ model
class CoessentialityModel:
    """Genome-wide state for F, B, E (lazily computed and cached) and their frozen null z-moments."""

    def __init__(self, X: np.ndarray, config: Config = Config()):
        self.cfg = c = config
        X = np.asarray(X, dtype=np.float64)
        self.n, self.G = X.shape
        nan = np.isnan(X)
        self.nan_frac = nan.mean(0)
        if (self.nan_frac == 1).any():
            raise ValueError("genes with no observation")
        self.bg = self.nan_frac < c.max_nan
        self.bg_idx = np.flatnonzero(self.bg)
        self.n_bg = len(self.bg_idx)
        self.bg_pos = np.full(self.G, -1)
        self.bg_pos[self.bg_idx] = np.arange(self.n_bg)
        self.mean = np.nanmean(X, 0)
        self.sd = np.nanstd(X, 0)
        Xc = X - self.mean
        Xc[nan] = 0.0
        self.Xc = np.ascontiguousarray(Xc.T)                 # genes x lines, centred, mean-imputed
        self.K = max(1, int(round(c.q / 100.0 * self.n_bg)))
        self._cache: dict = {}
        self._moments: dict = {}

    def _blocks(self):
        for b in range(math.ceil(self.G / self.cfg.block)):
            yield np.arange(b * self.cfg.block, min((b + 1) * self.cfg.block, self.G))

    def _cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # -------------------------------------------------------------------------- F: filtered profiles
    def eig(self):
        def run():
            c = self.cfg
            Xb = self.Xc[self.bg_idx]
            if c.standardize:
                Xb = Xb / self.sd[self.bg_idx, None]
            if c.center == "gene":
                Q = householder_complement(self.n)
                Y = Xb @ Q
                S = (Y.T @ Y) / len(Y)
            else:                                            # Wainberg: np.cov over genes (lines centred)
                Q = None
                Y = Xb + self.mean[self.bg_idx, None]
                Y = Y - Y.mean(0, keepdims=True)
                S = (Y.T @ Y) / (len(Y) - 1)
            lam, U = np.linalg.eigh(S)
            o = np.argsort(lam)[::-1]
            lam, U = np.clip(lam[o], 0.0, None), U[:, o]
            null = lam <= len(lam) * np.finfo(float).eps * lam[0]
            return lam, U, Q, null
        return self._cached("eig", run)

    def filter_weights(self) -> np.ndarray:
        lam, _, _, null = self.eig()
        c = self.cfg
        if c.tau == math.inf:
            w = np.ones_like(lam)
        elif c.tau == 0:
            w = np.zeros_like(lam)
            w[~null] = lam[~null] ** -0.5
        else:
            w = (lam + c.tau * lam.mean()) ** -0.5
        w[: c.k] = 0.0
        return w

    def profiles(self) -> tuple[np.ndarray, np.ndarray]:
        """Unit-norm filtered profiles (genes x m), float64 and float32."""
        def run():
            _, U, Q, _ = self.eig()
            w = self.filter_weights()
            sel = w > 0
            M = U[:, sel] * w[sel]
            if Q is not None:
                M = Q @ M
            P = self.Xc @ M
            if self.cfg.center == "line":                   # project out the filtered intercept
                cvec = M.sum(0)
                cvec /= np.linalg.norm(cvec)
                P -= np.outer(P @ cvec, cvec)
            with np.errstate(invalid="ignore", divide="ignore"):
                Z = P / np.linalg.norm(P, axis=1, keepdims=True)
            return np.ascontiguousarray(Z), np.ascontiguousarray(Z, dtype=np.float32)
        return self._cached("profiles", run)

    def F(self, ia, ib) -> np.ndarray:
        ia, ib = _check_pairs(ia, ib)
        return rowdot(self.profiles()[0], ia, ib)

    def _F_block32(self, rows):
        Z32 = self.profiles()[1]
        return Z32[rows] @ Z32.T                             # fixed block matmul (bit-stable rows)

    def f_row_stats(self):
        def run():
            m, s = np.empty(self.G), np.empty(self.G)
            for rows in self._blocks():
                S = self._F_block32(rows)
                m[rows], s[rows] = row_moments(S[:, self.bg_idx], self.bg_pos[rows])
            return m, s
        return self._cached("f_row_stats", run)

    # -------------------------------------------------------------------------- B: second order
    def neighbours(self):
        """Top-K BG neighbours (self excluded) by the balanced similarity: (idx G x K, value G x K), descending."""
        def run():
            K = self.K
            m, s = self.f_row_stats()
            idx, val = np.empty((self.G, K), np.int64), np.empty((self.G, K))
            for rows in self._blocks():
                S = self._F_block32(rows)
                V = 0.5 * ((S - m[rows, None]) / s[rows, None] + (S - m[None, :]) / s[None, :])
                V = V[:, self.bg_idx]
                r = np.flatnonzero(self.bg_pos[rows] >= 0)
                V[r, self.bg_pos[rows[r]]] = -np.inf
                part = np.argpartition(-V, K - 1, axis=1)[:, :K]
                pv = np.take_along_axis(V, part, 1)
                o = np.argsort(-pv, axis=1, kind="stable")
                idx[rows] = self.bg_idx[np.take_along_axis(part, o, 1)]
                val[rows] = np.take_along_axis(pv, o, 1)
            return idx, val
        return self._cached("neighbours", run)

    def knn_weights(self):
        """Directed neighbour-weight matrix W (csr G x G, weights max(V, 0)), its row weights and squared norms."""
        def run():
            idx, val = self.neighbours()
            w = np.clip(val, 0.0, None)
            K = self.K
            W = sp.csr_matrix((w.ravel().copy(), idx.ravel().copy(), np.arange(0, self.G * K + 1, K)),
                              shape=(self.G, self.G))
            W.sort_indices()
            sq = np.asarray(W.multiply(W).sum(1)).ravel()
            return W, w, sq
        return self._cached("knn", run)

    def second_order(self, ia, ib) -> np.ndarray:
        """Partner-excluded cosine of the neighbour-weight vectors (SO)."""
        ia, ib = _check_pairs(ia, ib)
        W, w, sq = self.knn_weights()
        idx = self.neighbours()[0]
        out = np.empty(len(ia))
        for s in range(0, len(ia), PAIR_CHUNK):
            sl = slice(s, s + PAIR_CHUNK)
            a, b = ia[sl], ib[sl]
            dot = np.asarray(W[a].multiply(W[b]).sum(1)).ravel()     # self never in own vector
            wab = (w[a] * (idx[a] == b[:, None])).sum(1)             # b's weight in a's vector
            wba = (w[b] * (idx[b] == a[:, None])).sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                den = np.sqrt(np.clip(sq[a] - wab ** 2, 0, None) * np.clip(sq[b] - wba ** 2, 0, None))
                out[sl] = np.where(den > 0, dot / den, 0.0)
        return out

    def _so_block(self, rows) -> np.ndarray:
        """SO of genes `rows` against all genes (dense, partner-excluded)."""
        W, _, sq = self.knn_weights()
        WT = self._cached("knnT", lambda: W.T.tocsr())
        D = (W[rows] @ W.T).toarray()
        Wa = W[rows].toarray()                               # W[a, j]
        Wt = WT[rows].toarray()                              # W[j, a]
        with np.errstate(invalid="ignore", divide="ignore"):
            den = np.sqrt(np.clip(sq[rows, None] - Wa ** 2, 0, None) * np.clip(sq[None, :] - Wt ** 2, 0, None))
            return np.where(den > 0, D / den, 0.0)

    def so_row_stats(self):
        def run():
            m, s = np.empty(self.G), np.empty(self.G)
            for rows in self._blocks():
                S = self._so_block(rows)
                m[rows], s[rows] = row_moments(S[:, self.bg_idx], self.bg_pos[rows])
            return m, s
        return self._cached("so_row_stats", run)

    def B(self, ia, ib) -> np.ndarray:
        ia, ib = _check_pairs(ia, ib)
        s = self.second_order(ia, ib)
        m, sd = self.so_row_stats()
        return 0.5 * ((s - m[ia]) / sd[ia] + (s - m[ib]) / sd[ib])

    # -------------------------------------------------------------------------- E: graph embedding
    def graph(self) -> sp.csr_matrix:
        """Union kNN graph max(W, W') on the BG block; non-BG rows keep their directed edges."""
        def run():
            W = self.knn_weights()[0]
            bgd = sp.diags(self.bg.astype(np.float64))
            WT = (W.T @ bgd).tocsr()
            S = W.maximum(WT).tocsr()
            nb = ~self.bg
            if nb.any():
                S = (bgd @ S + sp.diags(nb.astype(np.float64)) @ W).tocsr()
            S.eliminate_zeros()
            S.sort_indices()
            return S
        return self._cached("graph", run)

    def embedding(self) -> np.ndarray:
        """Row-normalised Laplacian-eigenmap coordinates (G x d)."""
        def run():
            d = self.cfg.d
            if d > self.n_bg - 2:
                raise ValueError(f"embedding dimension {d} > |BG| - 2 = {self.n_bg - 2}")
            A = self.graph()
            deg = np.asarray(A.sum(1)).ravel()
            with np.errstate(divide="ignore"):
                r = np.where(deg > 0, deg ** -0.5, 0.0)
            N = (sp.diags(r) @ A @ sp.diags(r)).tocsr().astype(np.float32)
            N.sort_indices()
            N = N.astype(np.float64)
            Nbb = N[self.bg_idx][:, self.bg_idx]
            v0 = np.random.default_rng(self.cfg.seed).standard_normal(self.n_bg)
            lam, U = eigsh(Nbb, k=d + 1, which="LA", v0=v0, tol=1e-8)
            o = np.argsort(lam)[::-1]
            lam, U = lam[o], U[:, o]
            E = np.zeros((self.G, d + 1))
            E[self.bg_idx] = U
            nb = np.flatnonzero(~self.bg)
            if len(nb):
                with np.errstate(divide="ignore", invalid="ignore"):
                    E[nb] = (N[nb][:, self.bg_idx] @ U) / np.where(np.abs(lam) > 1e-12, lam, np.inf)
            E = E[:, 1:]                                     # drop the trivial leading eigenvector
            with np.errstate(invalid="ignore", divide="ignore"):
                E = E / np.linalg.norm(E, axis=1, keepdims=True)
            return np.ascontiguousarray(np.nan_to_num(E))
        return self._cached("embedding", run)

    def E(self, ia, ib) -> np.ndarray:
        ia, ib = _check_pairs(ia, ib)
        return rowdot(self.embedding(), ia, ib)

    # -------------------------------------------------------------------------- null, z, zavg
    def null_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixed random unordered BG gene pairs (a != b, drawn with replacement)."""
        def run():
            rng = np.random.default_rng(self.cfg.seed)
            a, b = rng.choice(self.bg_idx, self.cfg.n_null), rng.choice(self.bg_idx, self.cfg.n_null)
            while (bad := a == b).any():
                b[bad] = rng.choice(self.bg_idx, int(bad.sum()))
            return np.minimum(a, b), np.maximum(a, b)
        return self._cached("null_pairs", run)

    def component(self, name: str, ia, ib) -> np.ndarray:
        return {"B": self.B, "E": self.E, "F": self.F}[name](ia, ib)

    def null_moments(self, name: str) -> tuple[float, float]:
        if name not in self._moments:
            s = self.component(name, *self.null_pairs())
            s = s[np.isfinite(s)]
            self._moments[name] = (float(s.mean()), float(s.std()))
        return self._moments[name]

    def z(self, name: str, ia, ib) -> np.ndarray:
        mu, sd = self.null_moments(name)
        return (self.component(name, ia, ib) - mu) / sd

    def zavg(self, ia, ib, components=COMPONENTS) -> tuple[np.ndarray, dict]:
        """(equal-weight mean of component z-scores, {component: z})."""
        Z = {c: self.z(c, ia, ib) for c in components}
        out = np.zeros(len(np.asarray(ia)))
        for c in components:
            out = out + Z[c]
        return out / len(components), Z

    # -------------------------------------------------------------------------- genome-wide
    def z_block(self, rows) -> dict:
        """Fixed-null z-scores {B, E, F} of genes `rows` against all genes (dense float64 blocks)."""
        Z64 = self.profiles()[0]
        out = {}
        mu, sd = self.null_moments("B")
        m, s = self.so_row_stats()
        S = self._so_block(rows)
        out["B"] = (0.5 * ((S - m[rows, None]) / s[rows, None] + (S - m[None, :]) / s[None, :]) - mu) / sd
        del S
        Em = self.embedding()
        mu, sd = self.null_moments("E")
        out["E"] = (Em[rows] @ Em.T - mu) / sd
        mu, sd = self.null_moments("F")
        out["F"] = (Z64[rows] @ Z64.T - mu) / sd
        return out

    def zavg_block(self, rows) -> np.ndarray:
        """zavg_BEF of genes `rows` against all genes (dense block; for genome-wide ranking)."""
        Z = self.z_block(rows)
        return (Z["B"] + Z["E"] + Z["F"]) / 3.0
