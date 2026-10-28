"""Unsupervised co-essentiality baselines: Pearson, Spearman, GLS (Wainberg et al. 2021, Nat Genet).

Input X is samples x genes (cell lines or screens x genes), may contain NaN. Pairs are given as gene column
indices (ia, ib). Every method returns a dict of per-pair arrays:
    signed : r (Pearson/Spearman) or GLS t-statistic
    abs    : |r|, or for GLS the two-sided -log10 p (Wainberg rank pairs by two-sided p)
    n_obs  : number of samples used for that pair
(GLS additionally returns ``p``, ``log10p`` and ``df``.) Pairs whose score is undefined (e.g. the genes share
fewer than 3 observed samples, as happens for genes absent from a library when restricting to it) are NaN; the
caller decides how to treat them (scripts exclude and report them).

NaN handling: pairwise-complete, computed EXACTLY. DepMap missingness is block-structured (a gene is missing
in all models of a library that lacks guides for it), so the score-relevant genes fall into a few missingness
patterns (11 among the 1,114 HuMAP genes in CRISPRGeneEffect 26Q3). For every pair of patterns (p, q) we take
the samples S observed in both and compute the method on X[S] for the genes having pattern p or q. This is
identical to per-pair complete-case analysis (tests compare with scipy.stats on NaN data):
  * Pearson  : r on the pairwise-complete samples (== scipy.stats.pearsonr on the complete cases).
  * Spearman : ranks recomputed within the pairwise-complete samples, average ranks for ties
               (== scipy.stats.spearmanr(nan_policy='omit')). Ranking each gene once over its own non-missing
               samples would NOT be exact when patterns differ, so we re-rank per pattern pair.
  * GLS      : sample x sample covariance Sigma estimated as in Wainberg (np.cov over genes, i.e. each sample
               centred across genes) using all genes with no missing value in X (17,082 of 18,434 genes for
               CRISPRGeneEffect). For a pattern pair with sample set S the model is the marginal of the same
               Gaussian, so we use Sigma[S, S] (whitening L = chol(pinv(Sigma[S,S])), df = |S| - 2).
               Wainberg's code instead drops samples (cell lines) with any NaN; that is infeasible here (1,356 of
               1,361 models have at least one NaN gene) and mean-imputation would distort genes missing in up to
               84% of models, so exact marginalisation is the principled choice. For genes with no NaN this is
               exactly Wainberg's procedure.
               Rank deficiency: Sigma for CRISPRGeneEffect 26Q3 has rank n_models - 4 (exact linear dependencies
               among models after per-sample centring), so cholesky(pinv(Sigma)) fails. See ``whitening``:
               default drops the dependent samples chosen by pivoted QR; alternative 'pinv' whitens on range(Sigma).

GLS vectorisation: Wainberg regress warped y on [warped intercept, warped x] with np.linalg.lstsq per gene and
take t = coef / se with se = sqrt(pinv(X'X)[1,1] * RSS / df). By Frisch-Waugh-Lovell this equals
t = r * sqrt(df / (1 - r^2)) where r is the cosine similarity of warped x and warped y after projecting out the
warped intercept, so all pairs come from one matrix product. ``gls_reference`` is a line-by-line port of their
loop used in tests to verify equality.

Optional ``olfactory_correction`` reproduces Wainberg's load_screens preprocessing: PCA(4) fitted on olfactory
receptor genes (genes as observations, cells as features); the reconstruction (mean + top-4-PC projection) is
subtracted from every gene; then the last 4 samples are dropped to restore a full-rank covariance. NaN entries
are filled with the PCA mean for the projection and restored to NaN afterwards (documented deviation).
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import stats
from scipy.linalg import qr
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)

METHODS = ("pearson", "spearman", "gls")


# ----------------------------------------------------------------------------------------- utilities
def _corr_cols(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Pearson correlation between columns of A (n x m) [and B (n x k)], no NaN allowed. float64."""
    A = A - A.mean(0)
    na = np.sqrt((A * A).sum(0))
    if B is None:
        B, nb = A, na
    else:
        B = B - B.mean(0)
        nb = np.sqrt((B * B).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = (A.T @ B) / np.outer(na, nb)
    return np.clip(C, -1.0, 1.0)


def _pattern_blocks(Xg: np.ndarray, la: np.ndarray, lb: np.ndarray):
    """Group pairs (local gene indices la, lb into Xg's columns) by the missingness patterns of the two genes.

    Yields (pair_idx, S, cols, pa, pb): pair_idx = positions in la/lb; S = boolean sample mask observed for both
    patterns; cols = local gene indices involved; pa/pb = positions of each pair's genes within cols."""
    valid = ~np.isnan(Xg)
    keys = [valid[:, j].tobytes() for j in range(Xg.shape[1])]
    uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    pid = np.array([uniq[k] for k in keys])
    pmask = {i: np.frombuffer(k, dtype=bool) for k, i in uniq.items()}
    p1, p2 = np.minimum(pid[la], pid[lb]), np.maximum(pid[la], pid[lb])
    code = p1 * len(uniq) + p2
    for c in np.unique(code):
        idx = np.flatnonzero(code == c)
        P, Q = divmod(int(c), len(uniq))
        S = pmask[P] & pmask[Q]
        cols = np.unique(np.r_[la[idx], lb[idx]])
        pos = {g: i for i, g in enumerate(cols)}
        pa = np.array([pos[g] for g in la[idx]])
        pb = np.array([pos[g] for g in lb[idx]])
        yield idx, S, cols, pa, pb


def _local(ia, ib):
    genes = np.unique(np.r_[ia, ib])
    m = {g: i for i, g in enumerate(genes)}
    return genes, np.array([m[g] for g in ia]), np.array([m[g] for g in ib])


# ------------------------------------------------------------------------------------- correlations
def pearson_pairs(X, ia, ib) -> dict:
    return _corr_pairs(X, ia, ib, rank=False)


def spearman_pairs(X, ia, ib) -> dict:
    return _corr_pairs(X, ia, ib, rank=True)


def _corr_pairs(X, ia, ib, rank):
    ia, ib = np.asarray(ia), np.asarray(ib)
    genes, la, lb = _local(ia, ib)
    Xg = np.asarray(X[:, genes], dtype=np.float64)
    r = np.full(len(ia), np.nan)
    n = np.zeros(len(ia), np.int64)
    for idx, S, cols, pa, pb in _pattern_blocks(Xg, la, lb):
        A = Xg[np.ix_(S, cols)]
        if rank:
            A = stats.rankdata(A, axis=0)
        C = _corr_cols(A)
        r[idx] = C[pa, pb]
        n[idx] = S.sum()
    bad = np.isnan(r)
    if bad.any():
        log.warning("%d pairs with undefined correlation (zero variance / <2 obs) left NaN", bad.sum())
    return dict(signed=r, abs=np.abs(r), n_obs=n)


# ------------------------------------------------------------------------------------------- GLS
def olfactory_correction(X: np.ndarray, olf_idx, n_pc: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Wainberg load_screens bias correction. X samples x genes; olf_idx: column indices of olfactory receptor
    genes. Returns (corrected X with the last n_pc retained samples dropped, boolean mask over X's rows of the
    samples returned).

    Adaptation: samples in which *every* olfactory gene is missing (DepMap 26Q3: the 118 KY-only models; the KY
    library has no olfactory-receptor guides) cannot be corrected and are excluded. As in their code
    (``reindex(olfactory_genes).dropna()``), olfactory genes with any NaN in the retained samples are not used to
    fit the PCA."""
    X = np.asarray(X, dtype=np.float64)
    olf_idx = np.asarray(olf_idx)
    keep = ~np.isnan(X[:, olf_idx]).all(1)
    D = X[keep].T                                              # genes x samples, as in their code
    olf = D[olf_idx]
    olf = olf[~np.isnan(olf).any(1)]
    pca = PCA(n_components=n_pc).fit(olf)
    nan = np.isnan(D)
    Df = np.where(nan, pca.mean_[None, :], D)
    D = Df - pca.inverse_transform(pca.transform(Df))
    D[nan] = np.nan
    kept_rows = np.flatnonzero(keep)[:-n_pc]
    mask = np.zeros(X.shape[0], bool)
    mask[kept_rows] = True
    return D[:, :-n_pc].T, mask


def sample_covariance(X: np.ndarray) -> tuple[np.ndarray, int]:
    """Wainberg's np.cov(screens.T): samples x samples covariance with genes as observations, using genes with
    no NaN. Returns (Sigma, n_genes_used)."""
    ok = ~np.isnan(X).any(0)
    Xc = np.asarray(X[:, ok], dtype=np.float64)
    Xc = Xc - Xc.mean(1, keepdims=True)
    return (Xc @ Xc.T) / (Xc.shape[1] - 1), int(ok.sum())


def whitening(Sigma: np.ndarray, rank_deficiency: str = "drop") -> tuple[np.ndarray, np.ndarray]:
    """Whitening operator A (k x n) with A' A = Sigma^-1 on the kept samples, and the boolean mask of kept
    samples. Warped data = A @ y, warped intercept = A @ 1, df = k - 2.

    Full-rank Sigma: A = L', L = cholesky(pinv(Sigma)) exactly as Wainberg (their warped_screens = screens @ L).
    Rank-deficient Sigma (rank r < n; DepMap 26Q3 CRISPRGeneEffect has exactly 4 linear dependencies among
    models, the same situation Wainberg created with their 4-PC removal and fixed by dropping the last 4 cell
    lines):
      'drop' (default): drop the n - r samples identified as dependent by rank-revealing QR with column
              pivoting of Sigma (deterministic), then Cholesky on the rest. Dropping the *last* 4 samples as
              Wainberg did does not restore full rank for this release, hence the pivoted choice.
      'pinv': keep all samples; A = Lambda_r^-1/2 U_r' from the eigendecomposition (whitening on range(Sigma),
              k = r).
    """
    n = Sigma.shape[0]
    r = np.linalg.matrix_rank(Sigma, hermitian=True)
    keep = np.ones(n, bool)
    if r == n:
        return np.linalg.cholesky(np.linalg.pinv(Sigma)).T, keep
    if rank_deficiency == "drop":
        _, _, piv = qr(Sigma, pivoting=True, mode="economic")
        keep[piv[r:]] = False
        S = Sigma[np.ix_(keep, keep)]
        return np.linalg.cholesky(np.linalg.pinv(S)).T, keep
    if rank_deficiency == "pinv":
        w, U = np.linalg.eigh(Sigma)
        idx = np.argsort(w)[::-1][:r]
        return (U[:, idx] / np.sqrt(w[idx])).T, keep
    raise ValueError(rank_deficiency)


def _gls_t_block(Y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Y: n x m (samples x genes, no NaN), A: k x n whitening operator. Returns m x m GLS t matrix."""
    W = A @ Y                                    # warped genes as columns (== screens @ L in their layout)
    w0 = A.sum(1)                                # warped intercept (cholsigmainv.sum(axis=0))
    w0 = w0 / np.linalg.norm(w0)
    R = W - np.outer(w0, w0 @ W)                 # project out warped intercept
    nr = np.linalg.norm(R, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.clip((R.T @ R) / np.outer(nr, nr), -1.0, 1.0)
        df = A.shape[0] - 2
        t = r * np.sqrt(df / (1.0 - r * r))
    return t


def gls_pairs(X, ia, ib, Sigma: np.ndarray | None = None, rank_deficiency: str = "drop") -> dict:
    """GLS t-statistic / two-sided p for the given pairs. Sigma defaults to sample_covariance(X)."""
    ia, ib = np.asarray(ia), np.asarray(ib)
    if Sigma is None:
        Sigma, _ = sample_covariance(X)
    genes, la, lb = _local(ia, ib)
    Xg = np.asarray(X[:, genes], dtype=np.float64)
    t = np.full(len(ia), np.nan)
    df = np.zeros(len(ia), np.int64)
    n = np.zeros(len(ia), np.int64)
    cache, dropped = {}, {}
    for idx, S, cols, pa, pb in _pattern_blocks(Xg, la, lb):
        key = S.tobytes()
        if key not in cache:
            A, keep = whitening(Sigma[np.ix_(S, S)], rank_deficiency)
            Sk = np.flatnonzero(S)[keep]
            cache[key] = (A, Sk)
            dropped[int(S.sum())] = int((~keep).sum())
        A, Sk = cache[key]
        T = _gls_t_block(Xg[np.ix_(Sk, cols)], A)
        t[idx] = T[pa, pb]
        df[idx] = A.shape[0] - 2
        n[idx] = len(Sk)
    bad = np.isnan(t) | (df < 1)
    if bad.any():
        log.warning("%d GLS pairs undefined (too few shared samples / zero variance) left NaN", bad.sum())
        t[bad] = np.nan
    logp = np.log(2.0) + stats.t.logsf(np.abs(t), df)       # log two-sided p, no underflow
    log10p = logp / np.log(10)
    return dict(signed=t, abs=-log10p, p=np.exp(logp), log10p=log10p, n_obs=n, df=df,
                n_cov_blocks=len(cache), samples_dropped_for_rank={str(k): v for k, v in dropped.items()})


def gls_reference(screens: np.ndarray, Sigma: np.ndarray | None = None):
    """Line-by-line port of kundajelab/coessentiality gene_pairs.py (screens: genes x cells, no NaN).
    Returns (GLS_coef, GLS_se, t, p) as full gene x gene matrices. Slow; for tests only."""
    if Sigma is None:
        Sigma = np.cov(screens.T)
    cholsigmainv = np.linalg.cholesky(np.linalg.pinv(Sigma))
    warped_screens = screens @ cholsigmainv
    warped_intercept = cholsigmainv.sum(axis=0)
    G = len(warped_screens)
    GLS_coef, GLS_se = np.empty((G, G)), np.empty((G, G))
    ys = warped_screens.T
    for gene_index in range(G):
        Xd = np.stack((warped_intercept, warped_screens[gene_index]), axis=1)
        coef, residues = np.linalg.lstsq(Xd, ys, rcond=None)[:2]
        df = warped_screens.shape[1] - 2
        GLS_coef[gene_index] = coef[1]
        GLS_se[gene_index] = np.sqrt(np.linalg.pinv(Xd.T @ Xd)[1, 1] * residues / df)
    df = warped_screens.shape[1] - 2
    t = GLS_coef / GLS_se
    p = 2 * stats.t(df).sf(np.abs(t))
    np.fill_diagonal(p, 1)
    return GLS_coef, GLS_se, t, p


def score(method: str, X, ia, ib, **kw) -> dict:
    if method == "pearson":
        return pearson_pairs(X, ia, ib)
    if method == "spearman":
        return spearman_pairs(X, ia, ib)
    if method == "gls":
        return gls_pairs(X, ia, ib, **kw)
    raise ValueError(method)
