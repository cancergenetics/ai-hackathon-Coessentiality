"""Independent verification of the Phase-A best label-free layer rowz[socos_q0.25[spec_k20_t0.1_std]].

Re-implements, from the raw DepMap CSV and WITHOUT importing src/layers*.py, src/metrics.py, src/baselines.py or
src/evaluate.py:
  * gene centring / mean imputation / per-gene standardisation, background genes (NaN < 10%),
  * Sigma over BG genes in a Householder basis of 1-perp, top-k removal, ridge whitening (lam + tau*mean lam)^-1/2,
  * cosine (Pearson) of filtered profiles, genome-wide row moments, balanced z, top-q% neighbours,
  * partner-excluded weighted cosine of neighbour vectors, exact genome-wide row-z of that second-order layer,
  * weighted AP, Horvitz-Thompson weights, paired cluster bootstrap, pairwise-complete Pearson.
Only src/splits.make_split (fold assignment) is imported from the project. Dev pairs are re-derived from the
committed protein-assignment file (SHA-256 checked); rows with any IFT-side protein are dropped on load and no
metric is ever computed on them.

Usage: ./.env/bin/python scripts/verify/verify_phaseA_best.py
Outputs: results/verify/phaseA_best.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.splits import make_split  # noqa: E402  (fold assignment helper only)

CSV = ROOT / "datasets/DepMAP/CRISPRGeneEffect.csv"
ASSIGN = ROOT / "data_cache/splits/protein_assignment.csv"
ASSIGN_SHA = ROOT / "results/splits/protein_assignment.sha256"
PAIRS = ROOT / "data_cache/pairs.pkl"
OUT = ROOT / "results/verify/phaseA_best.json"
K_FOLDS = 5
SEED = 0
MAX_NAN = 0.10
K_TOP, TAU, Q = 20, 0.1, 0.25
N_BOOT = 1000
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- metrics (own)
def weighted_ap(y, s, w=None):
    """Step-wise AP: sum over distinct thresholds of (R_t - R_{t-1}) * P_t, ties grouped (sklearn convention)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, np.float64)
    o = np.argsort(-s, kind="mergesort")
    y, s, w = y[o], s[o], w[o]
    tp = np.cumsum(w * y)
    fp = np.cumsum(w * ~y)
    last = np.r_[s[1:] != s[:-1], True]          # last index of each tie group
    tp, fp = tp[last], fp[last]
    P = tp[-1]
    if P <= 0:
        return float("nan")
    prec = tp / (tp + fp)
    rec = tp / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def ht_w(ca, cb, K):
    return np.where(np.asarray(ca) == np.asarray(cb), 1.0, float(K))


def boot_ap_diff(y, s1, s2, ca, cb, base_w, n_boot=N_BOOT, seed=SEED):
    """Paired cluster bootstrap: resample clusters with replacement; within pairs weight m_c, between m_c1*m_c2."""
    y = np.asarray(y).astype(bool)
    uniq, inv = np.unique(np.r_[ca, cb], return_inverse=True)
    ia, ib = inv[:len(ca)], inv[len(ca):]
    same = ia == ib
    C = len(uniq)
    prep = []
    for s in (s1, s2):
        o = np.argsort(-s, kind="mergesort")
        ss = s[o]
        last = np.r_[ss[1:] != ss[:-1], True]
        prep.append((o, last))
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, 2))
    for r in range(n_boot):
        m = np.bincount(rng.integers(0, C, C), minlength=C).astype(np.float64)
        w = base_w * np.where(same, m[ia], m[ia] * m[ib])
        for j, (o, last) in enumerate(prep):
            yy, ww = y[o], w[o]
            tp = np.cumsum(ww * yy)[last]
            fp = np.cumsum(ww * ~yy)[last]
            P = tp[-1]
            prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
            out[r, j] = np.sum(np.diff(np.r_[0.0, tp / P]) * prec)
    d = out[:, 0] - out[:, 1]
    return dict(diff_ci=[float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                diff_boot_mean=float(d.mean()), frac_le0=float((d <= 0).mean()))


# ------------------------------------------------------------------------------------------------ data
def load_matrix():
    df = pd.read_csv(CSV, index_col=0)
    X = df.values.astype(np.float64)                     # lines x genes
    return X, list(df.columns)


def dev_pairs():
    sha = hashlib.sha256(ASSIGN.read_bytes()).hexdigest()
    committed = ASSIGN_SHA.read_text().split()[0]
    assert sha == committed, (sha, committed)
    a = pd.read_csv(ASSIGN)
    a = a[a.mapped]
    side = dict(zip(a.protein, a.side))
    clus = dict(zip(a.protein, a.cluster))
    allp = pd.read_pickle(PAIRS)
    sc = allp[allp.scorable].reset_index(drop=True)      # pair_id = row of scorable table (cache alignment)
    sc["pair_id"] = np.arange(len(sc))
    keep = (sc.a.map(side) == "dev").values & (sc.b.map(side) == "dev").values
    dev = sc[keep].reset_index(drop=True)[["a", "b", "label", "ia", "ib", "pair_id"]].copy()
    del allp
    dev["ca"] = dev.a.map(clus).astype(int)
    dev["cb"] = dev.b.map(clus).astype(int)
    f_dev = float((a.side == "dev").mean())
    return dev, sc[["ia", "ib"]].copy(), f_dev, sha


# ------------------------------------------------------------------------------------------------ layer
def householder_basis(n):
    u = np.ones(n) / np.sqrt(n)
    u[0] -= 1.0
    H = np.eye(n) - 2.0 * np.outer(u, u) / (u @ u)       # H e1 = 1/sqrt(n)
    return H[:, 1:]                                      # n x (n-1), orthonormal, orthogonal to 1


def spectral_profiles(X, k, tau, standardize, ddof=0):
    """Unit-norm filtered profiles (genes x (n-1)) + diagnostics."""
    nanf = np.isnan(X).mean(0)
    bg = nanf < MAX_NAN
    mu = np.nanmean(X, 0)
    Xc = np.where(np.isnan(X), 0.0, X - mu)
    if standardize:
        sd = np.nanstd(X, 0, ddof=ddof)
        Xc = Xc / sd
    Q = householder_basis(X.shape[0])
    Y = Xc.T @ Q                                         # genes x (n-1)
    Sig = Y[bg].T @ Y[bg] / bg.sum()
    lam, U = np.linalg.eigh(Sig)
    lam, U = lam[::-1], U[:, ::-1]
    if tau == 0:
        w = np.where(lam > len(lam) * np.finfo(float).eps * lam[0], 1.0 / np.sqrt(np.clip(lam, 1e-300, None)), 0.0)
    else:
        w = 1.0 / np.sqrt(lam + tau * lam.mean())
    w[:k] = 0.0
    F = (Y @ U) * w
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    return F, bg, dict(mean_eig=float(lam.mean()), n_bg=int(bg.sum()), top_frac=(lam[:5] / lam.sum()).tolist())


def second_order_rowz(F, bg, pairs_ia, pairs_ib, q=Q, block=1024):
    """Returns dict of per-pair arrays: spec (first order), socos, rowz_socos."""
    G = F.shape[0]
    bgi = np.flatnonzero(bg)
    nb = len(bgi)
    col_of = -np.ones(G, np.int64)
    col_of[bgi] = np.arange(nb)
    F32 = F.astype(np.float32)
    Fbg = np.ascontiguousarray(F32[bgi])
    # pass 1: row moments of S over BG \ {self}
    mean = np.zeros(G)
    sd = np.zeros(G)
    for s0 in range(0, G, block):
        rows = np.arange(s0, min(G, s0 + block))
        S = (F32[rows] @ Fbg.T).astype(np.float64)
        selfc = col_of[rows]
        hs = selfc >= 0
        S1 = S.sum(1)
        S2 = (S * S).sum(1)
        sv = np.zeros(len(rows))
        sv[hs] = S[np.flatnonzero(hs), selfc[hs]]
        n = nb - hs.astype(int)
        m = (S1 - sv) / n
        mean[rows] = m
        sd[rows] = np.sqrt(np.maximum((S2 - sv * sv) / n - m * m, 0))
    Kn = int(round(q / 100 * nb))
    log(f"  row moments done; K neighbours = {Kn}")
    # pass 2: balanced z and top-K neighbours over BG \ {self}
    mb, sb = mean[bgi], sd[bgi]
    nbr_idx = np.zeros((G, Kn), np.int64)
    nbr_val = np.zeros((G, Kn))
    for s0 in range(0, G, block):
        rows = np.arange(s0, min(G, s0 + block))
        S = (F32[rows] @ Fbg.T).astype(np.float64)
        B = 0.5 * ((S - mean[rows, None]) / sd[rows, None] + (S - mb[None, :]) / sb[None, :])
        selfc = col_of[rows]
        hs = np.flatnonzero(selfc >= 0)
        B[hs, selfc[hs]] = -np.inf
        part = np.argpartition(-B, Kn - 1, axis=1)[:, :Kn]
        vals = np.take_along_axis(B, part, 1)
        nbr_idx[rows] = part
        nbr_val[rows] = vals
    log("  neighbours done")
    Wv = np.maximum(nbr_val, 0.0)
    V = sps.csr_matrix((Wv.ravel(), nbr_idx.ravel(), np.arange(0, G * Kn + 1, Kn)), shape=(G, nb))
    norm2 = np.asarray(V.multiply(V).sum(1)).ravel()

    def vget(r, c_bgcol):
        """V[r, c] for arrays r (gene rows), c (BG column index, -1 = not BG -> 0)."""
        out = np.zeros(len(r))
        ok = c_bgcol >= 0
        if ok.any():
            out[ok] = np.asarray(V[r[ok], c_bgcol[ok]]).ravel()
        return out

    def so_pairs(ia, ib):
        dot = np.asarray(V[ia].multiply(V[ib]).sum(1)).ravel()
        na = norm2[ia] - vget(ia, col_of[ib]) ** 2
        nbv = norm2[ib] - vget(ib, col_of[ia]) ** 2
        den = np.sqrt(np.maximum(na, 0) * np.maximum(nbv, 0))
        return np.divide(dot, den, out=np.zeros_like(dot), where=den > 0)

    # exact genome-wide row moments of SO for genes that occur in pairs
    genes = np.unique(np.r_[pairs_ia, pairs_ib])
    Vbg = V[bgi].tocsr()
    Vbg_csc = Vbg.tocsc()
    so_mean = np.full(G, np.nan)
    so_sd = np.full(G, np.nan)
    for s0 in range(0, len(genes), 256):
        g = genes[s0:s0 + 256]
        Vr = V[g].toarray()                                      # r x nb
        D = (Vbg @ Vr.T).T                                       # r x nb  : v_a . v_j
        vaj2 = Vr ** 2                                           # v_a[j]^2
        cg = col_of[g]
        vja2 = np.zeros_like(D)                                  # v_j[a]^2
        ok = np.flatnonzero(cg >= 0)
        if len(ok):
            vja2[ok] = (Vbg_csc[:, cg[ok]].toarray().T) ** 2
        den = np.sqrt(np.maximum(norm2[g][:, None] - vaj2, 0) * np.maximum(norm2[bgi][None, :] - vja2, 0))
        SO = np.divide(D, den, out=np.zeros_like(D), where=den > 0)
        hs = cg >= 0
        n = nb - hs.astype(int)
        sv = np.zeros(len(g))
        sv[ok] = SO[ok, cg[ok]]
        m = (SO.sum(1) - sv) / n
        so_mean[g] = m
        so_sd[g] = np.sqrt(np.maximum(((SO * SO).sum(1) - sv * sv) / n - m * m, 0))
    log("  second-order row moments done")
    out = dict(spec=np.einsum("ij,ij->i", F[pairs_ia], F[pairs_ib]))
    so = np.empty(len(pairs_ia))
    for s0 in range(0, len(pairs_ia), 100000):
        sl = slice(s0, s0 + 100000)
        so[sl] = so_pairs(pairs_ia[sl], pairs_ib[sl])
    out["socos"] = so
    out["rowz_socos"] = 0.5 * ((so - so_mean[pairs_ia]) / so_sd[pairs_ia] + (so - so_mean[pairs_ib]) / so_sd[pairs_ib])
    out["_diag"] = dict(K=Kn, n_bg=nb, frac_zero_norm=float((norm2 == 0).mean()))
    return out


def pearson_pairwise(X, ia, ib, chunk=8000):
    r = np.empty(len(ia))
    for s0 in range(0, len(ia), chunk):
        A, B = X[:, ia[s0:s0 + chunk]], X[:, ib[s0:s0 + chunk]]
        M = ~np.isnan(A) & ~np.isnan(B)
        A0, B0 = np.where(M, A, 0.0), np.where(M, B, 0.0)
        n = M.sum(0)
        sa, sb = A0.sum(0), B0.sum(0)
        num = n * (A0 * B0).sum(0) - sa * sb
        den = np.sqrt((n * (A0 * A0).sum(0) - sa ** 2) * (n * (B0 * B0).sum(0) - sb ** 2))
        r[s0:s0 + chunk] = num / den
    return r


# ------------------------------------------------------------------------------------------------ eval
def fold_aps(dev, split, s):
    y = dev.label.values
    res = []
    for k in range(K_FOLDS):
        te = split.test_fold == k
        res.append(weighted_ap(y[te], s[te], ht_w(dev.ca.values[te], dev.cb.values[te], K_FOLDS)))
    return res


def per_cluster_ap(dev, split, s, clusters):
    """For cluster c (in its test fold): pairs of that fold with >=1 protein in c; HT weights; AP."""
    y = dev.label.values
    out = {}
    for c in clusters:
        m = (split.test_fold >= 0) & ((dev.ca.values == c) | (dev.cb.values == c))
        out[int(c)] = weighted_ap(y[m], s[m], ht_w(dev.ca.values[m], dev.cb.values[m], K_FOLDS))
    return out


def summarize(arr):
    return dict(mean=float(np.mean(arr)), sd=float(np.std(arr, ddof=1)), folds=[float(x) for x in arr])


def main():
    res = dict(script="scripts/verify/verify_phaseA_best.py", layer="rowz[socos_q0.25[spec_k20_t0.1_std]]")
    X, cols = load_matrix()
    genes_cache = (ROOT / "data_cache/CRISPRGeneEffect.genes.txt").read_text().splitlines()
    res["csv_shape"] = list(X.shape)
    res["csv_columns_equal_cache_gene_list"] = cols == genes_cache
    log(f"matrix {X.shape}; columns == cache gene list: {cols == genes_cache}")
    dev, sc, f_dev, sha = dev_pairs()
    res["assignment_sha256"] = sha
    res["n_dev_pairs"], res["n_dev_pos"] = int(len(dev)), int(dev.label.sum())
    log(f"dev pairs {len(dev)}, pos {dev.label.sum()}, f_dev {f_dev:.4f}")
    split = make_split(dev[["a", "b", "label"]], "cluster", K_FOLDS, SEED,
                       clusters=dict(zip(np.r_[dev.a.values, dev.b.values], np.r_[dev.ca.values, dev.cb.values])))
    h = hashlib.sha256(split.test_fold.astype(np.int16).tobytes())
    h.update(split.group_a.astype(np.int16).tobytes()); h.update(split.group_b.astype(np.int16).tobytes())
    res["split_fingerprint"] = h.hexdigest()
    log(f"split fingerprint {res['split_fingerprint'][:12]} (reported 912b53116c50)")

    # cached scores (for comparison only)
    v2 = np.load(ROOT / "data_cache/layers/screen_v2.scores.f32.npy", mmap_mode="r")
    v1 = np.load(ROOT / "data_cache/layers/screen_v1.scores.f32.npy", mmap_mode="r")
    pz = np.load(ROOT / "data_cache/layers/screen_v2.pairs.npz")
    res["cache_pairs_aligned"] = bool(np.array_equal(pz["ia"], sc.ia.values) and np.array_equal(pz["ib"], sc.ib.values))
    bz = np.load(ROOT / "data_cache/scores/CRISPRGeneEffect_all.npz")
    pid = dev.pair_id.values
    cached = dict(rowz_socos=np.asarray(v2[:, 17])[pid], socos=np.asarray(v2[:, 10])[pid],
                  spec=np.asarray(v1[:, 62])[pid], gls_signed=bz["gls_signed"][pid], pearson_signed=bz["pearson_signed"][pid])
    ia, ib = dev.ia.values, dev.ib.values

    # standardisation ddof diagnostic (manifest mean_eig 0.99863 for eig_gene_std)
    F, bg, d0 = spectral_profiles(X, K_TOP, TAU, True, ddof=0)
    res["spectral_diag_ddof0"] = d0
    log(f"spectral: n_bg {d0['n_bg']}, mean_eig {d0['mean_eig']:.6f} (manifest 0.998630)")
    mine = second_order_rowz(F, bg, ia, ib)
    res["second_order_diag"] = mine.pop("_diag")
    mine["pearson_signed"] = pearson_pairwise(X, ia, ib)
    Fg, _, dg = spectral_profiles(X, 0, 0, False)
    mine["glslike_k0_t0_imputed"] = np.einsum("ij,ij->i", Fg[ia], Fg[ib])
    res["glslike_diag"] = dg
    del Fg

    # (1) score agreement
    agree = {}
    for key, ref in (("rowz_socos", "rowz_socos"), ("socos", "socos"), ("spec", "spec"), ("pearson_signed", "pearson_signed")):
        a, b = mine[key], cached[ref].astype(np.float64)
        agree[key] = dict(pearson_r=float(np.corrcoef(a, b)[0, 1]), max_abs_diff=float(np.max(np.abs(a - b))),
                          mean_abs_diff=float(np.mean(np.abs(a - b))))
    res["score_agreement_vs_cache"] = agree
    log(f"agreement: {json.dumps(agree)}")

    # (2) HT-AP per fold
    rep = {}
    for name, f in (("rowz_socos", "results/phaseA/screen/rowz_socos_q0.25_spec_k20_t0.1_std_cluster_seed0.json"),
                    ("socos", "results/phaseA/screen/socos_q0.25_spec_k20_t0.1_std_cluster_seed0.json"),
                    ("gls_signed", "results/baselines_v2/gls_signed_cluster_seed0.json"),
                    ("pearson_signed", "results/baselines_v2/pearson_signed_cluster_seed0.json")):
        d = json.load(open(ROOT / f))
        rep[name] = dict(folds=[fo["ap_ht"] for fo in d["folds"]], pooled_ht_ap=d["pooled"]["ap_ht"],
                         fingerprint=d["split"]["fingerprint"])
    y = dev.label.values
    w_pool = np.where(dev.ca.values == dev.cb.values, 1.0, 1.0 / f_dev)
    ht = {}
    for name, s in (("rowz_socos[mine]", mine["rowz_socos"]), ("socos[mine]", mine["socos"]),
                    ("spec_k20_t0.1_std[mine]", mine["spec"]), ("pearson_signed[mine]", mine["pearson_signed"]),
                    ("glslike_k0_t0_imputed[mine]", mine["glslike_k0_t0_imputed"]),
                    ("rowz_socos[cache]", cached["rowz_socos"]), ("gls_signed[cache]", cached["gls_signed"]),
                    ("pearson_signed[cache]", cached["pearson_signed"])):
        fa = fold_aps(dev, split, s)
        ht[name] = dict(**summarize(fa), pooled_ht_ap_all_dev=weighted_ap(y, s, w_pool),
                        pooled_ap_natural=weighted_ap(y, s))
        log(f"{name}: HT-AP {np.mean(fa):.4f} folds {np.round(fa, 4).tolist()}  pooled {ht[name]['pooled_ht_ap_all_dev']:.4f}")
    res["ht_ap"] = ht
    res["reported"] = rep
    prev = []
    for k in range(K_FOLDS):
        te = split.test_fold == k
        w = ht_w(dev.ca.values[te], dev.cb.values[te], K_FOLDS)
        prev.append(float((w * y[te]).sum() / w.sum()))
    res["ht_prevalence_per_fold"] = prev

    # pooled OOF Δ vs GLS with paired cluster bootstrap
    oof = split.test_fold >= 0
    wo = ht_w(dev.ca.values[oof], dev.cb.values[oof], K_FOLDS)
    ap1 = weighted_ap(y[oof], mine["rowz_socos"][oof], wo)
    ap2 = weighted_ap(y[oof], cached["gls_signed"][oof], wo)
    ap3 = weighted_ap(y[oof], mine["pearson_signed"][oof], wo)
    b = boot_ap_diff(y[oof], mine["rowz_socos"][oof], cached["gls_signed"][oof], dev.ca.values[oof], dev.cb.values[oof], wo)
    res["pooled_oof"] = dict(n_pairs=int(oof.sum()), n_pos=int(y[oof].sum()), ap_rowz_socos=ap1, ap_gls=ap2,
                             ap_pearson_mine=ap3, delta_vs_gls=ap1 - ap2, rel_gain=ap1 / ap2 - 1, **b,
                             reported="+0.155 [+0.065, +0.246] (+60%)")
    log(f"pooled OOF: best {ap1:.4f} gls {ap2:.4f} delta {ap1 - ap2:+.4f} CI {b['diff_ci']}")

    # (5) per-cluster AP, top-10 dev clusters by within-cluster positives; leave-cluster-out delta
    pos = dev[(dev.label == 1) & (dev.ca == dev.cb) & oof]
    top = pos.ca.value_counts().head(10)
    pc_best = per_cluster_ap(dev, split, mine["rowz_socos"], top.index)
    pc_gls = per_cluster_ap(dev, split, cached["gls_signed"], top.index)
    rows = []
    for c, npos in top.items():
        m = oof & ~((dev.ca.values == c) | (dev.cb.values == c))
        wd = ht_w(dev.ca.values[m], dev.cb.values[m], K_FOLDS)
        dl = weighted_ap(y[m], mine["rowz_socos"][m], wd) - weighted_ap(y[m], cached["gls_signed"][m], wd)
        rows.append(dict(cluster=int(c), n_within_pos=int(npos), fold=int(split.protein_group[pos[pos.ca == c].a.iloc[0]]),
                         ap_best=pc_best[int(c)], ap_gls=pc_gls[int(c)], delta=pc_best[int(c)] - pc_gls[int(c)],
                         pooled_oof_delta_without_cluster=dl))
    res["top10_clusters"] = rows
    m = oof & ~(np.isin(dev.ca.values, top.index) | np.isin(dev.cb.values, top.index))
    wd = ht_w(dev.ca.values[m], dev.cb.values[m], K_FOLDS)
    a1, a2 = weighted_ap(y[m], mine["rowz_socos"][m], wd), weighted_ap(y[m], cached["gls_signed"][m], wd)
    b2 = boot_ap_diff(y[m], mine["rowz_socos"][m], cached["gls_signed"][m], dev.ca.values[m], dev.cb.values[m], wd)
    res["pooled_oof_without_top10"] = dict(n_pos=int(y[m].sum()), ap_best=a1, ap_gls=a2, delta=a1 - a2, **b2)
    # all-cluster per-cluster win rate
    allc = np.unique(pos.ca.values)
    pcb = per_cluster_ap(dev, split, mine["rowz_socos"], allc)
    pcg = per_cluster_ap(dev, split, cached["gls_signed"], allc)
    dd = np.array([pcb[int(c)] - pcg[int(c)] for c in allc])
    res["per_cluster_all"] = dict(n_clusters=int(len(allc)), frac_best_wins=float((dd > 0).mean()),
                                  frac_ties=float((dd == 0).mean()), median_delta=float(np.median(dd)),
                                  macro_ap_best=float(np.mean(list(pcb.values()))), macro_ap_gls=float(np.mean(list(pcg.values()))))
    log(f"per-cluster: {res['per_cluster_all']}; without top10 {res['pooled_oof_without_top10']}")

    # (4) negative control: shuffle profiles across genes, rerun the full pipeline
    rng = np.random.default_rng(SEED)
    Xp = X[:, rng.permutation(X.shape[1])]
    Fp, bgp, _ = spectral_profiles(Xp, K_TOP, TAU, True)
    perm = second_order_rowz(Fp, bgp, ia, ib)
    perm.pop("_diag")
    del Fp, Xp
    fa = fold_aps(dev, split, perm["rowz_socos"])
    res["negative_control_gene_shuffle"] = dict(ht_ap=summarize(fa), ht_prevalence_mean=float(np.mean(prev)),
                                                ratio_to_prevalence=float(np.mean(fa) / np.mean(prev)),
                                                pooled_oof_ap=weighted_ap(y[oof], perm["rowz_socos"][oof], wo))
    log(f"negative control: HT-AP {np.mean(fa):.4f} vs prevalence {np.mean(prev):.4f}")
    res["runtime_s"] = round(time.time() - T0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
