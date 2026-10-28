"""Ranking metrics for co-complex prediction and stratified bootstrap confidence intervals.

AUPRC = average precision (AP) as in sklearn.metrics.average_precision_score:
    AP = sum_n (R_n - R_{n-1}) P_n over the distinct score thresholds (no trapezoidal interpolation).

Ties: sklearn's precision_recall_curve uses one threshold per *distinct* score value, so all items sharing a
score enter the ranking together: within a tie block, precision is evaluated once, after the whole block
(i.e. the expected-value-free "block" treatment, neither optimistic nor pessimistic ordering). A constant
scorer therefore gets AP = prevalence exactly. ``ap_presorted`` below reproduces this rule and agrees with
sklearn to floating-point precision (tests/test_metrics.py).

Bootstrap: positives and negatives are resampled separately with replacement (stratified), keeping class
counts and hence prevalence fixed. Implemented with multiplicity weights on a pre-sorted score vector, so a
replicate costs O(n) instead of a sort. Paired differences use the same resampled indices for both scorers.
Percentile intervals. Seeds come from src.config.SEEDS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import PRIMARY_SEED

RECALL_LEVELS = (0.1, 0.25, 0.5)


def auprc(y, s) -> float:
    return float(average_precision_score(y, s))


def auroc(y, s) -> float:
    return float(roc_auc_score(y, s))


def precision_at_recall(y, s, levels=RECALL_LEVELS) -> dict:
    """Precision at the highest score threshold whose recall is >= level (non-interpolated: the operating
    point a user would get by thresholding the score so that at least `level` of positives are recovered;
    tie blocks are taken whole)."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    last = np.r_[ss[1:] != ss[:-1], True]           # end of each tie block
    tp = np.cumsum(ys)[last]
    n = np.arange(1, len(ys) + 1)[last]
    rec = tp / max(ys.sum(), 1)
    prec = tp / n
    out = {}
    for lv in levels:
        k = np.searchsorted(rec, lv - 1e-12)        # first block reaching the level
        out[f"P@R{lv:g}"] = float(prec[min(k, len(prec) - 1)])
    return out


class PresortedScores:
    """Scores sorted once (descending) with tie-block boundaries, for fast weighted AP."""

    def __init__(self, y, s):
        y = np.asarray(y).astype(bool)
        s = np.asarray(s, dtype=np.float64)
        if np.isnan(s).any():
            raise ValueError("NaN scores; impute or drop before evaluating")
        self.order = np.argsort(-s, kind="mergesort")
        self.ys = y[self.order]
        ss = s[self.order]
        self.block_end = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])

    def ap(self, w=None) -> float:
        """AP with per-item multiplicity weights w (None = all ones), sklearn tie semantics."""
        ys = self.ys
        ws = np.ones(len(ys)) if w is None else np.asarray(w, dtype=np.float64)[self.order]
        tp = np.cumsum(ws * ys)[self.block_end]
        fp = np.cumsum(ws * ~ys)[self.block_end]
        P = tp[-1]
        if P <= 0:
            return np.nan
        prec = tp / np.maximum(tp + fp, 1e-300)
        rec = tp / P
        drec = np.diff(np.r_[0.0, rec])
        return float(np.sum(drec * prec))


def _boot_weights(y, n_boot, seed):
    """Yield multiplicity weight vectors for stratified (pos/neg separately) bootstrap replicates."""
    y = np.asarray(y).astype(bool)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    n = len(y)
    for _ in range(n_boot):
        w = np.zeros(n)
        w[pos] = np.bincount(rng.integers(0, len(pos), len(pos)), minlength=len(pos))
        w[neg] = np.bincount(rng.integers(0, len(neg), len(neg)), minlength=len(neg))
        yield w


def bootstrap_auprc(y, s, n_boot=1000, seed=PRIMARY_SEED, alpha=0.05) -> dict:
    ps = PresortedScores(y, s)
    reps = np.array([ps.ap(w) for w in _boot_weights(y, n_boot, seed)])
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return dict(auprc=ps.ap(), ci_low=float(lo), ci_high=float(hi), boot_sd=float(reps.std(ddof=1)),
                n_boot=n_boot, seed=seed)


def bootstrap_auprc_diff(y, s1, s2, n_boot=1000, seed=PRIMARY_SEED, alpha=0.05) -> dict:
    """Paired stratified bootstrap of AUPRC(s1) - AUPRC(s2) on the same pairs. p_two_sided is the
    bootstrap percentile p-value for the null diff = 0 (2 * min tail fraction)."""
    p1, p2 = PresortedScores(y, s1), PresortedScores(y, s2)
    d = np.array([p1.ap(w) - p2.ap(w) for w in _boot_weights(y, n_boot, seed)])
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return dict(diff=p1.ap() - p2.ap(), ci_low=float(lo), ci_high=float(hi), p_two_sided=float(min(p, 1.0)),
                n_boot=n_boot, seed=seed)


def evaluate(y, s, n_boot=0, seed=PRIMARY_SEED) -> dict:
    """All point metrics (+ bootstrap AUPRC CI if n_boot > 0)."""
    y = np.asarray(y)
    out = dict(n_pos=int(y.sum()), n_neg=int((1 - y).sum()), prevalence=float(y.mean()),
               auprc=auprc(y, s), auroc=auroc(y, s))
    out.update(precision_at_recall(y, s))
    if n_boot:
        b = bootstrap_auprc(y, s, n_boot=n_boot, seed=seed)
        out.update(auprc_ci_low=b["ci_low"], auprc_ci_high=b["ci_high"], auprc_boot_sd=b["boot_sd"])
    return out


# ================================================================================ DESIGN v2 §3.5-3.6 metrics
"""Weighted AP family (v2).

* Horvitz-Thompson (HT) weighted AP -- PRIMARY (PI addendum 2, supersedes uniform reweighting). Under cluster
  grouping, a within-cluster pair (both proteins in the same §3.1 cluster) is in an evaluation subset whenever its
  cluster is (probability f), a between-cluster pair only when both clusters are (probability ~f^2). Weighting
  within-cluster pairs 1 and between-cluster pairs 1/f makes the subset's weighted pair mix estimate that of the
  full universe. Dev CV test fold: f = 1/K (weight K = 5). IFT: f = realised fraction of scorable proteins in IFT.
  Pooled dev: f = realised dev protein fraction.
* Uniform prevalence-reweighted AP (secondary; DESIGN §3.5 v2 original): all negatives get one common weight so
  that sum(w_neg)/sum(w_pos) = NEG_POS_RATIO (593,406/3,191, the scorable universe).
* Raw AP (secondary).
All weighted APs use sklearn ``average_precision_score(sample_weight=...)`` semantics (= AP with each item
replicated w times, ties as blocks); ``PresortedScores.ap(w)`` is the identical fast path used in bootstraps.

Cluster bootstrap (§3.6): resample the clusters of the evaluated subset with replacement; multiplicity m_c; a
within-cluster pair gets weight m_c, a between-cluster pair m_c1*m_c2 (copies of the same cluster drawn twice are
not paired with each other). The design weights (HT or uniform reweighting, recomputed per replicate) are then
applied on top. All scorers share the same replicates (paired). Percentile intervals.
"""
from sklearn.metrics import roc_auc_score as _roc_auc  # noqa: E402

from src.config import NEG_POS_RATIO  # noqa: E402


def ht_weights(ca, cb, f: float) -> np.ndarray:
    """1 for within-cluster pairs, 1/f for between-cluster pairs."""
    ca, cb = np.asarray(ca), np.asarray(cb)
    return np.where(ca == cb, 1.0, 1.0 / f)


def uniform_reweight(y, w=None, ratio: float = NEG_POS_RATIO) -> np.ndarray:
    """Scale negatives' weights so that sum(w_neg) / sum(w_pos) == ratio (positives keep w)."""
    y = np.asarray(y).astype(bool)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64).copy()
    sp, sn = w[y].sum(), w[~y].sum()
    if sp > 0 and sn > 0:
        w[~y] *= ratio * sp / sn
    return w


def weighted_ap(y, s, w=None) -> float:
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s, sample_weight=w))


def weighted_precision_at_recall(y, s, w=None, levels=RECALL_LEVELS) -> dict:
    """Weighted analogue of precision_at_recall (precision = weighted TP / weighted predicted positives)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    ys, ss, ws = y[order], s[order], w[order]
    last = np.r_[ss[1:] != ss[:-1], True]
    tp = np.cumsum(ws * ys)[last]
    tot = np.cumsum(ws)[last]
    rec = tp / max(tp[-1], 1e-300)
    prec = tp / tot
    out = {}
    for lv in levels:
        k = np.searchsorted(rec, lv - 1e-12)
        out[f"P@R{lv:g}"] = float(prec[min(k, len(prec) - 1)])
    return out


def _groups_touching(ca, cb):
    """cluster id -> array of pair indices with at least one endpoint in that cluster."""
    idx = np.r_[np.arange(len(ca)), np.arange(len(cb))]
    cl = np.r_[ca, cb]
    df = pd.DataFrame(dict(c=cl, i=idx)).drop_duplicates()
    return {c: g.i.values for c, g in df.groupby("c")}


def macro_cluster_ap(y, s, ca, cb, w) -> dict:
    """For each cluster with >=1 positive among the pairs touching it: weighted AP over those pairs. Unweighted mean
    over clusters."""
    y = np.asarray(y).astype(bool)
    aps = []
    for c, ix in _groups_touching(np.asarray(ca), np.asarray(cb)).items():
        if y[ix].any() and (~y[ix]).any():
            aps.append(weighted_ap(y[ix], s[ix], w[ix]))
    return dict(macro_cluster_ap=float(np.mean(aps)) if aps else float("nan"), n_clusters_macro=len(aps))


def metric_suite(y, s, w, ca, cb, comp_a=None, comp_b=None, ess_a=None, ess_b=None,
                 largest_component=None) -> dict:
    """All §3.5 metrics on one evaluation subset.

    w: design (HT) weights per pair (``ht_weights``); ca/cb: §3.1 cluster ids (macro per-cluster AP);
    comp_a/b: connected-component ids (hard negatives = same component; AP without the largest component);
    ess_a/b: common-essential flags per endpoint (pair essentiality strata)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if np.isnan(s).any():
        raise ValueError("NaN scores")
    ca, cb = np.asarray(ca), np.asarray(cb)
    out = dict(n_pos=int(y.sum()), n_neg=int((~y).sum()), prevalence=float(y.mean()),
               n_pos_between_clusters=int((y & (ca != cb)).sum()),
               ht_prevalence=float(w[y].sum() / w.sum()))
    out["ap_ht"] = weighted_ap(y, s, w)
    out["ap_reweighted"] = weighted_ap(y, s, uniform_reweight(y))
    out["ap_raw"] = weighted_ap(y, s)
    out["auroc_ht"] = float(_roc_auc(y, s, sample_weight=w)) if 0 < y.sum() < len(y) else float("nan")
    out.update({f"ht_{k}": v for k, v in weighted_precision_at_recall(y, s, w).items()})
    out.update(macro_cluster_ap(y, s, ca, cb, w))
    if comp_a is not None:
        comp_a, comp_b = np.asarray(comp_a), np.asarray(comp_b)
        if largest_component is not None:
            keep = (comp_a != largest_component) & (comp_b != largest_component)
            out["ap_ht_no_largest_component"] = weighted_ap(y[keep], s[keep], w[keep])
            out["n_pos_no_largest_component"] = int(y[keep].sum())
        hard = y | (comp_a == comp_b)
        out["ap_ht_hard_neg"] = weighted_ap(y[hard], s[hard], w[hard])
        out["n_hard_neg"] = int((hard & ~y).sum())
    if ess_a is not None:
        n_ess = np.asarray(ess_a).astype(int) + np.asarray(ess_b).astype(int)
        for k, name in enumerate(("both_nonessential", "mixed", "both_essential")):
            m = n_ess == k
            out[f"ap_ht_{name}"] = weighted_ap(y[m], s[m], w[m]) if m.any() else float("nan")
            out[f"n_pos_{name}"] = int(y[m].sum())
            out[f"n_neg_{name}"] = int((~y[m]).sum())
    return out


def cluster_bootstrap(y, scores: dict, ca, cb, f: float | None = None, n_boot: int = 1000, seed: int = PRIMARY_SEED,
                      alpha: float = 0.05, diffs=(), weighting: str = "ht", base_w=None) -> dict:
    """Paired cluster bootstrap of weighted AP for several scorers on the same pairs.

    weighting: 'ht' (design weights 1 / 1/f, f required, or explicit ``base_w``), 'reweighted' (uniform prevalence
    reweighting recomputed per replicate) or 'raw'. ca/cb are the resampling units (clusters). diffs: iterable of (name1, name2) for AP(name1) - AP(name2) and relative gain
    AP1/AP2 - 1. p_two_sided = 2 * min(P(d <= 0), P(d >= 0)) over replicates."""
    y = np.asarray(y).astype(bool)
    ca, cb = np.asarray(ca), np.asarray(cb)
    uniq, inv = np.unique(np.r_[ca, cb], return_inverse=True)
    ia, ib = inv[:len(ca)], inv[len(ca):]
    same = ia == ib
    C = len(uniq)
    if base_w is not None:
        base = np.asarray(base_w, dtype=np.float64)
    else:
        base = ht_weights(ca, cb, f) if weighting == "ht" else np.ones(len(y))
    ps = {k: PresortedScores(y, v) for k, v in scores.items()}

    def apply(w):
        return uniform_reweight(y, w) if weighting == "reweighted" else w

    w0 = apply(base)
    point = {k: p.ap(w0) for k, p in ps.items()}
    rng = np.random.default_rng(seed)
    reps = {k: np.empty(n_boot) for k in ps}
    for r in range(n_boot):
        m = np.bincount(rng.integers(0, C, C), minlength=C).astype(np.float64)
        w = np.where(same, m[ia], m[ia] * m[ib]) * base
        w = apply(w)
        for k, p in ps.items():
            reps[k][r] = p.ap(w)
    q = [alpha / 2, 1 - alpha / 2]
    out = dict(n_boot=n_boot, seed=seed, weighting=weighting, n_clusters=int(C), ap={}, diff={})
    for k in ps:
        lo, hi = np.nanquantile(reps[k], q)
        out["ap"][k] = dict(ap=point[k], ci_low=float(lo), ci_high=float(hi), boot_sd=float(np.nanstd(reps[k], ddof=1)))
    for a, b in diffs:
        d = reps[a] - reps[b]
        rel = reps[a] / reps[b] - 1
        lo, hi = np.nanquantile(d, q)
        rlo, rhi = np.nanquantile(rel, q)
        p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
        out["diff"][f"{a}-{b}"] = dict(diff=point[a] - point[b], ci_low=float(lo), ci_high=float(hi),
                                       rel_gain=point[a] / point[b] - 1, rel_ci_low=float(rlo), rel_ci_high=float(rhi),
                                       p_two_sided=float(min(p, 1.0)))
    return out


def holm(pvals: dict) -> dict:
    """Holm step-down adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out
