"""Phase-B supervised pair scorers (DESIGN v3 §4 Phase B): shallow monotone LightGBM and L2 logistic regression.

Interface (src/scorers.py, src/evaluate.py): ``bind(universe)`` builds the label-free pair feature matrix for every
pair of the universe (timed as setup); ``fit(train_idx)`` reads labels of ``train_idx`` only; ``score(idx)`` returns
the bag-averaged raw margin (log-odds) of each pair from its own feature row only (batch/order invariant; all
features are symmetric in (a, b)).

Feature blocks (each a ``FeatureBlock`` with ``build(universe) -> (F, names, monotone)``):
  LayerFeatures      cached Phase-A layers (data_cache/layers/<tag>.scores.f32.npy, rows = scorable pair_id),
                     z-scored with the frozen genome-wide null moments in the manifest; monotone +1.
  BaselineFeatures   Pearson / Spearman / GLS baseline scores (src/scorers.make_baseline); monotone +1.
  ArrayFeatures      any per-pair_id arrays (toys/tests); monotone given.
  GeneFeatures       src/gene_features: min/max of per-gene CRISPRGeneEffect stats, binned (<= 5 genome-wide
                     quantile bins, default) or raw ('raw_gene_feats' ablation); monotone 0.
  ChromFeatures      same chromosome / same chromosome arm from HGNC cytoband (diagnostic only); monotone 0.

Training (``fit``), all randomness from ``seed`` (a member of src.config.SEEDS) via numpy SeedSequence:
  * labels: ``universe.pairs.label`` at train_idx only, read as float; NaN or values outside {0,1} raise (tests
    pass a universe whose held-out labels are NaN).
  * cluster-balanced weights: n_c = number of training positives touching §3.1 cluster c. A positive pair
    within cluster c gets weight 1/sqrt(n_c); a positive spanning clusters c1 != c2 gets the mean of the two.
    Positive weights are rescaled to mean 1 (total positive weight unchanged, i.e. NO class re-weighting);
    negatives weight 1. Purpose: the largest communities (up to ~26% of positives in the giant component) do not
    dominate the fit; sqrt (not 1/n) keeps large complexes informative.
  * inner validation: the training pairs are split with the same cluster-grouped randomised scheme
    (src.splits.make_split, 4 inner folds, inner fold 0 = validation: pairs with both proteins in inner fold 0;
    inner-train = both proteins outside it).
  * bagging: ``n_bags`` bags; each keeps all training positives and a random sample of ``neg_ratio`` x n_pos
    training negatives (without replacement; all if fewer).
  * LightGBM: per bag, fit on the inner-train part of the bag with early stopping on the full (un-subsampled)
    validation pairs (metric average precision, unweighted), then refit on the whole bag with the best iteration
    count. Constraints: num_leaves <= 7, min_child_samples >= 2000 on real data (enforced unless
    ``allow_small_leaves`` for toys), monotone increasing in similarity features (``monotone=False`` switches the
    constraints off; used only for the label-shuffle control, where the constraint itself encodes the prior that
    higher similarity = more likely co-complex).
  * Logistic regression: StandardScaler (fit on the bag) + L2 LogisticRegression (C fixed, default 1.0) per bag,
    same bags and weights; no early stopping (inner validation AP is recorded only).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.config import CACHE, RESULTS, SEEDS
from src.scorers import Scorer

LGBM_DEFAULTS = dict(objective="binary", num_leaves=7, learning_rate=0.05, n_estimators=1500, min_child_samples=2000,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, min_split_gain=0.0,
                     max_bin=255, n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True,
                     monotone_constraints_method="intermediate")
EARLY_STOP = 100
LAYER_TAG = "screen_v1"


# ------------------------------------------------------------------------------------------ feature blocks
class FeatureBlock:
    kind = "block"

    def build(self, u):
        raise NotImplementedError

    def describe(self) -> dict:
        return dict(kind=self.kind)


_LAYER_MEMO: dict = {}


def load_layer_cache(tag: str = LAYER_TAG):
    """(mmap scores [n_scorable x n_layers], manifest dict, name -> column). Verifies the pair alignment once."""
    if tag not in _LAYER_MEMO:
        man = json.loads((CACHE / "layers" / f"{tag}.manifest.json").read_text())
        M = np.load(CACHE / "layers" / f"{tag}.scores.f32.npy", mmap_mode="r")
        from src.universe import scorable_table
        sc = scorable_table()
        h = hashlib.sha256()
        for a in (sc.ia.values.astype(np.int64), sc.ib.values.astype(np.int64)):
            h.update(np.ascontiguousarray(a).tobytes())
        if h.hexdigest() != man["pairs"]["sha256_ia_ib"] or M.shape[0] != len(sc):
            raise RuntimeError("layer cache rows are not aligned to the scorable pair table")
        _LAYER_MEMO[tag] = (M, man, {c["name"]: c for c in man["columns"]})
    return _LAYER_MEMO[tag]


class LayerFeatures(FeatureBlock):
    kind = "layers"

    def __init__(self, names, tag: str = LAYER_TAG):
        self.names, self.tag = list(names), tag

    def build(self, u):
        M, _, cols = load_layer_cache(self.tag)
        missing = [n for n in self.names if n not in cols]
        if missing:
            raise KeyError(f"layers not in cache {self.tag}: {missing}")
        pid = u.col("pair_id")
        j = [cols[n]["col"] for n in self.names]
        raw = np.asarray(M[np.sort(pid)][:, j], dtype=np.float64)          # sorted read for mmap locality
        F = np.empty_like(raw)
        F[np.argsort(pid, kind="stable")] = raw
        for t, n in enumerate(self.names):
            F[:, t] = (F[:, t] - cols[n]["null_mean"]) / cols[n]["null_sd"]
        if not np.isfinite(F).all():
            raise ValueError("non-finite layer values")
        return F, list(self.names), [1] * len(self.names)

    def describe(self):
        return dict(kind=self.kind, names=self.names, tag=self.tag, z="manifest genome-wide null moments")


_BASE_MEMO: dict = {}


class BaselineFeatures(FeatureBlock):
    kind = "baselines"

    def __init__(self, variants=("pearson_signed", "spearman_signed", "gls_signed"), mat=None):
        self.variants, self.mat = list(variants), mat

    def build(self, u):
        from src import scorers
        cols = []
        for v in self.variants:
            key = (v, u.name, len(u), int(u.col("pair_id")[:50].sum())) if self.mat is None else None
            if key is not None and key in _BASE_MEMO:
                cols.append(_BASE_MEMO[key])
                continue
            mat = self.mat if self.mat is not None else scorers.real_matrix()
            s = scorers.make_baseline(v, mat).bind(u).score(np.arange(len(u)))
            if key is not None:
                _BASE_MEMO[key] = s
            cols.append(s)
        return np.column_stack(cols), list(self.variants), [1] * len(self.variants)

    def describe(self):
        return dict(kind=self.kind, variants=self.variants)


class ArrayFeatures(FeatureBlock):
    kind = "arrays"

    def __init__(self, arrays: dict, monotone: int = 1):
        self.arrays, self.monotone = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()}, monotone

    def build(self, u):
        pid = u.col("pair_id")
        return (np.column_stack([a[pid] for a in self.arrays.values()]), list(self.arrays),
                [self.monotone] * len(self.arrays))

    def describe(self):
        return dict(kind=self.kind, names=list(self.arrays), monotone=self.monotone)


class GeneFeatures(FeatureBlock):
    """mode 'binned' (default, <= 5 genome-wide quantile bins) or 'raw' (ablation 'raw_gene_feats').
    table: per-gene stats (genes x stats) for toys; default = real CRISPRGeneEffect tables."""
    kind = "gene"

    def __init__(self, mode: str = "binned", table: np.ndarray | None = None, names=None):
        if mode not in ("binned", "raw"):
            raise ValueError(mode)
        self.mode, self.table, self.stat_names = mode, table, names

    def build(self, u):
        from src import gene_features as gf
        if self.table is not None:
            st = np.asarray(self.table, dtype=np.float64)
            names = self.stat_names or [f"s{j}" for j in range(st.shape[1])]
            G = st if self.mode == "raw" else gf.bin_stats(st, gf.quantile_edges(st))
        else:
            t = gf.real_gene_tables()
            names = list(gf.STATS)
            G = t["stats"] if self.mode == "raw" else t["bins"]
        F, cols = gf.pair_features(G, u.col("ia"), u.col("ib"), names)
        tag = "graw" if self.mode == "raw" else "gbin"
        return F, [f"{tag}_{c}" for c in cols], [0] * F.shape[1]

    def describe(self):
        return dict(kind=self.kind, mode=self.mode, stats=self.stat_names or "gene_features.STATS",
                    binning="<=5 genome-wide quantile bins" if self.mode == "binned" else "raw")


class ChromFeatures(FeatureBlock):
    """same_chrom, same_arm (HGNC cytoband; MT counted as a chromosome without arms). Diagnostic only."""
    kind = "chrom"

    def __init__(self, chrom=None, arm=None):
        self.chrom, self.arm = chrom, arm

    def build(self, u):
        chrom, arm = self.chrom, self.arm
        if chrom is None:
            from src.data import depmap_columns, gene_annotation
            ann = gene_annotation(depmap_columns("CRISPRGeneEffect"))
            chrom, arm = ann.chrom.values, ann.arm.values
        chrom = np.array([c if isinstance(c, str) else "" for c in chrom], dtype=object)
        arm = np.array([a if isinstance(a, str) else "" for a in arm], dtype=object)
        ia, ib = u.col("ia"), u.col("ib")
        same_c = (chrom[ia] == chrom[ib]) & (chrom[ia] != "")
        same_a = same_c & (arm[ia] == arm[ib]) & (arm[ia] != "")
        return np.c_[same_c, same_a].astype(np.float64), ["same_chrom", "same_arm"], [0, 0]

    def describe(self):
        return dict(kind=self.kind, features=["same_chrom", "same_arm"], source="HGNC location")


# ------------------------------------------------------------------------------------------ helpers
def cluster_balanced_weights(y, ca, cb) -> np.ndarray:
    """Positive weights 1/sqrt(n_pos touching the cluster) (mean over the two clusters for spanning positives),
    rescaled to mean 1; negatives 1."""
    y = np.asarray(y).astype(bool)
    ca, cb = np.asarray(ca), np.asarray(cb)
    w = np.ones(len(y))
    if not y.any():
        return w
    pa, pb = ca[y], cb[y]
    uniq, inv = np.unique(np.r_[pa, pb], return_inverse=True)
    ia, ib = inv[:len(pa)], inv[len(pa):]
    n = np.bincount(ia, minlength=len(uniq)) + np.bincount(ib[ia != ib], minlength=len(uniq))
    wp = 0.5 * (1 / np.sqrt(n[ia]) + 1 / np.sqrt(n[ib]))
    w[y] = wp / wp.mean()
    return w


def _ap(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


# ------------------------------------------------------------------------------------------ scorer
class SupervisedPairScorer(Scorer):
    label_free = False

    def __init__(self, name: str, blocks: list, model: str = "lgbm", seed: int = SEEDS[0], n_bags: int = 5,
                 neg_ratio: float = 50.0, weighting: str = "cluster_sqrt", lgbm_params: dict | None = None,
                 C: float = 1.0, n_inner: int = 4, early_stop: int = EARLY_STOP, allow_small_leaves: bool = False,
                 monotone: bool = True):
        super().__init__()
        if model not in ("lgbm", "logreg"):
            raise ValueError(model)
        if seed not in SEEDS:
            raise ValueError("seed must come from src.config.SEEDS")
        if weighting not in ("cluster_sqrt", "none"):
            raise ValueError(weighting)
        self.name, self.blocks, self.model, self.seed = name, list(blocks), model, seed
        self.n_bags, self.neg_ratio, self.weighting, self.C, self.n_inner = n_bags, neg_ratio, weighting, C, n_inner
        self.early_stop, self.use_monotone = early_stop, monotone
        self.lgbm_params = {**LGBM_DEFAULTS, **(lgbm_params or {})}
        if not allow_small_leaves and (self.lgbm_params["num_leaves"] > 7 or self.lgbm_params["min_child_samples"] < 2000):
            raise ValueError("DESIGN §4 B: num_leaves <= 7 and min_child_samples >= 2000")
        self.params = dict(model=model, seed=seed, n_bags=n_bags, neg_ratio=neg_ratio, weighting=weighting,
                           monotone_constraints=monotone,
                           blocks=[b.describe() for b in self.blocks], n_inner=n_inner,
                           **(dict(lgbm=self.lgbm_params, early_stop=early_stop) if model == "lgbm" else dict(C=C)))
        self.models, self.fit_log, self._fit_count = [], [], 0

    # -------------------------------------------------------------- features
    def features(self, u):
        Fs, names, mono = [], [], []
        for b in self.blocks:
            F, n, m = b.build(u)
            if F.shape[0] != len(u):
                raise ValueError(f"block {b.kind}: {F.shape[0]} rows for {len(u)} pairs")
            Fs.append(F)
            names += n
            mono += m
        return np.hstack(Fs), names, mono

    def bind(self, universe):
        super().bind(universe)
        t0 = time.perf_counter()
        self.F, self.feature_names, self.monotone = self.features(universe)
        if not self.use_monotone:
            self.monotone = [0] * len(self.monotone)
        self.params["feature_names"] = self.feature_names
        self.params["monotone"] = self.monotone
        self.bind_seconds = time.perf_counter() - t0
        self.models = []
        return self

    def _labels(self, idx):
        y = self.u.pairs["label"].to_numpy(dtype=np.float64)[np.asarray(idx)]
        if not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("training labels must be finite 0/1 (a held-out label was requested?)")
        return y.astype(np.int8)

    # -------------------------------------------------------------- fit
    def _inner_split(self, train_idx, y):
        from src import splits
        p = self.u.pairs.iloc[train_idx][["a", "b"]].reset_index(drop=True)
        p["label"] = y
        cmap = dict(zip(np.r_[p.a.values, p.b.values],
                        np.r_[self.u.col("cluster_a")[train_idx], self.u.col("cluster_b")[train_idx]]))
        ss = np.random.SeedSequence([self.seed, self._fit_count, 7])
        sp = splits.make_split(p, "cluster", self.n_inner, int(ss.generate_state(1)[0] % (2**31)), clusters=cmap)
        return sp.train_mask(0), sp.test_mask(0)

    def fit(self, train_idx):
        train_idx = np.asarray(train_idx)
        y = self._labels(train_idx)
        X = self.F[train_idx]
        w = (cluster_balanced_weights(y, self.u.col("cluster_a")[train_idx], self.u.col("cluster_b")[train_idx])
             if self.weighting == "cluster_sqrt" else np.ones(len(y)))
        itr, val = self._inner_split(train_idx, y)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        n_neg = int(min(len(neg), round(self.neg_ratio * len(pos))))
        ss = np.random.SeedSequence([self.seed, self._fit_count])
        self._fit_count += 1
        self.models, log = [], dict(n_train=int(len(y)), n_pos=int(len(pos)), n_val=int(val.sum()),
                                    n_val_pos=int(y[val].sum()), bags=[])
        for b, child in enumerate(ss.spawn(self.n_bags)):
            rng = np.random.default_rng(child)
            bag = np.sort(np.r_[pos, rng.choice(neg, n_neg, replace=False)])
            bseed = int(child.generate_state(1)[0] % (2**31))
            if self.model == "lgbm":
                m, info = self._fit_lgbm(X, y, w, bag, itr, val, bseed)
            else:
                m, info = self._fit_logreg(X, y, w, bag, val, bseed)
            self.models.append(m)
            log["bags"].append(info)
        self.fit_log.append(log)
        return self

    def _fit_lgbm(self, X, y, w, bag, itr, val, bseed):
        import lightgbm as lgb
        prm = {**self.lgbm_params, "random_state": bseed, "monotone_constraints": self.monotone}
        info = {}
        n_iter = prm["n_estimators"]
        inner = bag[itr[bag]]
        if y[val].sum() > 0 and y[inner].sum() > 0:
            es = lgb.LGBMClassifier(**prm)
            es.fit(X[inner], y[inner], sample_weight=w[inner], eval_X=(X[val],), eval_y=(y[val],),
                   eval_metric="average_precision",
                   callbacks=[lgb.early_stopping(self.early_stop, first_metric_only=True, verbose=False)])
            n_iter = max(int(es.best_iteration_ or prm["n_estimators"]), 1)
            info["val_ap"] = float(es.best_score_["valid_0"]["average_precision"])
        info["n_iter"] = n_iter
        m = lgb.LGBMClassifier(**{**prm, "n_estimators": n_iter})
        m.fit(X[bag], y[bag], sample_weight=w[bag])
        return m, info

    def _fit_logreg(self, X, y, w, bag, val, bseed):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        m = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=5000, random_state=bseed))
        m.fit(X[bag], y[bag], logisticregression__sample_weight=w[bag])
        info = dict(val_ap_after_fit=_ap(y[val], m.decision_function(X[val])) if val.any() else float("nan"))
        return m, info

    # -------------------------------------------------------------- score
    def predict_features(self, F) -> np.ndarray:
        if not self.models:
            raise RuntimeError("fit first")
        F = np.asarray(F, dtype=np.float64)
        if self.model == "lgbm":
            out = [m.predict(F, raw_score=True) for m in self.models]
        else:
            lr = [m[-1] for m in self.models]
            Z = [m[0].transform(F) for m in self.models]
            out = [(z * l.coef_[0]).sum(1) + l.intercept_[0] for z, l in zip(Z, lr)]   # row-wise (batch-invariant)
        return np.mean(out, axis=0)

    def score(self, idx):
        return self.predict_features(self.F[np.asarray(idx)])

    def feature_importance(self) -> dict:
        if self.model == "lgbm":
            g = np.mean([m.booster_.feature_importance("gain") for m in self.models], axis=0)
            g = g / max(g.sum(), 1e-300)
        else:
            g = np.mean([m[-1].coef_[0] for m in self.models], axis=0)
        return dict(zip(self.feature_names, map(float, g)))


# ------------------------------------------------------------------------------------------ factories
def make_supervised(name, blocks, model="lgbm", seed=SEEDS[0], **kw) -> SupervisedPairScorer:
    return SupervisedPairScorer(name, blocks, model=model, seed=seed, **kw)


class ZMeanScorer(Scorer):
    """Label-free reference: unweighted mean of manifest-z-scored cached layers (A6-style)."""
    label_free = True

    def __init__(self, name, layer_names, tag: str = LAYER_TAG):
        super().__init__()
        self.name, self.block = name, LayerFeatures(layer_names, tag)
        self.params = dict(layers=list(layer_names), tag=tag, combine="mean of null-z")

    def bind(self, universe):
        super().bind(universe)
        self.s = self.block.build(universe)[0].mean(1)
        return self

    def score(self, idx):
        return self.s[np.asarray(idx)]


PHASEB_OUT = RESULTS / "phaseB"
