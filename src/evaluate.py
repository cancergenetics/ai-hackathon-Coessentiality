"""Single development-evaluation entry point (DESIGN v2 §3.3-3.6, §5, §6).

``run_dev_cv(scorer, seed)``:
  1. refuses to write results if src/ has uncommitted changes, unless allow_dirty=True (recorded);
  2. writes a ``start`` record to results/trials.jsonl (one trial = one grid point = one call), and an ``end``
     record (status ok/error) when done, so aborted runs stay visible. Calls on the real dev universe are logged
     even with write=False (no unlogged probes);
  3. binds the scorer to the dev universe (label-free setup, timed), builds the split (default: §3.3 cluster-grouped
     5-fold CV with randomised balanced assignment, seed-dependent), and for each fold calls fit(train_idx) and
     score(test_idx);
  4. fold metrics (``metrics.metric_suite``): primary = HT-weighted AP (within-cluster pairs weight 1,
     between-cluster pairs weight K; K = 5; for the 'component' scheme the grouping is the connected component;
     for 'gene' and 'random' schemes all pairs have equal inclusion probability, weight 1). Secondary: uniform
     prevalence-reweighted AP, raw AP, AUROC, P@R, macro per-cluster AP, AP without the largest component,
     hard-negative AP, essentiality strata;
  5. label-free scorers: scores are computed once on all dev pairs (fold scores are slices; batch invariance is
     tested) and additionally summarised as pooled dev AP at natural prevalence and pooled HT AP (f = dev protein
     fraction) with an optional paired cluster bootstrap CI;
  6. returns the result dict (+ ``run_record``) and writes JSON to out_dir/<tag>.json.

Train pairs for fold k = pairs with both proteins outside fold k (includes pairs spanning two other folds);
test = both proteins in fold k; cross-fold pairs are unused for that fold.

``label_shuffle_control`` (held-fold label shuffle, DESIGN §3.9) is a helper for supervised scorers: mode 'train'
binds the scorer to a universe with ALL labels permuted (so neither training nor any reachable label is real) and
evaluates against the true held-fold labels; mode 'held' trains on true labels and evaluates against permuted
held-fold labels. Both should give AP ~ the (weighted) prevalence.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from src import metrics, splits
from src.config import CACHE, N_FOLDS, RESULTS, ROOT, SEEDS

TRIALS = RESULTS / "trials.jsonl"
PRIMARY = "ap_ht"
SUMMARY_KEYS = ("ap_ht", "ap_reweighted", "ap_raw", "auroc_ht", "macro_cluster_ap", "ap_ht_no_largest_component",
                "ap_ht_hard_neg", "ap_ht_both_nonessential", "ap_ht_mixed", "ap_ht_both_essential",
                "ht_P@R0.1", "ht_P@R0.25", "ht_P@R0.5")
CACHE_FILES = ("pairs.pkl", "CRISPRGeneEffect.f32.npy", "splits/protein_assignment.csv")


# ------------------------------------------------------------------------------------------- records
def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def git_state() -> dict:
    src = _git("status", "--porcelain", "--", "src")
    return dict(commit=_git("rev-parse", "HEAD"), dirty=bool(_git("status", "--porcelain")), dirty_src=bool(src),
                dirty_src_paths=src.splitlines()[:50])


_HASH_MEMO: dict = {}


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _HASH_MEMO:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        _HASH_MEMO[key] = h.hexdigest()
    return _HASH_MEMO[key]


def run_record(seed=None, counts=None, runtime_s=None, allow_dirty=False, **extra) -> dict:
    g = git_state()
    return dict(git_commit=g["commit"], git_dirty=g["dirty"], git_dirty_src=g["dirty_src"],
                git_dirty_src_paths=g.get("dirty_src_paths", []), allow_dirty=allow_dirty,
                cache_sha256={f: file_sha256(CACHE / f) for f in CACHE_FILES}, seed=seed, counts=counts,
                runtime_s=runtime_s, timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
                machine=dict(cpu_count=os.cpu_count(), platform=platform.platform(), python=platform.python_version()),
                **extra)


def check_clean(allow_dirty: bool) -> None:
    if git_state()["dirty_src"] and not allow_dirty:
        raise RuntimeError("src/ has uncommitted changes; commit first or pass allow_dirty=True (--allow-dirty)")


def log_trial(event: str, trial_id: str, path: Path | None = None, **fields) -> None:
    """Append one JSON line (event = 'start' | 'end') to results/trials.jsonl."""
    path = path or TRIALS
    rec = dict(event=event, trial_id=trial_id, time=_dt.datetime.now().isoformat(timespec="seconds"), **fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ------------------------------------------------------------------------------------------- splits
def make_dev_split(universe, scheme: str = "cluster", seed: int = SEEDS[0], n_folds: int = N_FOLDS):
    cm = universe.cluster_map() if scheme == "cluster" else universe.component_map() if scheme == "component" else None
    return splits.make_split(universe.pairs, scheme, n_folds, seed, clusters=cm)


def design_weights(universe, idx, scheme: str, n_folds: int = N_FOLDS) -> np.ndarray:
    """HT weights for a test fold of the given scheme."""
    if scheme == "cluster":
        return metrics.ht_weights(universe.col("cluster_a")[idx], universe.col("cluster_b")[idx], 1.0 / n_folds)
    if scheme in ("component", "complex"):
        return metrics.ht_weights(universe.col("comp_a")[idx], universe.col("comp_b")[idx], 1.0 / n_folds)
    return np.ones(len(idx))


def suite(universe, idx, s, w) -> dict:
    u = universe
    has_ess = "ess_a" in u.pairs
    return metrics.metric_suite(
        u.y[idx], s, w, u.col("cluster_a")[idx], u.col("cluster_b")[idx], u.col("comp_a")[idx], u.col("comp_b")[idx],
        u.col("ess_a")[idx] if has_ess else None, u.col("ess_b")[idx] if has_ess else None,
        largest_component=u.largest_component)


def _summ(folds):
    out = {}
    for k in SUMMARY_KEYS:
        v = np.array([f.get(k, np.nan) for f in folds], dtype=float)
        out[k] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                      folds=[float(x) for x in v])
    return out


# ------------------------------------------------------------------------------------------ main entry
def run_dev_cv(scorer, seed: int = SEEDS[0], scheme: str = "cluster", universe=None, n_folds: int = N_FOLDS,
               n_boot_pooled: int = 0, out_dir: Path | None = None, tag: str | None = None, allow_dirty: bool = False,
               write: bool = True, trials_path: Path | None = None, extra: dict | None = None) -> dict:
    if seed not in SEEDS:
        raise ValueError("seeds must come from src.config.SEEDS")
    if write:
        check_clean(allow_dirty)
    if universe is None:
        from src.universe import load_dev_universe
        universe = load_dev_universe()
    tag = tag or f"{scorer.name}_{scheme}_seed{seed}"
    trial_id = uuid.uuid4().hex[:12]
    desc = scorer.describe()
    # every evaluation on the REAL dev universe is logged, even when no JSON is written (no silent probes);
    # toy universes are logged only when writing or when a trials_path is given
    log = write or universe.name == "dev" or trials_path is not None
    if log:
        log_trial("start", trial_id, trials_path, tag=tag, scorer=desc, seed=seed, scheme=scheme,
                  universe=universe.name, git=git_state(), allow_dirty=allow_dirty, extra=extra)
    t_all = time.perf_counter()
    try:
        res = _run(scorer, seed, scheme, universe, n_folds, n_boot_pooled)
    except BaseException as e:
        if log:
            log_trial("end", trial_id, trials_path, tag=tag, status="error", error=repr(e)[:500])
        raise
    res["runtime_s"]["total_wall"] = round(time.perf_counter() - t_all, 3)
    res.update(tag=tag, trial_id=trial_id, scorer=desc, extra=extra or {})
    res["run_record"] = run_record(seed=seed, counts=res["counts"], runtime_s=res["runtime_s"], allow_dirty=allow_dirty,
                                   universe=universe.name, split_fingerprint=res["split"]["fingerprint"])
    path = None
    if log and not write:
        log_trial("end", trial_id, trials_path, tag=tag, status="ok_not_written", split_fingerprint=res["split"]["fingerprint"],
                  primary=dict(metric=PRIMARY, mean=res["summary"][PRIMARY]["mean"], sd=res["summary"][PRIMARY]["sd"]))
    if write:
        out_dir = Path(out_dir or RESULTS / "dev")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{tag}.json"
        path.write_text(json.dumps(res, indent=1, default=_json_default))
        log_trial("end", trial_id, trials_path, tag=tag, status="ok", result=str(path.relative_to(ROOT)) if
                  path.is_relative_to(ROOT) else str(path), split_fingerprint=res["split"]["fingerprint"],
                  primary=dict(metric=PRIMARY, mean=res["summary"][PRIMARY]["mean"], sd=res["summary"][PRIMARY]["sd"],
                               folds=res["summary"][PRIMARY]["folds"]),
                  pooled_ap_raw=res.get("pooled", {}).get("ap_raw"), runtime_s=res["runtime_s"]["total_wall"])
    res["path"] = str(path) if path else None
    return res


def _run(scorer, seed, scheme, u, n_folds, n_boot_pooled):
    rt = {}
    t0 = time.perf_counter()
    scorer.bind(u)
    rt["setup"] = round(time.perf_counter() - t0, 3)
    sp = make_dev_split(u, scheme, seed, n_folds)
    y = u.y
    all_idx = np.arange(len(u))
    s_all = None
    if scorer.label_free:
        t0 = time.perf_counter()
        s_all = scorer.score(all_idx)
        rt["score_all_dev"] = round(time.perf_counter() - t0, 3)
    folds, fold_rt = [], []
    for k in range(n_folds):
        tr, te = np.flatnonzero(sp.train_mask(k)), np.flatnonzero(sp.test_mask(k))
        t0 = time.perf_counter()
        if s_all is None:
            scorer.fit(tr)
            s = scorer.score(te)
        else:
            s = s_all[te]
        fold_rt.append(round(time.perf_counter() - t0, 3))
        if np.isnan(s).any():
            raise ValueError(f"{scorer.name}: NaN scores in fold {k}")
        m = suite(u, te, s, design_weights(u, te, scheme, n_folds))
        m.update(fold=k, n_train=int(len(tr)), n_train_pos=int(y[tr].sum()))
        folds.append(m)
    rt["fit_score_per_fold"] = fold_rt
    res = dict(scheme=scheme, seed=seed, n_folds=n_folds, folds=folds, summary=_summ(folds), runtime_s=rt,
               split=dict(fingerprint=sp.fingerprint(), stats=sp.stats(y)),
               counts=dict(n_pairs=int(len(u)), n_pos=int(y.sum()), n_neg=int(len(u) - y.sum()),
                           n_proteins=int(len(set(u.pairs.a) | set(u.pairs.b))),
                           n_clusters=int(len(set(u.col("cluster_a")) | set(u.col("cluster_b"))))))
    if s_all is not None:
        w = metrics.ht_weights(u.col("cluster_a"), u.col("cluster_b"), u.f)
        pooled = suite(u, all_idx, s_all, w)
        pooled["ht_f"] = u.f
        if n_boot_pooled:
            b = metrics.cluster_bootstrap(y, {"s": s_all}, u.col("cluster_a"), u.col("cluster_b"), f=u.f,
                                          n_boot=n_boot_pooled, seed=seed)
            pooled["ap_ht_ci"] = [b["ap"]["s"]["ci_low"], b["ap"]["s"]["ci_high"]]
            pooled["ap_ht_boot_sd"] = b["ap"]["s"]["boot_sd"]
        res["pooled"] = pooled
    return res


# --------------------------------------------------------------------------------- negative controls
def label_shuffle_control(scorer_factory, universe, seed: int = SEEDS[0], mode: str = "train",
                          scheme: str = "cluster", n_folds: int = N_FOLDS) -> dict:
    """Held-fold label-shuffle control for supervised scorers (not logged; returns per-fold AP vs prevalence).
    scorer_factory() must return a fresh, unbound scorer."""
    rng = np.random.default_rng(seed)
    y_true = universe.y
    sp = make_dev_split(universe, scheme, seed, n_folds)
    perm_u = universe.with_labels(rng.permutation(y_true)) if mode == "train" else universe
    out = []
    for k in range(n_folds):
        tr, te = np.flatnonzero(sp.train_mask(k)), np.flatnonzero(sp.test_mask(k))
        sc = scorer_factory().bind(perm_u)
        sc.fit(tr)
        s = sc.score(te)
        y_eval = y_true[te] if mode == "train" else rng.permutation(y_true[te])
        w = design_weights(universe, te, scheme, n_folds)
        out.append(dict(fold=k, ap_ht=metrics.weighted_ap(y_eval, s, w),
                        ht_prevalence=float((w * y_eval).sum() / w.sum())))
    return dict(mode=mode, folds=out, mean_ap_ht=float(np.mean([f["ap_ht"] for f in out])),
                mean_prevalence=float(np.mean([f["ht_prevalence"] for f in out])))
