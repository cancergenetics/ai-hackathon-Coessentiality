"""Phase C (DESIGN v3 §4, time-boxed): supervised metric learning on the dev cluster CV (src/metric_learning.py).

Capacity ladder (each rung = MetricScorer configuration; λ and the number of steps are chosen INSIDE each fold
by an inner cluster-grouped validation split of that fold's training pairs):
  identity   no learning (sanity: must reproduce spec_k20_t0.1_std = 0.365 and the Phase-A best 0.435)
  L1line     learned cell-line weights on the whitened profiles (weighted Pearson), shrunk to 1
  L1eig      learned spectral filter (diagonal in the eigenbasis), shrunk to 1
  L2d{32,64,128}  low-rank residual projection, BCE with hard negatives;  L2d*nce  same with InfoNCE
Outputs: cos (first order), rowzso (L3: genome-wide top-0.25% neighbour cosine on the learned embedding of ALL
genes + genome-wide row-z; label-free given the fitted map).
Every evaluation goes through src.evaluate.run_dev_cv (logged trial, JSON -> results/phaseC/<tag>.json); OOF scores
(data_cache/phaseC_oof/) feed a 1,000-resample paired cluster bootstrap on pooled OOF dev predictions vs the
current best label-free layer and GLS (results/phaseC/compare.json). IFT is never loaded.

Usage:
  ./.env/bin/python scripts/30_phaseC_metric.py --run identity:cos,identity:rowzso [--seeds 0] [--allow-dirty]
  ./.env/bin/python scripts/30_phaseC_metric.py --control L2d64:cos            (training-label shuffle, seed 0)
  ./.env/bin/python scripts/30_phaseC_metric.py --compare            (bootstrap over all available OOFs)
  ./.env/bin/python scripts/30_phaseC_metric.py --readme
"""
import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from src import evaluate, metrics, scorers  # noqa: E402
from src import metric_learning as ML  # noqa: E402
from src.config import CACHE, N_FOLDS, RESULTS, ROOT, SEEDS  # noqa: E402

OUT = RESULTS / "phaseC"
OOF_DIR = CACHE / "phaseC_oof"
BEST_NAME = "rowz[socos_q0.25[spec_k20_t0.1_std]]"
BEST_CACHE = "screen_v2"
N_BOOT = 1000

CONFIGS = {
    "identity": dict(level="identity"),
    "L1line": dict(level="L1", basis="line"),
    "L1eig": dict(level="L1", basis="eig"),
    "L2d32": dict(level="L2", d=32),
    "L2d64": dict(level="L2", d=64),
    "L2d128": dict(level="L2", d=128),
    "L2d32nce": dict(level="L2", d=32, loss="infonce"),
    "L2d64nce": dict(level="L2", d=64, loss="infonce"),
    "L2d128nce": dict(level="L2", d=128, loss="infonce"),
}
REFS = ("ref_best", "ref_gls")

_MAT = {}


def ref_scorer(name):
    if name == "ref_gls":
        if "m" not in _MAT:
            _MAT["m"] = scorers.real_matrix()
        sc = scorers.make_baseline("gls_signed", _MAT["m"])
        sc.name = name
        return sc
    man = json.loads((RESULTS / "layers" / f"{BEST_CACHE}.manifest.json").read_text())
    col = {c["name"]: c["col"] for c in man["columns"]}[BEST_NAME]
    M = np.load(CACHE / "layers" / f"{BEST_CACHE}.scores.f32.npy", mmap_mode="r")
    return scorers.PrecomputedScorer(name, np.asarray(M[:, col], dtype=np.float64),
                                     params=dict(layer=BEST_NAME, cache=BEST_CACHE, col=col))


def factory(cfg, output, seed):
    if cfg in REFS:
        return ref_scorer(cfg)
    return ML.MetricScorer(f"{cfg}_{output}", output=output, seed=seed, **CONFIGS[cfg])


class Recorder:
    def __init__(self, sc):
        self.sc, self.name, self.label_free, self.calls = sc, sc.name, sc.label_free, []

    def describe(self):
        return self.sc.describe()

    def bind(self, u):
        self.sc.bind(u)
        return self

    def fit(self, idx):
        self.sc.fit(idx)
        return self

    def score(self, idx):
        s = self.sc.score(idx)
        self.calls.append((np.asarray(idx).copy(), np.asarray(s).copy()))
        return s


def tag_of(cfg, output, seed):
    return f"{cfg}_cluster_seed{seed}" if cfg in REFS else f"{cfg}_{output}_cluster_seed{seed}"


def oof_from(rec, u, seed):
    sp = evaluate.make_dev_split(u, "cluster", seed)
    oof = np.full(len(u), np.nan)
    if rec.label_free:
        idx, s = rec.calls[0]
        full = np.full(len(u), np.nan)
        full[idx] = s
        te = sp.test_fold >= 0
        oof[te] = full[te]
    else:
        assert len(rec.calls) == N_FOLDS
        for k, (idx, s) in enumerate(rec.calls):
            assert np.array_equal(idx, np.flatnonzero(sp.test_mask(k)))
            oof[idx] = s
    return oof


def top_clusters(u, idx, n=10):
    """Clusters ranked by within-cluster positives among the given (pooled OOF) pairs."""
    y = u.y[idx].astype(bool)
    ca, cb = u.col("cluster_a")[idx], u.col("cluster_b")[idx]
    c, n_pos = np.unique(ca[y & (ca == cb)], return_counts=True)
    return [int(x) for x in c[np.argsort(-n_pos, kind="stable")][:n]]


def pooled_metrics(u, oof, seed):
    idx = np.flatnonzero(np.isfinite(oof))
    w = evaluate.design_weights(u, idx, "cluster")
    out = evaluate.suite(u, idx, oof[idx], w)
    top = top_clusters(u, idx)
    ca, cb = u.col("cluster_a")[idx], u.col("cluster_b")[idx]
    keep = ~(np.isin(ca, top) | np.isin(cb, top))
    out["ap_ht_no_top10"] = metrics.weighted_ap(u.y[idx][keep], oof[idx][keep], w[keep])
    out["n_pos_no_top10"] = int(u.y[idx][keep].sum())
    out["top10_clusters"] = top
    return out


def run_one(cfg, output, seed, u, allow_dirty):
    tag = tag_of(cfg, output, seed)
    sc = factory(cfg, output, seed)
    rec = Recorder(sc)
    t0 = time.perf_counter()
    res = evaluate.run_dev_cv(rec, seed=seed, scheme="cluster", universe=u, out_dir=OUT, tag=tag,
                              allow_dirty=allow_dirty, extra=dict(grid="phaseC_v1", config=cfg, output=output))
    oof = oof_from(rec, u, seed)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OOF_DIR / f"{tag}.npy", oof)
    extra = dict(pooled_oof=pooled_metrics(u, oof, seed), wall_s=round(time.perf_counter() - t0, 1))
    if isinstance(sc, ML.MetricScorer):
        extra["fit_summary"] = sc.fit_summary()
        extra["fit_log"] = sc.fit_log
    path = Path(res["path"])
    full = json.loads(path.read_text())
    full["phaseC"] = extra
    path.write_text(json.dumps(full, indent=1, default=evaluate._json_default))
    fs = extra.get("fit_summary", [])
    print(f"{tag:34s} HT-AP {res['summary']['ap_ht']['mean']:.3f} ± {res['summary']['ap_ht']['sd']:.3f}  pooled-OOF "
          f"{extra['pooled_oof']['ap_ht']:.3f}  no-top10 {extra['pooled_oof']['ap_ht_no_top10']:.3f}  "
          f"{extra['wall_s']}s  steps {[f.get('chosen_steps') for f in fs]} lam {[f.get('chosen_lam') for f in fs]}",
          flush=True)
    return res


def run_control(cfg, output, u, allow_dirty, seed=SEEDS[0]):
    """Training-label shuffle: all labels permuted before fitting, evaluated on the true held-fold labels (mode
    'train' of src.evaluate.label_shuffle_control). The learned metric should not beat its identity initialisation."""
    evaluate.check_clean(allow_dirty)
    tag = f"control_shuffle_train_{cfg}_{output}_cluster_seed{seed}"
    tid = uuid.uuid4().hex[:12]
    evaluate.log_trial("start", tid, tag=tag, scorer=factory(cfg, output, seed).describe(), seed=seed,
                       scheme="cluster", universe=u.name, git=evaluate.git_state(), allow_dirty=allow_dirty,
                       extra=dict(grid="phaseC_v1", control="label_shuffle_train"))
    t0 = time.perf_counter()
    made = []

    def fac():
        made.append(factory(cfg, output, seed))
        return made[-1]
    try:
        c = evaluate.label_shuffle_control(fac, u, seed=seed, mode="train", scheme="cluster")
    except BaseException as e:
        evaluate.log_trial("end", tid, tag=tag, status="error", error=repr(e)[:500])
        raise
    c.update(tag=tag, trial_id=tid, config=cfg, output=output, runtime_s=round(time.perf_counter() - t0, 1),
             fit_summary=[f for m in made for f in m.fit_summary()],
             run_record=evaluate.run_record(seed=seed, allow_dirty=allow_dirty, universe=u.name))
    path = OUT / f"{tag}.json"
    path.write_text(json.dumps(c, indent=1, default=evaluate._json_default))
    evaluate.log_trial("end", tid, tag=tag, status="ok", result=str(path.relative_to(ROOT)),
                       primary=dict(metric="ap_ht", mean=c["mean_ap_ht"], prevalence=c["mean_prevalence"]))
    print(f"{tag:45s} AP {c['mean_ap_ht']:.4f} vs HT prevalence {c['mean_prevalence']:.4f}; steps "
          f"{[f.get('chosen_steps') for f in c['fit_summary']]}", flush=True)


def compare(u):
    """Paired cluster bootstrap (pooled OOF, per seed) of every available phase-C OOF vs ref_best and ref_gls, on all
    OOF pairs and on pairs not touching the top-10 clusters."""
    out = dict(note="pooled OOF predictions of the 5 cluster-CV test folds; HT weights 1 within / K between clusters; "
                    "1,000-resample paired cluster bootstrap over dev clusters, bootstrap seed = split seed",
               seeds={}, run_record=evaluate.run_record(universe=u.name))
    for seed in SEEDS:
        files = {p.stem.replace(f"_cluster_seed{seed}", ""): p for p in OOF_DIR.glob(f"*_cluster_seed{seed}.npy")}
        if not all(r in files for r in REFS) or len(files) <= len(REFS):
            continue
        S = {k: np.load(p) for k, p in sorted(files.items())}
        idx = np.flatnonzero(np.isfinite(S["ref_gls"]))
        for k, v in S.items():
            assert np.array_equal(np.flatnonzero(np.isfinite(v)), idx), k
        ca, cb = u.col("cluster_a")[idx], u.col("cluster_b")[idx]
        w = metrics.ht_weights(ca, cb, 1.0 / N_FOLDS)
        cands = [k for k in S if k not in REFS]
        diffs = [(c, r) for c in cands for r in REFS] + [("ref_best", "ref_gls")]
        if "identity_cos" in S:
            diffs += [(c, "identity_cos") for c in cands if c.endswith("_cos") and c != "identity_cos"]
        res = {}
        top = top_clusters(u, idx)
        keep = ~(np.isin(ca, top) | np.isin(cb, top))
        for sub, m in (("all", np.ones(len(idx), bool)), ("no_top10", keep)):
            b = metrics.cluster_bootstrap(u.y[idx][m], {k: v[idx][m] for k, v in S.items()}, ca[m], cb[m],
                                          n_boot=N_BOOT, seed=seed, diffs=diffs, base_w=w[m])
            b.update(n_pairs=int(m.sum()), n_pos=int(u.y[idx][m].sum()))
            res[sub] = b
        res["top10_clusters"] = top
        # macro per-cluster AP on pooled OOF
        res["macro_cluster_ap"] = {k: metrics.macro_cluster_ap(u.y[idx], v[idx], ca, cb, w)["macro_cluster_ap"]
                                   for k, v in S.items()}
        out["seeds"][str(seed)] = res
        print(f"seed {seed}: " + ", ".join(f"{d}: {v['diff']:+.3f} [{v['ci_low']:+.3f},{v['ci_high']:+.3f}]"
                                           for d, v in res["all"]["diff"].items() if d.endswith("ref_best")), flush=True)
    (OUT / "compare.json").write_text(json.dumps(out, indent=1, default=evaluate._json_default))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--run", default="")
    ap.add_argument("--control", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--readme", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    u = None
    if a.run or a.control or a.compare:
        subprocess.run(["sh", str(ROOT / "scripts" / "check_data.sh")], check=True, stdout=subprocess.DEVNULL)
        from src.universe import load_dev_universe
        u = load_dev_universe()
    seeds = [SEEDS[int(s)] for s in a.seeds.split(",")]
    for seed in seeds if a.run else []:
        for item in a.run.split(","):
            cfg, _, output = item.partition(":")
            if cfg not in REFS and cfg not in CONFIGS:
                raise KeyError(cfg)
            if (OUT / f"{tag_of(cfg, output, seed)}.json").exists() and cfg in REFS:
                continue
            run_one(cfg, output or "cos", seed, u, a.allow_dirty)
    for item in filter(None, a.control.split(",")):
        cfg, _, output = item.partition(":")
        run_control(cfg, output or "cos", u, a.allow_dirty)
    if a.compare:
        compare(u)
    if a.readme:
        import importlib.util
        spec = importlib.util.spec_from_file_location("readme30", ROOT / "scripts" / "30_phaseC_readme.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.write_readme()


if __name__ == "__main__":
    main()
