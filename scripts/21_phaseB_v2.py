"""Phase B v2 (DESIGN v3 §4 Phase B/D; RESEARCH_TREE D3): final combination round on the FROZEN layer set, 5 seeds.

Frozen candidate layers (chosen in Phase A / A7; no new layers here), z-scored with the genome-wide null moments of
their cache manifests:
  B  = rowz[socos_q0.25[spec_k20_t0.1_std]]                      (C_single; screen_v2)
  E  = emb_lap512[knn_union_lin_q0.25[spec_k20_t0.1_std]]          (graph_v1)
  H  = rowz[cos_heat1[knn_union_lin_q0.25[spec_k20_t0.1_std]]]     (graph_v1)
  F  = spec_k20_t0.1_std   (first order; screen_v1)     F0 = spec_k0_t0.1_std (screen_v1)
  SO = socos_q0.25[spec_k20_t0.1_std]  (raw second order; screen_v2)
  G  = GLS signed t (baseline scorer, src.scorers)
Pre-specified combinations (no others):
  1. label-free z-averages (equal weights): B+E, B+F, B+E+F, B+E+H+F; plus nested-by-fold selection among the 4.
  2. L2 logistic regression (standardised) on {B,E,H,F,F0,G,SO} fitted in each training fold, +/- binned gene feats.
  3. LightGBM (num_leaves 7, min_child_samples 2000, monotone +1 in every similarity feature, bagged negatives,
     cluster-balanced weights, inner cluster-grouped early stopping; src.supervised defaults) on the same features
     +/- binned gene features.
  4. (optional) rank-based stacker. Rule fixed before any 2/3 result was looked at: run it only if some
     supervised entry (2/3) has a higher 5-seed mean HT-AP than the best z-average AND a pooled-OOF Δ vs that
     z-average > 0 in >= 3/5 seeds; otherwise not run (supervision shows no promise over equal weights).
References (same OOF pairs): every single layer above, GLS / Spearman / Pearson (best dev variants = signed).

Stages (every evaluation is a logged trial through src.evaluate.run_dev_cv / log_trial; IFT never loaded):
  run       all entries x SEEDS (JSON -> results/phaseB_v2/runs/<tag>.json, OOF -> data_cache/phaseB_v2_oof/)
  control   held-fold label shuffle (src.evaluate.label_shuffle_control mode 'held') for each supervised entry,
            SEEDS[0]
  nested    per seed: nested-by-fold selection among the 4 z-averages, and among all combination entries
  compare   per seed: paired cluster bootstrap (1,000) of pooled OOF HT-AP, all pairs and pairs not touching the
            top-10 clusters; pooled-OOF macro per-cluster AP
  runtime   fresh-process wall clock to compute each combination's label-free inputs for all 596,597 scorable pairs
  readme    tables + D3 decision (results/phaseB_v2/README.md, decision.json)

Usage: ./.env/bin/python scripts/21_phaseB_v2.py --stages run,control,nested,compare,runtime,readme [--allow-dirty]
"""
import argparse
import json
import resource
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from src import evaluate, metrics, scorers, supervised as S  # noqa: E402
from src.config import CACHE, N_FOLDS, RESULTS, ROOT, SEEDS  # noqa: E402

OUT = RESULTS / "phaseB_v2"
RUNS = OUT / "runs"
OOF_DIR = CACHE / "phaseB_v2_oof"
N_BOOT = 1000

LAYERS = {
    "B": ("screen_v2", "rowz[socos_q0.25[spec_k20_t0.1_std]]"),
    "E": ("graph_v1", "emb_lap512[knn_union_lin_q0.25[spec_k20_t0.1_std]]"),
    "H": ("graph_v1", "rowz[cos_heat1[knn_union_lin_q0.25[spec_k20_t0.1_std]]]"),
    "F": ("screen_v1", "spec_k20_t0.1_std"),
    "F0": ("screen_v1", "spec_k0_t0.1_std"),
    "SO": ("screen_v2", "socos_q0.25[spec_k20_t0.1_std]"),
}
BASES = {"G": "gls_signed", "SP": "spearman_signed", "PE": "pearson_signed"}
ZAVGS = {"zavg_BE": ["B", "E"], "zavg_BF": ["B", "F"], "zavg_BEF": ["B", "E", "F"], "zavg_BEHF": ["B", "E", "H", "F"]}
FEAT7 = ["B", "E", "H", "F", "F0", "G", "SO"]
SUP = ["lr_L7", "lr_L7+genebin", "lgbm_L7", "lgbm_L7+genebin"]
REFS = [f"ref_{k}" for k in LAYERS] + [f"ref_{k}" for k in BASES]
C_SINGLE = "ref_B"
BASE_REFS = ["ref_G", "ref_SP", "ref_PE"]
EXTRA = []                         # optional stacker entries are appended here if run (see module docstring)


def combos():
    return list(ZAVGS) + SUP + EXTRA


def entries():
    return REFS + combos()


# ------------------------------------------------------------------------------------------ scores / scorers
_MAT, _Z = {}, {}


def mat():
    if "m" not in _MAT:
        _MAT["m"] = scorers.real_matrix()
    return _MAT["m"]


def zlayer(key):
    """Manifest-null-z-scored layer, indexed by scorable pair_id."""
    if key not in _Z:
        tag, name = LAYERS[key]
        M, _, cols = S.load_layer_cache(tag)
        c = cols[name]
        _Z[key] = (np.asarray(M[:, c["col"]], dtype=np.float64) - c["null_mean"]) / c["null_sd"]
    return _Z[key]


def feature_blocks(spec):
    by_tag = {}
    for k in FEAT7:
        if k in LAYERS:
            by_tag.setdefault(LAYERS[k][0], []).append(LAYERS[k][1])
    out = [S.LayerFeatures(names, tag) for tag, names in by_tag.items()]
    out.append(S.BaselineFeatures(["gls_signed"], mat()))
    if spec == "L7+genebin":
        out.append(S.GeneFeatures("binned"))
    elif spec != "L7":
        raise ValueError(spec)
    return out


def factory(name, seed):
    if name.startswith("ref_"):
        k = name[4:]
        if k in BASES:
            sc = scorers.make_baseline(BASES[k], mat())
            sc.name = name
            return sc
        return scorers.PrecomputedScorer(name, zlayer(k), params=dict(layer=LAYERS[k][1], cache=LAYERS[k][0],
                                                                      z="manifest genome-wide null moments"))
    if name in ZAVGS:
        s = np.mean([zlayer(k) for k in ZAVGS[name]], axis=0)
        return scorers.PrecomputedScorer(name, s, params=dict(members={k: LAYERS[k][1] for k in ZAVGS[name]},
                                                              combine="equal-weight mean of manifest-null z"))
    model, spec = name.split("_", 1)
    return S.make_supervised(name, feature_blocks(spec), model={"lr": "logreg", "lgbm": "lgbm"}[model], seed=seed)


class Recorder:
    """Transparent wrapper that keeps the scores handed to the evaluator (pooled OOF)."""

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


def safe(name):
    return name.replace("+", "-")


def tag_of(name, seed):
    return f"{safe(name)}_cluster_seed{seed}"


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
    y = u.y[idx].astype(bool)
    ca, cb = u.col("cluster_a")[idx], u.col("cluster_b")[idx]
    c, n_pos = np.unique(ca[y & (ca == cb)], return_counts=True)
    return [int(x) for x in c[np.argsort(-n_pos, kind="stable")][:n]]


def pooled_metrics(u, oof):
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


# ------------------------------------------------------------------------------------------ stage: run
def run_one(name, seed, u, allow_dirty):
    tag = tag_of(name, seed)
    if (RUNS / f"{tag}.json").exists() and (OOF_DIR / f"{tag}.npy").exists():
        return
    rec = Recorder(factory(name, seed))
    t0 = time.perf_counter()
    res = evaluate.run_dev_cv(rec, seed=seed, scheme="cluster", universe=u, out_dir=RUNS, tag=tag,
                              allow_dirty=allow_dirty, extra=dict(grid="phaseB_v2", entry=name))
    oof = oof_from(rec, u, seed)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OOF_DIR / f"{tag}.npy", oof)
    extra = dict(pooled_oof=pooled_metrics(u, oof), wall_s=round(time.perf_counter() - t0, 1))
    sc = rec.sc
    if isinstance(sc, S.SupervisedPairScorer):
        extra.update(feature_importance=sc.feature_importance(), fit_log=sc.fit_log,
                     bind_seconds=round(sc.bind_seconds, 2))
    path = Path(res["path"])
    full = json.loads(path.read_text())
    full["phaseB_v2"] = extra
    path.write_text(json.dumps(full, indent=1, default=evaluate._json_default))
    print(f"{tag:42s} HT-AP {res['summary']['ap_ht']['mean']:.3f} ± {res['summary']['ap_ht']['sd']:.3f}  pooled-OOF "
          f"{extra['pooled_oof']['ap_ht']:.3f}  no-top10 {extra['pooled_oof']['ap_ht_no_top10']:.3f}  "
          f"{extra['wall_s']}s", flush=True)


def stage_run(u, allow_dirty, names=None, seeds=SEEDS):
    for seed in seeds:
        for name in names or entries():
            run_one(name, seed, u, allow_dirty)


# ------------------------------------------------------------------------------------------ stage: control
def stage_control(u, allow_dirty, names=None, seed=SEEDS[0]):
    evaluate.check_clean(allow_dirty)
    for name in names or [n for n in combos() if n not in ZAVGS]:
        tag = f"control_shuffle_held_{safe(name)}_cluster_seed{seed}"
        if (OUT / "controls" / f"{tag}.json").exists():
            continue
        tid = uuid.uuid4().hex[:12]
        evaluate.log_trial("start", tid, tag=tag, scorer=factory(name, seed).describe(), seed=seed, scheme="cluster",
                           universe=u.name, git=evaluate.git_state(), allow_dirty=allow_dirty,
                           extra=dict(grid="phaseB_v2", control="label_shuffle_held"))
        t0 = time.perf_counter()
        try:
            c = evaluate.label_shuffle_control(lambda: factory(name, seed), u, seed=seed, mode="held", scheme="cluster")
        except BaseException as e:
            evaluate.log_trial("end", tid, tag=tag, status="error", error=repr(e)[:500])
            raise
        c.update(tag=tag, trial_id=tid, model=name, runtime_s=round(time.perf_counter() - t0, 1),
                 run_record=evaluate.run_record(seed=seed, allow_dirty=allow_dirty, universe=u.name))
        (OUT / "controls").mkdir(parents=True, exist_ok=True)
        path = OUT / "controls" / f"{tag}.json"
        path.write_text(json.dumps(c, indent=1, default=evaluate._json_default))
        evaluate.log_trial("end", tid, tag=tag, status="ok", result=str(path.relative_to(ROOT)),
                           primary=dict(metric="ap_ht", mean=c["mean_ap_ht"], prevalence=c["mean_prevalence"]))
        print(f"{tag:50s} AP {c['mean_ap_ht']:.4f} vs HT prevalence {c['mean_prevalence']:.4f}", flush=True)


# ------------------------------------------------------------------------------------------ stage: nested
def load_run(name, seed):
    p = RUNS / f"{tag_of(name, seed)}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _nested(cands, seed, label):
    F = np.array([load_run(c, seed)["summary"]["ap_ht"]["folds"] for c in cands])
    fp = load_run(cands[0], seed)["split"]["fingerprint"]
    recs = []
    for k in range(N_FOLDS):
        inner = [i for i in range(N_FOLDS) if i != k]
        j = int(np.argmax(F[:, inner].mean(1)))
        tg = f"phaseB_v2_nested_{label}_outer{k}_seed{seed}"
        tid = uuid.uuid4().hex[:12]
        evaluate.log_trial("start", tid, tag=tg, universe="dev", seed=seed, scheme="cluster", git=evaluate.git_state(),
                           scorer=dict(name=f"argmax over {len(cands)} ({label}, nested)", label_free=None,
                                       candidates=cands), extra=dict(grid="phaseB_v2_nested", outer_fold=k))
        rec = dict(outer_fold=k, selected=cands[j], inner_ap_ht=float(F[j, inner].mean()), held_out_ap_ht=float(F[j, k]))
        evaluate.log_trial("end", tid, tag=tg, status="ok_not_written", split_fingerprint=fp,
                           primary=dict(metric="ap_ht", held_out_fold=k, value=rec["held_out_ap_ht"]), result=rec)
        recs.append(rec)
    best = int(np.argmax(F.mean(1)))
    return dict(label=label, candidates=cands, outer=recs,
                nested_mean_held_out=float(np.mean([r["held_out_ap_ht"] for r in recs])),
                selected_in_sample=dict(name=cands[best], ap_ht=float(F[best].mean())), split_fingerprint=fp)


def stage_nested():
    out = {}
    for seed in SEEDS:
        out[str(seed)] = dict(zavg=_nested(list(ZAVGS), seed, "zavg4"), all_combos=_nested(combos(), seed, "all_combos"))
        print(seed, {k: (v["nested_mean_held_out"], v["selected_in_sample"]) for k, v in out[str(seed)].items()},
              flush=True)
    (OUT / "nested.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ stage: compare
def stage_compare(u):
    names = entries()
    diffs = ([(c, C_SINGLE) for c in combos()] + [(c, b) for c in combos() + [C_SINGLE] for b in BASE_REFS]
             + [(s, z) for s in combos() if s not in ZAVGS for z in ZAVGS] + [("ref_E", C_SINGLE), ("ref_H", C_SINGLE)])
    tid = uuid.uuid4().hex[:12]
    evaluate.log_trial("start", tid, tag="phaseB_v2_compare_bootstrap", universe="dev", seed=list(SEEDS),
                       git=evaluate.git_state(), scorer=dict(name="Phase B v2 pooled-OOF paired cluster bootstrap",
                                                            label_free=None), extra=dict(grid="phaseB_v2_compare"))
    out = dict(note="pooled OOF predictions of the 5 cluster-CV test folds; HT weights 1 within / K between clusters; "
                    "1,000-resample paired cluster bootstrap over dev clusters, bootstrap seed = split seed",
               seeds={})
    for seed in SEEDS:
        Sx = {n: np.load(OOF_DIR / f"{tag_of(n, seed)}.npy") for n in names}
        idx = np.flatnonzero(np.isfinite(Sx[C_SINGLE]))
        for n, v in Sx.items():
            assert np.array_equal(np.flatnonzero(np.isfinite(v)), idx), n
        y = u.y[idx]
        ca, cb = u.col("cluster_a")[idx], u.col("cluster_b")[idx]
        w = metrics.ht_weights(ca, cb, 1.0 / N_FOLDS)
        top = top_clusters(u, idx)
        keep = ~(np.isin(ca, top) | np.isin(cb, top))
        res = dict(top10_clusters=top)
        for sub, m in (("all", np.ones(len(idx), bool)), ("no_top10", keep)):
            d = diffs if sub == "all" else [(c, C_SINGLE) for c in combos()] + [(C_SINGLE, b) for b in BASE_REFS]
            b = metrics.cluster_bootstrap(y[m], {k: v[idx][m] for k, v in Sx.items()}, ca[m], cb[m], n_boot=N_BOOT,
                                          seed=seed, diffs=d, base_w=w[m])
            b.update(n_pairs=int(m.sum()), n_pos=int(y[m].sum()))
            res[sub] = b
        res["macro_cluster_ap_pooled"] = {k: metrics.macro_cluster_ap(y, v[idx], ca, cb, w)["macro_cluster_ap"]
                                          for k, v in Sx.items()}
        out["seeds"][str(seed)] = res
        print(f"seed {seed}: " + ", ".join(f"{k}: {v['diff']:+.3f} [{v['ci_low']:+.3f},{v['ci_high']:+.3f}]"
                                           for k, v in res["all"]["diff"].items() if k.endswith(f"-{C_SINGLE}")),
              flush=True)
    out["run_record"] = evaluate.run_record(seed=list(SEEDS), universe=u.name)
    (OUT / "compare.json").write_text(json.dumps(out, indent=1, default=evaluate._json_default))
    evaluate.log_trial("end", tid, tag="phaseB_v2_compare_bootstrap", status="ok", result="results/phaseB_v2/compare.json")


# ------------------------------------------------------------------------------------------ stage: runtime
RT_SETS = {"ref_B": ["B"], "zavg_BE": ZAVGS["zavg_BE"], "zavg_BF": ZAVGS["zavg_BF"], "zavg_BEF": ZAVGS["zavg_BEF"],
           "zavg_BEHF": ZAVGS["zavg_BEHF"], "feat7": FEAT7}


def runtime_child(which):
    from src import baselines, data, layers, layers_ext
    from src.layers_graph import GraphEngine, as_config
    P = np.load(CACHE / "layers" / "screen_v1.pairs.npz")
    ia, ib = P["ia"], P["ib"]
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    X, _, _ = data.load_matrix("CRISPRGeneEffect")
    out = dict(which=which, n_pairs=int(len(ia)), load_matrix_s=round(time.perf_counter() - t0, 2), members=[])
    t0 = time.perf_counter()
    ge = GraphEngine(layers.LayerEngine(X))
    out["engine_init_s"] = round(time.perf_counter() - t0, 2)
    Z, maxdiff = [], 0.0
    for k in RT_SETS[which]:
        t0 = time.perf_counter()
        if k == "G":
            Sigma, _ = baselines.sample_covariance(X)
            s = baselines.gls_pairs(X, ia, ib, Sigma=Sigma)["signed"]
        else:
            tag, name = LAYERS[k]
            M, _, cols = S.load_layer_cache(tag)
            c = cols[name]
            cfg = as_config(c["config"]) if c["config"]["kind"] == "graph" else layers_ext.as_config(c["config"])
            s = ge.compute(cfg, ia, ib)
            maxdiff = max(maxdiff, float(np.max(np.abs(s - np.asarray(M[:, c["col"]], dtype=np.float64)))))
            Z.append((s - c["null_mean"]) / c["null_sd"])
        out["members"].append(dict(key=k, s=round(time.perf_counter() - t0, 2)))
    t0 = time.perf_counter()
    if which.startswith("zavg"):
        z = np.mean(Z, axis=0)
        out["combine_s"] = round(time.perf_counter() - t0, 3)
        out["finite"] = bool(np.isfinite(z).all())
    out["max_abs_diff_vs_cache"] = maxdiff
    out["total_s"] = round(time.perf_counter() - t_all, 2)
    out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    print("RUNTIME_JSON " + json.dumps(out))


def stage_runtime():
    path = OUT / "runtime.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for which in RT_SETS:
        if which in out:
            continue
        p = subprocess.run([sys.executable, __file__, "--runtime-child", which], capture_output=True, text=True,
                           cwd=ROOT)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("RUNTIME_JSON ")]
        if p.returncode or not line:
            raise RuntimeError(p.stderr[-2000:])
        out[which] = json.loads(line[-1][len("RUNTIME_JSON "):])
        print(out[which], flush=True)
        path.write_text(json.dumps(out, indent=1))
    t0 = time.perf_counter()
    from src import gene_features as gf
    gf.real_gene_tables()
    out["gene_features_s"] = round(time.perf_counter() - t0, 2)
    path.write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="run,control,nested,compare,runtime,readme")
    ap.add_argument("--names", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--runtime-child", default="")
    a = ap.parse_args()
    if a.runtime_child:
        runtime_child(a.runtime_child)
        return
    st = a.stages.split(",")
    names = a.names.split(",") if a.names else None
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else SEEDS
    u = None
    if {"run", "control", "compare"} & set(st):
        from src.universe import load_dev_universe
        u = load_dev_universe()
    OUT.mkdir(parents=True, exist_ok=True)
    if "run" in st:
        stage_run(u, a.allow_dirty, names, seeds)
    if "control" in st:
        stage_control(u, a.allow_dirty, names)
    if "nested" in st:
        stage_nested()
    if "compare" in st:
        stage_compare(u)
    if "runtime" in st:
        stage_runtime()
    if "readme" in st:
        write_readme()


def f3(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3f}"


def ci(d):
    return f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def seed_mean(name, key, pooled=False):
    v = []
    for s in SEEDS:
        r = load_run(name, s)
        v.append(r["phaseB_v2"]["pooled_oof"][key] if pooled else r["summary"][key]["mean"])
    return float(np.mean(v))


def d3_decision(cmp):
    """D3 (pre-registered): C_comb = best combination by 5-seed mean of held-out HT-AP; primary = C_comb iff its
    pooled-OOF Δ vs C_single has a CI excluding 0 (above 0) in >= 3/5 seeds AND its macro per-cluster AP is not
    lower than C_single's; else primary = C_single and C_comb is secondary. Fixed sequence: primary first."""
    score = {c: seed_mean(c, "ap_ht") for c in combos()}
    c_comb = max(score, key=score.get)
    per_seed = [cmp["seeds"][str(s)]["all"]["diff"][f"{c_comb}-{C_SINGLE}"] for s in SEEDS]
    n_excl = int(sum(d["ci_low"] > 0 for d in per_seed))
    n_excl_neg = int(sum(d["ci_high"] < 0 for d in per_seed))
    macro_c, macro_s = seed_mean(c_comb, "macro_cluster_ap"), seed_mean(C_SINGLE, "macro_cluster_ap")
    macro_pc = float(np.mean([cmp["seeds"][str(s)]["macro_cluster_ap_pooled"][c_comb] for s in SEEDS]))
    macro_ps = float(np.mean([cmp["seeds"][str(s)]["macro_cluster_ap_pooled"][C_SINGLE] for s in SEEDS]))
    crit1, crit2 = n_excl >= 3, macro_c >= macro_s
    primary = c_comb if (crit1 and crit2) else C_SINGLE
    secondary = C_SINGLE if primary == c_comb else c_comb
    return dict(ranking={k: score[k] for k in sorted(score, key=lambda k: -score[k])}, c_comb=c_comb,
                n_seeds_ci_above_0=n_excl, n_seeds_ci_below_0=n_excl_neg,
                deltas=[dict(seed=s, **d) for s, d in zip(SEEDS, per_seed)],
                macro_fold_mean=dict(c_comb=macro_c, c_single=macro_s), macro_pooled=dict(c_comb=macro_pc,
                                                                                         c_single=macro_ps),
                criterion_ci=crit1, criterion_macro=crit2, primary=primary, secondary=secondary,
                fixed_sequence=[primary, secondary])


def write_readme():
    cmp = json.loads((OUT / "compare.json").read_text())
    nst = json.loads((OUT / "nested.json").read_text())
    rt = json.loads((OUT / "runtime.json").read_text()) if (OUT / "runtime.json").exists() else {}
    prior_rt = json.loads((RESULTS / "phaseA_graph" / "runtime.json").read_text())
    dec = d3_decision(cmp)
    (OUT / "decision.json").write_text(json.dumps(dec, indent=1))
    label = {"ref_B": "B = rowz[socos_q0.25] (C_single)", "ref_E": "E = emb_lap512", "ref_H": "H = rowz[cos_heat1]",
             "ref_F": "F = spec_k20_t0.1_std", "ref_F0": "F0 = spec_k0_t0.1_std", "ref_SO": "SO = socos_q0.25 (raw)",
             "ref_G": "GLS signed", "ref_SP": "Spearman signed", "ref_PE": "Pearson signed"}

    def lab(n):
        return f"`{n}`" if n not in label else label[n]

    # runtimes
    def runtime_cols(n):
        r0 = [load_run(n, s) for s in SEEDS]
        if n.startswith("ref_") or n in ZAVGS:
            key = n if n in rt else None
            if key:
                return f"{rt[key]['total_s']:.0f}", "0"
            single = {"ref_E": prior_rt["emb_lap512[knn_union_lin_q0.25[spec_k20_t0.1_std]]"]["total_s"],
                      "ref_H": prior_rt["rowz[cos_heat1[knn_union_lin_q0.25[spec_k20_t0.1_std]]]"]["total_s"],
                      "ref_G": prior_rt["reference"]["gls_signed"]["total_s"],
                      "ref_PE": prior_rt["reference"]["pearson_signed"]["total_s"]}.get(n)
            return (f"{single:.0f}†" if single else "—"), "0"
        inp = rt.get("feat7", {}).get("total_s")
        if inp is not None and "genebin" in n:
            inp += rt.get("gene_features_s", 0.0)
        per_fold = np.mean([np.mean(r["runtime_s"]["fit_score_per_fold"]) for r in r0])
        return (f"{inp:.0f}" if inp is not None else "—"), f"{per_fold:.1f}"

    L = ["# Phase B v2 — final combination round on the frozen layer set (dev cluster CV, 5 seeds)", "",
         "Generated by `scripts/21_phaseB_v2.py --stages readme` (do not edit by hand). DEV pairs only via "
         "`src.evaluate.run_dev_cv` (every entry × seed is a logged trial in `results/trials.jsonl`); IFT never loaded. "
         "Frozen inputs (no new layers): B = rowz[socos_q0.25[spec_k20_t0.1_std]] (C_single), "
         "E = emb_lap512[knn_union_lin_q0.25[spec_k20_t0.1_std]], H = rowz[cos_heat1[knn_union_lin_q0.25[spec_k20_t0.1_std]]], "
         "F = spec_k20_t0.1_std, F0 = spec_k0_t0.1_std, SO = socos_q0.25[spec_k20_t0.1_std], G = GLS signed. Layers are "
         "z-scored with the genome-wide null moments of their cache manifests.", "",
         "Pre-specified combinations: z-averages (equal weights) `zavg_BE`, `zavg_BF`, `zavg_BEF`, `zavg_BEHF`; "
         "`lr_L7` / `lgbm_L7` = L2 logistic regression (StandardScaler, C = 1) / shallow monotone LightGBM "
         "(num_leaves 7, min_child_samples 2,000, monotone +1 in all 7 similarity features, 5 bags of all positives + "
         "50× negatives, cluster-balanced 1/√n weights, inner cluster-grouped early stopping; `src.supervised` "
         "defaults, unchanged from Phase B v1) on {B, E, H, F, F0, G, SO}, fitted inside each training fold; "
         "`+genebin` adds the ≤5-bin genome-wide-quantile gene features (min/max of 7 per-gene stats).", "",
         "Columns: HT-AP = fold mean (5-seed mean of per-seed fold means; held-out by construction — z-averages have no "
         "fitted parameters, supervised models are fitted on training folds only); pooled = HT-AP of pooled OOF "
         "predictions; macro = macro per-cluster AP (fold mean; `macro pooled` = on pooled OOF); no-top10 = pooled-OOF "
         "HT-AP on pairs not touching the 10 clusters with most OOF positives; strata = fold-mean HT-AP for both "
         "non-essential / mixed / both essential; AUROC = HT AUROC fold mean. Runtime: `inputs` = fresh-process wall "
         "clock to compute the entry's label-free inputs for all 596,597 scorable pairs (8-core laptop, shared with "
         "another agent's jobs; † = from results/phaseA_graph/runtime.json); `fit+score` = mean per-fold training + "
         "scoring time inside the CV (supervised only).", ""]

    L += ["## Summary (5-seed means)", "",
          "| group | entry | HT-AP | pooled | macro | macro pooled | no-top10 | non-ess / mixed / ess | AUROC | inputs (s) | fit+score per fold (s) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in entries():
        grp = "ref" if n.startswith("ref_") else "z-avg" if n in ZAVGS else "supervised"
        mp = np.mean([cmp["seeds"][str(s)]["macro_cluster_ap_pooled"][n] for s in SEEDS])
        inp, fs = runtime_cols(n)
        L.append(f"| {grp} | {lab(n)} | {f3(seed_mean(n, 'ap_ht'))} | {f3(seed_mean(n, 'ap_ht', True))} | "
                 f"{f3(seed_mean(n, 'macro_cluster_ap'))} | {f3(mp)} | {f3(seed_mean(n, 'ap_ht_no_top10', True))} | "
                 f"{f3(seed_mean(n, 'ap_ht_both_nonessential'))} / {f3(seed_mean(n, 'ap_ht_mixed'))} / "
                 f"{f3(seed_mean(n, 'ap_ht_both_essential'))} | {f3(seed_mean(n, 'auroc_ht'))} | {inp} | {fs} |")

    L += ["", "## HT-AP per seed (fold mean ± sd over the 5 test folds)", "",
          "| entry | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean |", "|---|" + "---|" * (len(SEEDS) + 1)]
    for n in entries():
        cells = []
        for s in SEEDS:
            a = load_run(n, s)["summary"]["ap_ht"]
            cells.append(f"{a['mean']:.3f} ± {a['sd']:.3f}")
        L.append(f"| {lab(n)} | " + " | ".join(cells) + f" | {f3(seed_mean(n, 'ap_ht'))} |")

    L += ["", "## Nested-by-fold selection (select on 4 folds by mean HT-AP, score the 5th)", "",
          "| seed | among 4 z-averages: nested held-out | selected per outer fold | in-sample best | among all "
          f"{len(combos())} combinations: nested held-out | selected per outer fold | in-sample best |",
          "|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        z, a = nst[str(s)]["zavg"], nst[str(s)]["all_combos"]
        L.append(f"| {s} | {z['nested_mean_held_out']:.3f} | {', '.join(r['selected'] for r in z['outer'])} | "
                 f"{z['selected_in_sample']['name']} {z['selected_in_sample']['ap_ht']:.3f} | "
                 f"{a['nested_mean_held_out']:.3f} | {', '.join(r['selected'] for r in a['outer'])} | "
                 f"{a['selected_in_sample']['name']} {a['selected_in_sample']['ap_ht']:.3f} |")
    zm = np.mean([nst[str(s)]["zavg"]["nested_mean_held_out"] for s in SEEDS])
    am = np.mean([nst[str(s)]["all_combos"]["nested_mean_held_out"] for s in SEEDS])
    L += [f"| mean | {zm:.3f} | | | {am:.3f} | | |"]

    L += ["", "## Paired cluster bootstrap on pooled OOF predictions (ΔHT-AP, 95% CI; 1,000 resamples per seed)", "",
          "### vs C_single (B), all pairs", "", "| comparison | " + " | ".join(f"seed {s}" for s in SEEDS) + " |",
          "|---|" + "---|" * len(SEEDS)]
    for c in combos() + ["ref_E", "ref_H"]:
        L.append(f"| {lab(c)} − B | " + " | ".join(ci(cmp["seeds"][str(s)]["all"]["diff"][f"{c}-{C_SINGLE}"])
                                                    for s in SEEDS) + " |")
    L += ["", "### vs C_single (B), pairs not touching the top-10 clusters", "",
          "| comparison | " + " | ".join(f"seed {s}" for s in SEEDS) + " |", "|---|" + "---|" * len(SEEDS)]
    for c in combos():
        L.append(f"| {lab(c)} − B | " + " | ".join(ci(cmp["seeds"][str(s)]["no_top10"]["diff"][f"{c}-{C_SINGLE}"])
                                                    for s in SEEDS) + " |")
    for b in BASE_REFS:
        L += ["", f"### vs {label[b]} (all pairs)", "", "| comparison | " + " | ".join(f"seed {s}" for s in SEEDS) + " |",
              "|---|" + "---|" * len(SEEDS)]
        for c in [C_SINGLE] + combos():
            L.append(f"| {lab(c)} − {label[b]} | " + " | ".join(ci(cmp["seeds"][str(s)]["all"]["diff"][f"{c}-{b}"])
                                                               for s in SEEDS) + " |")
    L += ["", "### supervised vs z-averages (all pairs)", "",
          "| comparison | " + " | ".join(f"seed {s}" for s in SEEDS) + " |", "|---|" + "---|" * len(SEEDS)]
    for c in [x for x in combos() if x not in ZAVGS]:
        for z in ZAVGS:
            L.append(f"| `{c}` − `{z}` | " + " | ".join(ci(cmp["seeds"][str(s)]["all"]["diff"][f"{c}-{z}"])
                                                       for s in SEEDS) + " |")

    ctrl = sorted((OUT / "controls").glob("*.json")) if (OUT / "controls").exists() else []
    L += ["", "## Held-fold label-shuffle controls (SEEDS[0]; true training labels, held-fold labels permuted)", "",
          "| model | mean HT-AP | mean HT prevalence | per-fold AP |", "|---|---|---|---|"]
    for p in ctrl:
        c = json.loads(p.read_text())
        L.append(f"| `{c['model']}` | {c['mean_ap_ht']:.4f} | {c['mean_prevalence']:.4f} | "
                 + ", ".join(f"{f['ap_ht']:.4f}" for f in c["folds"]) + " |")

    L += ["", "## Supervised feature weights (SEEDS[0]; LR = mean standardised coefficient over bags × folds, "
          "LGBM = gain share)", "", "| model | " + " | ".join(FEAT7) + " | gene feats (sum) |", "|---|" + "---|" * 8]
    for n in [x for x in combos() if x not in ZAVGS]:
        fi = load_run(n, SEEDS[0])["phaseB_v2"]["feature_importance"]
        nm = {LAYERS[k][1]: k for k in LAYERS} | {"gls_signed": "G"}
        v = {nm[k]: x for k, x in fi.items() if k in nm}
        g = sum(x for k, x in fi.items() if k not in nm)
        L.append(f"| `{n}` | " + " | ".join(f"{v.get(k, float('nan')):+.3f}" for k in FEAT7) + f" | {g:+.3f} |")

    if rt:
        L += ["", "## Runtime of label-free inputs (fresh process, all 596,597 scorable pairs)", "",
              "| set | members (s each, computed in this order in one engine) | total (s) | peak RSS (GB) | max |Δ| vs cache |",
              "|---|---|---|---|---|"]
        for k, r in rt.items():
            if not isinstance(r, dict) or "members" not in r:
                continue
            L.append(f"| {k} | " + ", ".join(f"{m['key']} {m['s']:.0f}" for m in r["members"])
                     + f" | {r['total_s']:.0f} | {r['peak_rss_gb']} | {r['max_abs_diff_vs_cache']:.1e} |")
        if "gene_features_s" in rt:
            L.append(f"\nGene-feature tables (genome-wide stats + bins): {rt['gene_features_s']:.1f} s.")

    d = dec
    L += ["", "## D3 decision (applied literally)", "",
          f"- C_comb = argmax over the {len(combos())} pre-specified combinations of the 5-seed mean held-out HT-AP: "
          f"**`{d['c_comb']}`** (" + ", ".join(f"{k} {v:.3f}" for k, v in d["ranking"].items()) + ").",
          f"- Criterion 1 (pooled-OOF Δ vs C_single, CI excluding 0 in ≥3/5 seeds): CI above 0 in "
          f"**{d['n_seeds_ci_above_0']}/5** seeds (below 0 in {d['n_seeds_ci_below_0']}/5): "
          + "; ".join(f"seed {x['seed']} {x['diff']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}]" for x in d["deltas"])
          + f" → {'met' if d['criterion_ci'] else 'NOT met'}.",
          f"- Criterion 2 (macro per-cluster AP not lower; 5-seed fold mean): C_comb {d['macro_fold_mean']['c_comb']:.3f} "
          f"vs C_single {d['macro_fold_mean']['c_single']:.3f} → {'met' if d['criterion_macro'] else 'NOT met'} "
          f"(pooled-OOF macro {d['macro_pooled']['c_comb']:.3f} vs {d['macro_pooled']['c_single']:.3f}).",
          f"- **Primary = `{d['primary']}`; secondary = `{d['secondary']}`.** Fixed-sequence order for the IFT look: "
          f"1. `{d['fixed_sequence'][0]}`, 2. `{d['fixed_sequence'][1]}`.", ""]
    notes = OUT / "notes.md"
    if notes.exists():
        L += [notes.read_text()]
    (OUT / "README.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-8:]))


if __name__ == "__main__":
    main()
