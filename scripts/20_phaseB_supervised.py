"""Phase B (DESIGN v3 §4) first pass: supervised pair scorers on the DEV cluster CV, SEEDS[0] only.

PROVISIONAL: Phase-A layer configs are not frozen yet; the 12-layer set below was picked by name a priori to span
the families (plain/ridge-whitened/PC-removed spectral filters, Wainberg-centred GLS r, row-z, sparse second-order,
balanced mutual rank) without looking at any Phase-A screening result. Numbers here size the supervision effect
and check the controls; they are not a model selection.

Runs (every one is a logged trial via src.evaluate.run_dev_cv; JSON -> results/phaseB/<tag>.json):
  (ref) gls_signed baseline and the label-free z-mean of the 12 provisional layers (same OOF pairs as below).
  (a) LGBM / LR over Pearson + Spearman + GLS signed scores only.
  (b) gene-stat-only models (binned; raw = 'raw_gene_feats' ablation), chromosome-only control.
  (c) label-shuffle controls on (a) (src.evaluate.label_shuffle_control, wrapped in log_trial start/end):
      'train' = all labels permuted before fitting, evaluated on true held-fold labels; 'held' = true training
      labels, held-fold labels permuted.
  (d) LGBM / LR over the 12 provisional layers +/- binned gene features +/- chromosome features (+ raw gene ablation).
  (e) the same models under the secondary splits: 'gene' (C3, random protein-disjoint) and 'random' (pair-level,
      leaky: test proteins seen in training).
Pooled out-of-fold (OOF) metrics: the 5 test folds' scores are pooled (all within-fold pairs; HT weights 1 within
cluster / K between clusters for the cluster scheme) and a 1,000-resample paired cluster bootstrap gives CIs of
ΔHT-AP between models (results/phaseB/bootstrap_cluster_seed0.json).

Usage: ./.env/bin/python scripts/20_phaseB_supervised.py [--allow-dirty] [--only TAG,...] [--readme-only]
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

from src import evaluate, metrics, scorers, supervised as S  # noqa: E402
from src import gene_features as gf  # noqa: E402
from src.config import CACHE, N_FOLDS, RESULTS, ROOT, SEEDS  # noqa: E402

OUT = RESULTS / "phaseB"
OOF_DIR = CACHE / "phaseB_oof"
SEED = SEEDS[0]
BASE3 = ["pearson_signed", "spearman_signed", "gls_signed"]
PROV12 = ["spec_k0_tinf", "spec_k0_t0.1", "spec_k5_t0.1", "spec_k20_t0.1", "spec_k20_tinf", "spec_k0_t0_line",
          "rowz[spec_k0_t0.1]", "rowz[spec_k5_t0.1]", "rowz[spec_k20_t0.1]", "socos_q1[spec_k5_t0.1]",
          "socos_q1[spec_k20_t0.1]", "mrbal[spec_k20_t0.1]"]

_MAT = {}


def mat():
    if "m" not in _MAT:
        _MAT["m"] = scorers.real_matrix()
    return _MAT["m"]


def blocks(spec):
    out = []
    for part in spec.split("+"):
        if part == "base3":
            out.append(S.BaselineFeatures(BASE3, mat()))
        elif part == "prov12":
            out.append(S.LayerFeatures(PROV12))
        elif part == "genebin":
            out.append(S.GeneFeatures("binned"))
        elif part == "generaw":
            out.append(S.GeneFeatures("raw"))
        elif part == "chrom":
            out.append(S.ChromFeatures())
        else:
            raise ValueError(part)
    return out


def factory(name):
    """name = <model>_<features joined by '+'> or ref_<...>."""
    if name == "ref_gls_signed":
        sc = scorers.make_baseline("gls_signed", mat())
        sc.name = name
        return sc
    if name == "ref_zmean_prov12":
        return S.ZMeanScorer(name, PROV12)
    model, spec = name.split("_", 1)
    return S.make_supervised(name, blocks(spec), model={"lgbm": "lgbm", "lgbmnm": "lgbm", "lr": "logreg"}[model],
                             seed=SEED, monotone=model != "lgbmnm")


# (name, schemes, group)
EXPERIMENTS = [
    ("ref_gls_signed", ("cluster", "gene", "random"), "ref"),
    ("ref_zmean_prov12", ("cluster", "gene", "random"), "ref"),
    ("lgbm_base3", ("cluster", "gene", "random"), "a"),
    ("lr_base3", ("cluster",), "a"),
    ("lgbm_genebin", ("cluster", "gene", "random"), "b"),
    ("lgbm_generaw", ("cluster", "gene", "random"), "b"),
    ("lr_genebin", ("cluster",), "b"),
    ("lgbm_chrom", ("cluster",), "b"),
    ("lgbm_prov12", ("cluster", "gene", "random"), "d"),
    ("lgbm_prov12+genebin", ("cluster", "gene", "random"), "d"),
    ("lgbm_prov12+chrom", ("cluster",), "d"),
    ("lgbm_prov12+genebin+chrom", ("cluster",), "d"),
    ("lgbm_prov12+generaw", ("cluster", "random"), "d"),
    ("lr_prov12", ("cluster",), "d"),
    ("lr_prov12+genebin", ("cluster", "gene", "random"), "d"),
    ("lr_prov12+chrom", ("cluster",), "d"),
    ("lr_prov12+genebin+chrom", ("cluster",), "d"),
]
SHUFFLE = [("lgbm_base3", "train"), ("lgbm_base3", "held"), ("lr_base3", "train")]
# repeated all-label shuffle (permutation + split seed = each of SEEDS); lgbmnm = LGBM without monotone constraints
SHUFFLE_REP = ["lgbm_base3", "lgbmnm_base3", "lr_base3", "lgbm_genebin"]
DIFFS = [("lgbm_base3", "ref_gls_signed"), ("lr_base3", "ref_gls_signed"),
         ("lgbm_prov12", "ref_gls_signed"), ("lgbm_prov12", "ref_zmean_prov12"), ("ref_zmean_prov12", "ref_gls_signed"),
         ("lgbm_prov12+genebin", "lgbm_prov12"), ("lgbm_prov12+genebin", "lgbm_genebin"),
         ("lgbm_prov12+genebin", "ref_gls_signed"), ("lgbm_prov12+chrom", "lgbm_prov12"),
         ("lgbm_prov12+generaw", "lgbm_prov12"), ("lr_prov12", "ref_gls_signed"), ("lr_prov12+genebin", "lr_prov12"),
         ("lgbm_prov12", "lr_prov12"), ("lgbm_genebin", "ref_gls_signed")]


class Recorder:
    """Transparent wrapper that keeps the scores returned to the evaluator (for pooled OOF metrics)."""

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


def oof_scores(rec, u, scheme):
    sp = evaluate.make_dev_split(u, scheme, SEED)
    oof = np.full(len(u), np.nan)
    if rec.label_free:
        idx, s = rec.calls[0]
        full = np.full(len(u), np.nan)
        full[idx] = s
        for k in range(N_FOLDS):
            te = sp.test_mask(k)
            oof[te] = full[te]
    else:
        assert len(rec.calls) == N_FOLDS
        for k, (idx, s) in enumerate(rec.calls):
            assert np.array_equal(idx, np.flatnonzero(sp.test_mask(k)))
            oof[idx] = s
    return oof


def tag_of(name, scheme):
    return f"{name}_{scheme}_seed{SEED}"


def run_one(name, scheme, group, u, allow_dirty):
    rec = Recorder(factory(name))
    t0 = time.perf_counter()
    res = evaluate.run_dev_cv(rec, seed=SEED, scheme=scheme, universe=u, out_dir=OUT, tag=tag_of(name, scheme),
                              allow_dirty=allow_dirty, extra=dict(grid="phaseB_v1", group=group, provisional=True))
    oof = oof_scores(rec, u, scheme)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OOF_DIR / f"{tag_of(name, scheme)}.npy", oof.astype(np.float64))
    idx = np.flatnonzero(np.isfinite(oof))
    w = evaluate.design_weights(u, idx, scheme)
    pooled = evaluate.suite(u, idx, oof[idx], w)
    extra = dict(pooled_oof=pooled, wall_s=round(time.perf_counter() - t0, 1), group=group)
    sc = rec.sc
    if isinstance(sc, S.SupervisedPairScorer):
        extra.update(feature_importance=sc.feature_importance(), fit_log=sc.fit_log,
                     bind_seconds=round(sc.bind_seconds, 2))
    path = Path(res["path"])
    full = json.loads(path.read_text())
    full["phaseB"] = extra
    path.write_text(json.dumps(full, indent=1, default=evaluate._json_default))
    print(f"{tag_of(name, scheme):45s} HT-AP {res['summary']['ap_ht']['mean']:.3f} ± {res['summary']['ap_ht']['sd']:.3f}"
          f"  pooled-OOF {pooled['ap_ht']:.3f}  {extra['wall_s']}s", flush=True)


def run_shuffle(name, mode, u, allow_dirty):
    evaluate.check_clean(allow_dirty)
    tag = f"control_shuffle_{mode}_{name}_cluster_seed{SEED}"
    tid = uuid.uuid4().hex[:12]
    evaluate.log_trial("start", tid, tag=tag, scorer=factory(name).describe(), seed=SEED, scheme="cluster",
                       universe=u.name, git=evaluate.git_state(), allow_dirty=allow_dirty,
                       extra=dict(grid="phaseB_v1", group="c", control=f"label_shuffle_{mode}"))
    t0 = time.perf_counter()
    try:
        c = evaluate.label_shuffle_control(lambda: factory(name), u, seed=SEED, mode=mode, scheme="cluster")
    except BaseException as e:
        evaluate.log_trial("end", tid, tag=tag, status="error", error=repr(e)[:500])
        raise
    c.update(tag=tag, trial_id=tid, model=name, runtime_s=round(time.perf_counter() - t0, 1),
             run_record=evaluate.run_record(seed=SEED, allow_dirty=allow_dirty, universe=u.name))
    path = OUT / f"{tag}.json"
    path.write_text(json.dumps(c, indent=1, default=evaluate._json_default))
    evaluate.log_trial("end", tid, tag=tag, status="ok", result=str(path.relative_to(ROOT)),
                       primary=dict(metric="ap_ht", mean=c["mean_ap_ht"], prevalence=c["mean_prevalence"]))
    print(f"{tag:45s} AP {c['mean_ap_ht']:.4f} vs HT prevalence {c['mean_prevalence']:.4f}", flush=True)


def run_shuffle_rep(name, u, allow_dirty):
    """All-label shuffle repeated over SEEDS (each seed = new permutation and split). Records per fold the AP, the HT
    prevalence and the Spearman correlation of the shuffled model's test scores with GLS (direction diagnostic)."""
    from scipy.stats import spearmanr
    evaluate.check_clean(allow_dirty)
    tag = f"control_shuffle_rep_train_{name}_cluster"
    tid = uuid.uuid4().hex[:12]
    evaluate.log_trial("start", tid, tag=tag, scorer=factory(name).describe(), seed=list(SEEDS), scheme="cluster",
                       universe=u.name, git=evaluate.git_state(), allow_dirty=allow_dirty,
                       extra=dict(grid="phaseB_v1", group="c", control="label_shuffle_train_repeated"))
    t0 = time.perf_counter()
    gls = factory("ref_gls_signed").bind(u).score(np.arange(len(u)))
    reps = []
    try:
        for sd in SEEDS:
            recs = []

            def fac():
                recs.append(Recorder(factory(name)))
                return recs[-1]

            c = evaluate.label_shuffle_control(fac, u, seed=sd, mode="train", scheme="cluster")
            for f, r in zip(c["folds"], recs):
                idx, sc = r.calls[-1]
                f["spearman_with_gls"] = float(spearmanr(sc, gls[idx]).statistic) if np.ptp(sc) > 0 else 0.0
            c["perm_seed"] = sd
            reps.append(c)
    except BaseException as e:
        evaluate.log_trial("end", tid, tag=tag, status="error", error=repr(e)[:500])
        raise
    aps = np.array([f["ap_ht"] for c in reps for f in c["folds"]])
    prev = np.array([f["ht_prevalence"] for c in reps for f in c["folds"]])
    rho = np.array([f["spearman_with_gls"] for c in reps for f in c["folds"]])
    out = dict(tag=tag, trial_id=tid, model=name, reps=reps, mean_ap_ht=float(aps.mean()), median_ap_ht=float(np.median(aps)),
               mean_prevalence=float(prev.mean()), frac_folds_ap_above_2x_prev=float(np.mean(aps > 2 * prev)),
               frac_folds_positive_rho=float(np.mean(rho > 0)), mean_abs_rho=float(np.abs(rho).mean()),
               ap_when_rho_pos=float(aps[rho > 0].mean()) if (rho > 0).any() else None,
               ap_when_rho_nonpos=float(aps[rho <= 0].mean()) if (rho <= 0).any() else None,
               runtime_s=round(time.perf_counter() - t0, 1),
               run_record=evaluate.run_record(seed=list(SEEDS), allow_dirty=allow_dirty, universe=u.name))
    path = OUT / f"{tag}.json"
    path.write_text(json.dumps(out, indent=1, default=evaluate._json_default))
    evaluate.log_trial("end", tid, tag=tag, status="ok", result=str(path.relative_to(ROOT)),
                       primary=dict(metric="ap_ht", mean=out["mean_ap_ht"], prevalence=out["mean_prevalence"]))
    print(f"{tag:45s} AP mean {out['mean_ap_ht']:.4f} median {out['median_ap_ht']:.4f} prev {out['mean_prevalence']:.4f} "
          f"rho>0 {out['frac_folds_positive_rho']:.2f} |rho| {out['mean_abs_rho']:.2f}", flush=True)


def bootstrap(u):
    names = sorted({n for n, sch, _ in EXPERIMENTS if "cluster" in sch})
    oof = {n: np.load(OOF_DIR / f"{tag_of(n, 'cluster')}.npy") for n in names}
    idx = np.flatnonzero(np.isfinite(oof["ref_gls_signed"]))
    for n in names:
        assert np.array_equal(np.flatnonzero(np.isfinite(oof[n])), idx), n
    b = metrics.cluster_bootstrap(u.y[idx], {n: oof[n][idx] for n in names}, u.col("cluster_a")[idx],
                                  u.col("cluster_b")[idx], f=1.0 / N_FOLDS, n_boot=1000, seed=SEED, diffs=DIFFS)
    b.update(note="pooled OOF predictions of the 5 cluster-CV test folds (SEEDS[0]); HT weights 1 within / K between "
                  "clusters; paired cluster bootstrap over the dev clusters", n_pairs=int(len(idx)),
             n_pos=int(u.y[idx].sum()), run_record=evaluate.run_record(seed=SEED, universe=u.name))
    (OUT / f"bootstrap_cluster_seed{SEED}.json").write_text(json.dumps(b, indent=1, default=evaluate._json_default))


def diagnostics(u):
    t = gf.real_gene_tables()
    dev_genes = np.unique(np.r_[u.col("ia"), u.col("ib")])
    d = dict(gene_bins=dict(zip(gf.STATS, t["n_bins"])),
             bin_edges={s: e.tolist() for s, e in zip(gf.STATS, t["edges"])},
             binned_signature_uniqueness_dev_genes=gf.signature_uniqueness(t["bins"], dev_genes),
             raw_signature_uniqueness_dev_genes=gf.signature_uniqueness(np.round(t["stats"], 6), dev_genes),
             provisional_layers=PROV12, baselines=BASE3, lgbm_defaults=S.LGBM_DEFAULTS, early_stop=S.EARLY_STOP,
             run_record=evaluate.run_record(seed=SEED, universe=u.name))
    (OUT / "diagnostics.json").write_text(json.dumps(d, indent=1, default=evaluate._json_default))


# ------------------------------------------------------------------------------------------ README
def load(tag):
    p = OUT / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def f3(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3f}"


def write_readme():
    L = ["# Phase B (first pass): supervised pair scorers on the dev CV — PROVISIONAL", "",
         "Generated by `scripts/20_phaseB_supervised.py` (do not edit by hand). DEV pairs only; IFT never loaded. "
         f"SEEDS[0] only. Code: `src/supervised.py`, `src/gene_features.py`, tests `tests/test_supervised.py`.", "",
         "**Provisional:** Phase-A configs are not frozen. `prov12` = 12 cached layers picked by name a priori "
         "(no Phase-A screening result consulted): " + ", ".join(f"`{n}`" for n in PROV12) + ". "
         "Nothing here is a selection; it sizes the supervision effect and runs the §3.9 controls.", "",
         "Columns: HT-AP mean ± sd over the 5 test folds (primary, §3.5); `pooled OOF HT-AP` = AP over the pooled "
         "out-of-fold predictions of the 5 folds (HT weights 1 within / K between clusters for the cluster scheme; "
         "fold models differ, so pooling mixes score scales — reported because §3.7(ii) uses it); macro = macro "
         "per-cluster AP; no-largest = HT-AP without the largest component; non-ess = both-non-essential stratum; "
         "runtime = wall clock of the whole CV run (bind/setup + 5 × fit+score), 8-core laptop shared with another job.",
         "", "## Cluster-grouped dev CV (primary)", "",
         "| group | model | HT-AP mean ± sd | pooled OOF HT-AP | macro-cluster AP | AP w/o largest | both non-ess. | "
         "raw AP | runtime (s) |", "|---|---|---|---|---|---|---|---|---|"]
    for name, schemes, g in EXPERIMENTS:
        r = load(tag_of(name, "cluster"))
        if r is None:
            continue
        s, pb = r["summary"], r.get("phaseB", {})
        L.append(f"| {g} | `{name}` | {s['ap_ht']['mean']:.3f} ± {s['ap_ht']['sd']:.3f} | "
                 f"{f3(pb.get('pooled_oof', {}).get('ap_ht'))} | {f3(s['macro_cluster_ap']['mean'])} | "
                 f"{f3(s['ap_ht_no_largest_component']['mean'])} | {f3(s['ap_ht_both_nonessential']['mean'])} | "
                 f"{f3(s['ap_raw']['mean'])} | {r['runtime_s']['total_wall']:.0f} |")
    r0 = load(tag_of("ref_gls_signed", "cluster"))
    if r0:
        prev = np.mean([f["ht_prevalence"] for f in r0["folds"]])
        L += ["", f"HT prevalence of the test folds (mean): {prev:.4f}."]
    b = OUT / f"bootstrap_cluster_seed{SEED}.json"
    if b.exists():
        bb = json.loads(b.read_text())
        L += ["", "## Paired cluster bootstrap on pooled OOF predictions (1,000 resamples over dev clusters)", "",
              "| comparison | ΔHT-AP | 95% CI | relative gain [95% CI] | p (two-sided) |", "|---|---|---|---|---|"]
        for k, d in bb["diff"].items():
            L.append(f"| `{k}` | {d['diff']:+.3f} | [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] | "
                     f"{100 * d['rel_gain']:+.1f}% [{100 * d['rel_ci_low']:+.1f}, {100 * d['rel_ci_high']:+.1f}] | "
                     f"{d['p_two_sided']:.3f} |")
    L += ["", "## Label-shuffle controls (c), cluster CV", "",
          "| model | mode | mean HT-AP | mean HT prevalence | per-fold AP |", "|---|---|---|---|---|"]
    for name, mode in SHUFFLE:
        c = load(f"control_shuffle_{mode}_{name}_cluster_seed{SEED}")
        if c:
            L.append(f"| `{name}` | {mode} | {c['mean_ap_ht']:.4f} | {c['mean_prevalence']:.4f} | "
                     + ", ".join(f"{f['ap_ht']:.4f}" for f in c["folds"]) + " |")
    L += ["", "`train` = every label permuted before fitting (evaluated against the true held-fold labels); "
          "`held` = true training labels, held-fold labels permuted.", "",
          "Repeated all-label shuffle (5 permutations × 5 folds; permutation and split seed = each of SEEDS; "
          "`lgbmnm` = LGBM without monotone constraints; ρ = Spearman correlation of the shuffled model's test "
          "scores with GLS):", "",
          "| model | mean AP | median AP | mean HT prevalence | folds AP > 2×prev | folds ρ > 0 | mean abs(ρ) | "
          "AP when ρ > 0 | AP when ρ ≤ 0 |", "|---|---|---|---|---|---|---|---|---|"]
    for name in SHUFFLE_REP:
        c = load(f"control_shuffle_rep_train_{name}_cluster")
        if c:
            L.append(f"| `{name}` | {c['mean_ap_ht']:.4f} | {c['median_ap_ht']:.4f} | {c['mean_prevalence']:.4f} | "
                     f"{c['frac_folds_ap_above_2x_prev']:.2f} | {c['frac_folds_positive_rho']:.2f} | "
                     f"{c['mean_abs_rho']:.2f} | {f3(c['ap_when_rho_pos'])} | {f3(c['ap_when_rho_nonpos'])} |")
    L += ["",
          "## Split comparison (e): leakage gap", "",
          "| model | cluster (primary) | gene = C3 protein-disjoint | random pair (leaky, C1-type) | gap random − cluster | "
          "excess gap vs label-free z-mean |", "|---|---|---|---|---|---|"]
    zg = None
    zr, zc = load(tag_of("ref_zmean_prov12", "random")), load(tag_of("ref_zmean_prov12", "cluster"))
    if zr and zc:
        zg = zr["summary"]["ap_ht"]["mean"] - zc["summary"]["ap_ht"]["mean"]
    for name, schemes, g in EXPERIMENTS:
        if len(schemes) < 2:
            continue
        cells, means = [], {}
        for sch in ("cluster", "gene", "random"):
            r = load(tag_of(name, sch))
            if r is None:
                cells.append("—")
                continue
            means[sch] = r["summary"]["ap_ht"]["mean"]
            cells.append(f"{r['summary']['ap_ht']['mean']:.3f} ± {r['summary']['ap_ht']['sd']:.3f}")
        ok = {"random", "cluster"} <= set(means)
        gap = f"{means['random'] - means['cluster']:+.3f}" if ok else "—"
        ex = f"{means['random'] - means['cluster'] - zg:+.3f}" if ok and zg is not None else "—"
        L.append(f"| `{name}` | " + " | ".join(cells) + f" | {gap} | {ex} |")
    L += ["", "The random-pair split also changes which pairs are evaluated (label-free scorers, which cannot leak, "
          "gain +0.01–0.03), so the supervised-specific leakage is the excess over the label-free z-mean of the same "
          "layers."]
    d = OUT / "diagnostics.json"
    if d.exists():
        dd = json.loads(d.read_text())
        bu, ru = dd["binned_signature_uniqueness_dev_genes"], dd["raw_signature_uniqueness_dev_genes"]
        L += ["", "## Gene-feature identifiability diagnostic", "",
              f"Bins per statistic (genome-wide quantile edges over all 18,434 genes): {dd['gene_bins']}. "
              f"Among the {bu['n_genes']} dev genes, the binned 7-statistic signature takes {bu['n_signatures']} "
              f"distinct values and is unique for {100 * bu['frac_unique']:.0f}% of genes (raw: "
              f"{100 * ru['frac_unique']:.0f}%). Binning alone therefore does NOT make proteins anonymous; the protection "
              "is num_leaves ≤ 7 + min_child_samples ≥ 2,000 (a leaf cannot hold one protein's pairs) plus the "
              "cluster-disjoint evaluation, and the gene-only rows above measure what is left."]
    L += ["", "## Findings (seed 0, provisional layer set; written after the run)", "",
          "1. (a) Supervision alone over Pearson + Spearman + GLS: LGBM +0.026 fold-mean HT-AP over GLS (pooled OOF "
          "+0.008, CI includes 0); LR +0.034 (pooled +0.040, CI [+0.012, +0.075]). A small, real gain from "
          "re-weighting the three baselines; much smaller than the gain from new layers.",
          "2. (d) LGBM over the 12 provisional layers: 0.383 vs GLS 0.288 (pooled ΔOOF +0.066, CI [+0.001, +0.153]); "
          "but the label-free unweighted z-mean of the same 12 layers already reaches 0.357 (Δ LGBM − z-mean pooled "
          "+0.006, CI [−0.034, +0.032]). Most of the gain is the layers, not the supervision.",
          "3. Gene features: binned gene features add +0.002 (fold mean) / +0.033 (pooled, CI includes 0) to LGBM and "
          "nothing to LR; raw gene features +0.018 / +0.037 (CI includes 0) and the largest excess leakage gap. "
          "Chromosome features: +0.001 / +0.023 (CI includes 0); chromosome-only model = prevalence.",
          "4. Controls: gene-stat-only models give HT-AP 0.019–0.024 vs HT prevalence 0.009 (≈2.5× prevalence, "
          "transferable essentiality propensity, 6% of GLS); identical under cluster, C3 and random splits (no "
          "memorisation inflation: the leaf constraints work). Held-fold shuffle = prevalence. All-label shuffle is "
          "NOT ≈ prevalence for monotone LGBM (0.146; every shuffled model correlates ρ ≈ 0.78 with GLS) because the "
          "monotone constraint itself encodes 'more similar = more likely'; without constraints the median is at "
          "prevalence (0.009), and for LR the shuffled fit picks a random direction in the 3-feature similarity "
          "space (AP 0.121 when ρ > 0, 0.014 when ρ ≤ 0). Not leakage: permuted labels carry no information; "
          "gene-only shuffle = prevalence.", ""]
    L += ["", "## Model details", "",
          "- LightGBM: num_leaves 7, min_child_samples 2,000, learning rate 0.05, row/column subsample 0.8, L2 1, "
          "monotone +1 on every similarity feature (layers, baselines), unconstrained gene/chromosome features. "
          "5 bags; each = all training positives + 50 × n_pos random training negatives; per bag early stopping "
          "(100 rounds, AP) on an inner cluster-grouped validation split of the training fold (inner fold 1/4 of "
          "training clusters, val pairs un-subsampled), then refit on the whole bag with the best iteration count. "
          "Score = mean raw margin over bags.",
          "- Logistic regression: StandardScaler + L2 (C = 1) per bag, same bags and weights, mean decision function.",
          "- Sample weights: positives 1/sqrt(n_pos touching their cluster) (mean over the two clusters if the "
          "positive spans two), rescaled to mean 1; negatives 1. No class weights.",
          "- Layer features are z-scored with the manifest genome-wide null moments (affects LR only; trees are "
          "invariant). Baseline features are the raw scores of `src/scorers` (Pearson/Spearman pairwise-complete, "
          "GLS marginalised t).",
          "- Gene features (`src/gene_features.py`): mean, sd, skew, fraction of lines < −0.5, NaN fraction, min, "
          "5th percentile per gene; ≤ 5 genome-wide quantile bins; pair features min/max per statistic "
          "(`generaw` = raw values, ablation). Chromosome: same chromosome, same arm (HGNC cytoband).",
          "- Inner validation always uses the cluster-grouped scheme, also when the outer split is gene/random.",
          "- Per-run JSON: fold metrics, run_record, `phaseB.pooled_oof`, feature importances (LGBM gain share / LR "
          "standardised coefficients), per-bag early-stopping logs. OOF score arrays: data_cache/phaseB_oof/ "
          "(gitignored).", ""]
    (OUT / "README.md").write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--readme-only", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.readme_only:
        subprocess.run(["sh", str(ROOT / "scripts" / "check_data.sh")], check=True, stdout=subprocess.DEVNULL)
        from src.universe import load_dev_universe
        u = load_dev_universe()
        only = set(filter(None, a.only.split(",")))
        diagnostics(u)
        for name, schemes, g in EXPERIMENTS if "shuffle_rep" not in only else []:
            if only and name not in only:
                continue
            for sch in schemes:
                run_one(name, sch, g, u, a.allow_dirty)
        for name, mode in SHUFFLE if "shuffle_rep" not in only else []:
            if only and name not in only:
                continue
            run_shuffle(name, mode, u, a.allow_dirty)
        for name in SHUFFLE_REP:
            if only and name not in only and "shuffle_rep" not in only:
                continue
            run_shuffle_rep(name, u, a.allow_dirty)
        if not only:
            bootstrap(u)
    write_readme()


if __name__ == "__main__":
    main()
