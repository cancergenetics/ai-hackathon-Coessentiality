"""Unsupervised baselines (Pearson, Spearman, GLS; optional GLS with Wainberg olfactory-PC correction) on all
scorable HuMAP train pairs, with stratified-bootstrap AUPRC CIs, paired AUPRC differences, and per-fold metrics
for the three split schemes (5 folds, seed SEEDS[0]) so later supervised methods are compared on identical test
pairs (split fingerprints recorded).

Scores (derived from DepMap/HuMAP) are cached to data_cache/scores/ (gitignored); metrics to results/baselines/.

Usage:
  ./.env/bin/python scripts/02_run_baselines.py --matrix CRISPRGeneEffect --olfactory
  ./.env/bin/python scripts/02_run_baselines.py --matrix ScreenGeneEffect --library Avana
  ./.env/bin/python scripts/02_run_baselines.py --readme-only
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import baselines, data, metrics, splits  # noqa: E402
from src.config import CACHE, RESULTS, SEEDS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("02")
OUT = RESULTS / "baselines"
SCHEMES = ("random", "gene", "complex")
N_FOLDS = 5


def scorable_pairs():
    pairs, umap = data.load_pairs()
    sc = pairs[pairs.scorable].reset_index(drop=True)
    excl = dict(pos=int(((~pairs.scorable) & (pairs.label == 1)).sum()),
                neg=int(((~pairs.scorable) & (pairs.label == 0)).sum()))
    return sc, excl


def run(matrix, library, olfactory, n_boot, n_boot_fold):
    tag = f"{matrix}_{library or 'all'}"
    X, samples, genes = data.load_matrix(matrix)
    meta = data.sample_metadata(matrix, samples)
    if library:
        if matrix == "ScreenGeneEffect":
            keep = (meta.library == library).values
        else:
            keep = (meta.libraries == library).values        # models screened ONLY with this library
        X, samples, meta = X[keep], samples[keep], meta[keep]
    log.info("%s: X %s", tag, X.shape)
    sc, excl = scorable_pairs()
    y = sc.label.values.astype(int)
    ia, ib = sc.ia.values, sc.ib.values

    scores, runtime = {}, {}
    methods = ["pearson", "spearman", "gls", "gls_pinv"] + (["gls_olf"] if olfactory else [])
    extra = {}
    for m in methods:
        t0 = time.perf_counter()
        if m == "gls_olf":
            ann = data.gene_annotation(genes)
            olf = np.flatnonzero(ann.gene_group.fillna("").str.contains("Olfactory receptors").values)
            Xc, kept = baselines.olfactory_correction(X, olf, n_pc=4)
            out = baselines.gls_pairs(Xc, ia, ib)
            extra["gls_olf"] = dict(n_olfactory_genes=int(len(olf)), dropped_last_samples=4,
                                    n_samples_used=int(kept.sum()),
                                    samples_dropped_for_rank=out["samples_dropped_for_rank"])
            del Xc
        elif m in ("gls", "gls_pinv"):
            Sigma, ncov = baselines.sample_covariance(X)
            mode = "drop" if m == "gls" else "pinv"
            out = baselines.gls_pairs(X, ia, ib, Sigma=Sigma, rank_deficiency=mode)
            extra[m] = dict(n_genes_for_covariance=ncov, n_cov_blocks=out["n_cov_blocks"], rank_deficiency=mode,
                            sigma_rank=int(np.linalg.matrix_rank(Sigma, hermitian=True)), sigma_dim=len(Sigma),
                            samples_dropped_for_rank=out["samples_dropped_for_rank"],
                            df_range=[int(out["df"].min()), int(out["df"].max())])
            del Sigma
        else:
            out = baselines.score(m, X, ia, ib)
        runtime[m] = round(time.perf_counter() - t0, 2)
        log.info("%s %s: %.1fs", tag, m, runtime[m])
        scores[f"{m}_signed"] = out["signed"].astype(np.float64)
        scores[f"{m}_abs"] = out["abs"].astype(np.float64)
        extra.setdefault(m, {})["n_obs_min"] = int(out["n_obs"].min())
    (CACHE / "scores").mkdir(exist_ok=True)
    np.savez_compressed(CACHE / "scores" / f"{tag}.npz", **scores, label=y, ia=ia, ib=ib)
    # pairs undefined in this matrix (genes sharing < 3 samples, e.g. genes absent from the library) are
    # excluded from scoring and reported; splits are still computed on the canonical scorable set so test pairs
    # stay identical to the main analysis (restricted to defined pairs).
    valid = np.all([np.isfinite(s) for s in scores.values()], axis=0)
    undefined = dict(pos=int((~valid & (y == 1)).sum()), neg=int((~valid & (y == 0)).sum()))
    full_y = y
    y = y[valid]
    scores = {k: s[valid] for k, s in scores.items()}

    res = dict(matrix=matrix, library=library, n_samples=int(X.shape[0]), n_genes=int(X.shape[1]),
               n_pairs=len(y), n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()), prevalence=float(y.mean()),
               excluded_unscorable=excl, excluded_undefined_in_matrix=undefined, runtime_s=runtime, method_details=extra,
               machine=dict(cpu_count=os.cpu_count()), n_boot=n_boot, n_boot_fold=n_boot_fold,
               bootstrap_seed=SEEDS[0], split_seed=SEEDS[0], n_folds=N_FOLDS)
    t0 = time.perf_counter()
    res["full"] = {k: metrics.evaluate(y, s, n_boot=n_boot, seed=SEEDS[0]) for k, s in scores.items()}
    ref = "pearson_abs"
    res["paired_diff_vs_pearson_abs"] = {k: metrics.bootstrap_auprc_diff(y, scores[k], scores[ref], n_boot, SEEDS[0])
                                         for k in scores if k != ref}
    res["paired_diff_gls_abs_vs_spearman_abs"] = metrics.bootstrap_auprc_diff(
        y, scores["gls_abs"], scores["spearman_abs"], n_boot, SEEDS[0])
    res["sign_check"] = {m: dict(frac_pos_with_positive_score=float((scores[f"{m}_signed"][y == 1] > 0).mean()),
                                 frac_neg_with_positive_score=float((scores[f"{m}_signed"][y == 0] > 0).mean()))
                         for m in methods}

    ref_file = CACHE / "scores" / "CRISPRGeneEffect_all.npz"
    if tag != "CRISPRGeneEffect_all" and ref_file.exists() and undefined["pos"] + undefined["neg"] > 0:
        z = np.load(ref_file)
        assert (z["ia"] == ia).all() and (z["ib"] == ib).all()
        res["CRISPRGeneEffect_all_on_same_pairs"] = {
            k: metrics.evaluate(y, z[k][valid], n_boot=n_boot, seed=SEEDS[0]) for k in scores if k in z.files}

    res["splits"] = {}
    for scheme in SCHEMES:
        sp = splits.make_split(sc, scheme, N_FOLDS, SEEDS[0])
        folds = []
        for k in range(N_FOLDS):
            te = sp.test_mask(k)[valid]
            folds.append({key: metrics.evaluate(y[te], s[te], n_boot=n_boot_fold, seed=SEEDS[0])
                          for key, s in scores.items()})
        pooled_mask = (sp.test_fold >= 0)[valid]
        res["splits"][scheme] = dict(
            fingerprint=sp.fingerprint(), stats=sp.stats(full_y), folds=folds,
            fold_mean_auprc={key: float(np.mean([f[key]["auprc"] for f in folds])) for key in scores},
            fold_sd_auprc={key: float(np.std([f[key]["auprc"] for f in folds], ddof=1)) for key in scores},
            pooled_test={key: metrics.evaluate(y[pooled_mask], s[pooled_mask]) for key, s in scores.items()},
            n_pairs_never_test=int((~pooled_mask).sum()))
    res["eval_runtime_s"] = round(time.perf_counter() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
    log.info("wrote %s", OUT / f"{tag}.json")


def _fmt_ci(d):
    return f"{d['auprc']:.4f} [{d['auprc_ci_low']:.4f}, {d['auprc_ci_high']:.4f}]"


def write_readme():
    lines = ["# Unsupervised co-essentiality baselines on HuMAP train pairs", "",
             "Generated by `scripts/02_run_baselines.py` (do not edit by hand). Scores computed on all *scorable* "
             "train pairs (both proteins mapped to distinct DepMap genes; see `data_mapping.json`). AUPRC = "
             "average precision (sklearn tie semantics); 95% CI = stratified percentile bootstrap (positives and "
             "negatives resampled separately), seed SEEDS[0]. `abs` = |r| for correlations and two-sided "
             "-log10 p for GLS; `signed` = r or t. Runtime = wall clock of scoring all pairs (GLS includes "
             "covariance estimation) on the machine in `machine`.", ""]
    for f in sorted(OUT.glob("*_*.json")):
        if f.name == "data_mapping.json":
            continue
        r = json.loads(f.read_text())
        lines += [f"## {r['matrix']} ({r['library'] or 'all samples'}): {r['n_samples']} samples, "
                  f"{r['n_pos']} pos / {r['n_neg']} neg (prevalence {r['prevalence']:.4f})", "",
                  f"Excluded unscorable pairs (unmapped protein): {r['excluded_unscorable']}; excluded pairs "
                  f"undefined in this matrix (<3 shared samples): {r.get('excluded_undefined_in_matrix')}", "",
                  "| score | AUPRC [95% CI] | AUROC | P@R0.1 | P@R0.25 | P@R0.5 | ΔAUPRC vs pearson_abs [95% CI] | runtime (s) |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, d in r["full"].items():
            m = k.rsplit("_", 1)[0]
            pdiff = r["paired_diff_vs_pearson_abs"].get(k)
            ds = f"{pdiff['diff']:+.4f} [{pdiff['ci_low']:+.4f}, {pdiff['ci_high']:+.4f}]" if pdiff else "ref"
            lines.append(f"| {k} | {_fmt_ci(d)} | {d['auroc']:.4f} | {d['P@R0.1']:.3f} | {d['P@R0.25']:.3f} | "
                         f"{d['P@R0.5']:.3f} | {ds} | {r['runtime_s'][m]} |")
        if "CRISPRGeneEffect_all_on_same_pairs" in r:
            lines += ["", "CRISPRGeneEffect (all models) on the same pairs, for comparison: " + ", ".join(
                f"{k} {_fmt_ci(d)}" for k, d in r["CRISPRGeneEffect_all_on_same_pairs"].items()
                if k.endswith("_abs") or k.startswith("gls_signed"))]
        lines += ["", "Per-split test-fold AUPRC, mean ± sd over 5 folds (pooled = all test pairs of the scheme):", "",
                  "| score | " + " | ".join(f"{s} mean±sd | {s} pooled" for s in SCHEMES) + " |",
                  "|---|" + "---|---|" * len(SCHEMES)]
        for k in r["full"]:
            cells = []
            for s in SCHEMES:
                sp = r["splits"][s]
                cells += [f"{sp['fold_mean_auprc'][k]:.4f} ± {sp['fold_sd_auprc'][k]:.4f}",
                          f"{sp['pooled_test'][k]['auprc']:.4f}"]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines += ["", "Fold sizes (test pos/neg; C2 pairs excluded from that fold):", ""]
        for s in SCHEMES:
            st = r["splits"][s]["stats"]
            lines.append(f"- {s} (fingerprint `{r['splits'][s]['fingerprint'][:12]}`): "
                         + "; ".join(f"f{d['fold']} {d['test_pos']}/{d['test_neg']}" for d in st)
                         + f"; never-test pairs {r['splits'][s]['n_pairs_never_test']}")
        lines.append("")
    (OUT / "README.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="CRISPRGeneEffect", choices=data.MATRICES)
    ap.add_argument("--library", default=None, help="restrict samples to one library (e.g. Avana)")
    ap.add_argument("--olfactory", action="store_true", help="also run GLS with Wainberg olfactory-PC correction")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-boot-fold", type=int, default=200)
    ap.add_argument("--readme-only", action="store_true")
    a = ap.parse_args()
    if not a.readme_only:
        run(a.matrix, a.library, a.olfactory, a.n_boot, a.n_boot_fold)
    write_readme()


if __name__ == "__main__":
    main()
