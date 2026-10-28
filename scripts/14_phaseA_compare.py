"""Phase A candidates vs baselines on DEV (DESIGN v3.1 §3.6-3.7 dev part, §3.9 controls, §3.5 runtime).

Candidates: best single layer of the screen (results/phaseA/screen.json) and the greedy A6 ensemble
(results/phaseA/ensemble.json). Baselines: best dev variants (results/baselines_v2/best_variants.json).

1. Best single layer on SEEDS[1..4] (run_dev_cv, logged) so that both candidates have 5 seed repeats.
2. §3.7 dev checks per seed: (i) folds won (SEEDS[0]) and seed means; (ii) paired cluster bootstrap (1,000; seed = the
   fold seed) of ΔHT-AP on pooled out-of-fold dev predictions (union of the 5 test folds, fold HT weights 1 / K);
   (iii) non-inferiority (>= -5% relative) on macro per-cluster AP and AP without the largest component (SEEDS[0]).
3. Runtime: fresh subprocess, end to end from the cached gene-effect matrix (npy; CSV parsing excluded as for the
   baselines): load, preprocessing, Σ / eigendecomposition, genome-wide rows / neighbour sets, scores for all 596,597
   scorable pairs (+ z-average). Same protocol timed for GLS signed and Pearson.
4. Controls (§3.9): olfactory-receptor genes (HGNC gene group) -- all genome-wide OR x OR pairs scored by the
   candidates, percentiles within the 200k-pair genome-wide null; chromosome proximity -- HT-AP on all dev pairs vs
   excluding same-chromosome / same-arm pairs (HGNC cytoband location).
Output: results/phaseA/compare.json, results/phaseA/COMPARE.md. IFT pairs are never loaded.

Usage: ./.env/bin/python scripts/14_phaseA_compare.py [--allow-dirty]
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
from src.config import CACHE, N_FOLDS, RESULTS, ROOT, SEEDS  # noqa: E402

OUT = RESULTS / "phaseA"
MANIFESTS = {"screen_v1": RESULTS / "layers" / "manifest.json",
             "screen_v2": RESULTS / "layers" / "screen_v2.manifest.json"}
BASE_DIR = RESULTS / "baselines_v2"
N_BOOT = 1000


def col_meta(cache, name):
    for c in json.loads(MANIFESTS[cache].read_text())["columns"]:
        if c["name"] == name:
            return dict(cache=cache, col=c["col"], name=name, config=c["config"], null_mean=c["null_mean"],
                        null_sd=c["null_sd"])
    raise KeyError(name)


def candidates():
    screen = json.loads((OUT / "screen.json").read_text())["layers"]
    best = max(screen, key=lambda r: r["ap_ht"])
    ens = json.loads((OUT / "ensemble.json").read_text())["ensembles"]["A6_greedy"]
    return {"best_layer": dict(members=[col_meta(best["cache"], best["name"])], zavg=False, label=best["name"]),
            "A6_ensemble": dict(members=[dict(m) for m in ens["members_meta"]], zavg=True, label=ens["name"])}


def cached_scores(cand, pid):
    cols = []
    for m in cand["members"]:
        M = np.load(CACHE / "layers" / f"{m['cache']}.scores.f32.npy", mmap_mode="r")
        v = np.asarray(M[pid, m["col"]], dtype=np.float64)
        cols.append((v - m["null_mean"]) / m["null_sd"] if cand["zavg"] else v)
    return np.mean(cols, 0)


# ------------------------------------------------------------------------------------------- runtime child
def compute_scores(cand, ia, ib, eng):
    from src import layers_ext
    cols, per = [], []
    for m in cand["members"]:
        t0 = time.perf_counter()
        v = layers_ext.compute(m["config"], ia, ib, eng)
        per.append(dict(name=m["name"], s=round(time.perf_counter() - t0, 2)))
        cols.append((v - m["null_mean"]) / m["null_sd"] if cand["zavg"] else v)
    return np.mean(cols, 0), per


def runtime_child(which):
    from src import baselines, data, layers
    P = np.load(CACHE / "layers" / "screen_v1.pairs.npz", allow_pickle=True)
    ia, ib = P["ia"], P["ib"]
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    X, _, _ = data.load_matrix("CRISPRGeneEffect")
    out = dict(which=which, n_pairs=int(len(ia)), load_matrix_s=round(time.perf_counter() - t0, 2))
    if which in ("gls_signed", "pearson_signed"):
        t0 = time.perf_counter()
        if which == "gls_signed":
            Sigma, _ = baselines.sample_covariance(X)
            out["sigma_s"] = round(time.perf_counter() - t0, 2)
            s = baselines.gls_pairs(X, ia, ib, Sigma=Sigma)["signed"]
        else:
            s = baselines.pearson_pairs(X, ia, ib)["signed"]
        out["score_s"] = round(time.perf_counter() - t0, 2)
    else:
        cand = candidates()[which]
        t0 = time.perf_counter()
        eng = layers.LayerEngine(X)
        out["engine_init_s"] = round(time.perf_counter() - t0, 2)
        s, per = compute_scores(cand, ia, ib, eng)
        out["members"] = per
        cached = cached_scores(cand, np.arange(len(ia)))
        out["max_abs_diff_vs_cache"] = float(np.max(np.abs(s - cached)))
    out["total_s"] = round(time.perf_counter() - t_all, 2)
    out["finite"] = bool(np.isfinite(s).all())
    print("RUNTIME_JSON " + json.dumps(out))


def runtime(which):
    r = subprocess.run([sys.executable, __file__, "--runtime-child", which], capture_output=True, text=True, cwd=ROOT)
    line = [l for l in r.stdout.splitlines() if l.startswith("RUNTIME_JSON ")]
    if r.returncode or not line:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(line[-1][len("RUNTIME_JSON "):])


# ------------------------------------------------------------------------------------------- main parts
def oof_bootstrap(u, S, seed):
    sp = evaluate.make_dev_split(u, "cluster", seed)
    oof = np.flatnonzero(sp.test_fold >= 0)
    ca, cb = u.col("cluster_a")[oof], u.col("cluster_b")[oof]
    w = metrics.ht_weights(ca, cb, 1.0 / N_FOLDS)
    cands = [k for k in S if not k.startswith("BASE_")]
    bases = [k for k in S if k.startswith("BASE_")]
    diffs = [(c, b) for c in cands for b in bases] + [("A6_ensemble", "best_layer")]
    b = metrics.cluster_bootstrap(u.y[oof], {k: v[oof] for k, v in S.items()}, ca, cb, n_boot=N_BOOT, seed=seed,
                                  diffs=diffs, base_w=w)
    b["n_oof_pairs"], b["n_oof_pos"] = int(len(oof)), int(u.y[oof].sum())
    b["split_fingerprint"] = sp.fingerprint()
    return b


def or_control(fns, eng, genes):
    """fns: name -> f(ia, ib) scores. OR x OR pairs vs the fixed genome-wide null pairs; split by same chromosome."""
    from src import data, layers
    ann = data.gene_annotation(genes)
    is_or = ann.gene_group.fillna("").str.contains("Olfactory receptors").values
    or_idx = np.flatnonzero(is_or)
    P = np.load(CACHE / "layers" / "screen_v1.pairs.npz", allow_pickle=True)
    labelled = np.unique(np.r_[P["ia"], P["ib"]])
    a, b = np.triu_indices(len(or_idx), 1)
    oa, ob = or_idx[a], or_idx[b]
    chrom = ann.chrom.values
    same_chr = np.array([x == y and x is not None and x == x for x, y in zip(chrom[oa], chrom[ob])])
    na, nb = layers.null_pairs(eng)
    out = dict(n_or_genes=int(len(or_idx)), n_or_in_background=int(eng.bg[or_idx].sum()),
               or_nan_frac_median=float(np.median(eng.nan_frac[or_idx])),
               n_or_genes_labelled=int(np.isin(or_idx, labelled).sum()), n_or_pairs=int(len(oa)),
               n_or_pairs_same_chrom=int(same_chr.sum()), by_scorer={})
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    for k, fn in fns.items():
        s_or, s_null = fn(oa, ob), fn(na, nb)
        ok = np.isfinite(s_null)
        srt = np.sort(s_null[ok])
        s_or = np.where(np.isfinite(s_or), s_or, -np.inf)
        pct = np.searchsorted(srt, s_or, side="right") / len(srt)
        th = np.quantile(srt, [0.99, 0.999])
        out["by_scorer"][k] = dict(
            or_percentile_quantiles={str(x): float(v) for x, v in zip(q, np.quantile(pct, q))},
            mean_percentile=float(pct.mean()), frac_above_null_p99=float((s_or > th[0]).mean()),
            frac_above_null_p999=float((s_or > th[1]).mean()),
            frac_above_null_p99_same_chrom=float((s_or[same_chr] > th[0]).mean()),
            frac_above_null_p99_diff_chrom=float((s_or[~same_chr] > th[0]).mean()),
            n_null_nonfinite=int((~ok).sum()))
        print(f"OR control {k}: median pct {np.median(pct):.3f}, >p99 {out['by_scorer'][k]['frac_above_null_p99']:.4f}",
              flush=True)
    return out


def chrom_control(u, S, genes):
    from src import data
    ann = data.gene_annotation(genes)
    chrom, arm = ann.chrom.values, ann.arm.values
    ia, ib = u.col("ia"), u.col("ib")
    known = np.array([x is not None and x == x for x in chrom[ia]]) & np.array([x is not None and x == x for x in chrom[ib]])
    same_chr = known & (chrom[ia] == chrom[ib])
    same_arm = same_chr & (arm[ia] == arm[ib]) & np.array([x is not None and x == x for x in arm[ia]])
    y = u.y.astype(bool)
    w = metrics.ht_weights(u.col("cluster_a"), u.col("cluster_b"), u.f)
    sp = evaluate.make_dev_split(u, "cluster", SEEDS[0])
    out = dict(frac_same_chrom_pos=float(same_chr[y].mean()), frac_same_chrom_neg=float(same_chr[~y].mean()),
               frac_same_arm_pos=float(same_arm[y].mean()), frac_same_arm_neg=float(same_arm[~y].mean()),
               n_unknown_chrom=int((~known).sum()), by_scorer={})
    for k, s in S.items():
        d = {}
        for lab, keep in (("all", np.ones(len(y), bool)), ("excl_same_chrom", ~same_chr), ("excl_same_arm", ~same_arm),
                          ("only_same_chrom", same_chr)):
            d[f"pooled_ap_ht_{lab}"] = metrics.weighted_ap(y[keep], s[keep], w[keep])
            fm = []
            for f in range(N_FOLDS):
                te = sp.test_mask(f) & keep
                fm.append(metrics.weighted_ap(y[te], s[te], evaluate.design_weights(u, np.flatnonzero(te), "cluster")))
            d[f"fold_mean_ap_ht_{lab}"] = float(np.nanmean(fm))
        out["by_scorer"][k] = d
    return out


def main(allow_dirty):
    subprocess.run(["sh", str(ROOT / "scripts" / "check_data.sh")], check=True, stdout=subprocess.DEVNULL)
    from src import layers
    from src.data import load_matrix
    from src.universe import load_dev_universe
    tid = uuid.uuid4().hex[:12]
    evaluate.log_trial("start", tid, tag="phaseA_compare_controls", universe="dev", seed=SEEDS,
                       scorer=dict(name="phaseA candidates vs baselines (bootstrap, controls)", label_free=True),
                       git=evaluate.git_state(), allow_dirty=allow_dirty, extra=dict(grid="phaseA_compare"))
    u = load_dev_universe()
    pid = u.col("pair_id")
    cands = candidates()
    S = {k: cached_scores(c, pid) for k, c in cands.items()}
    best = json.loads((BASE_DIR / "best_variants.json").read_text())["best_variants"]
    mat = scorers.real_matrix()
    for fam, v in best.items():
        sc = scorers.make_baseline(v, mat).bind(u)
        S[f"BASE_{v}"] = sc.score(np.arange(len(u)))

    # 1. best single layer, seeds 1..4 (seed 0 = screen result)
    bl = cands["best_layer"]["members"][0]
    seed_res = {"best_layer": {}, "A6_ensemble": {}}
    screen_row = {r["name"]: r for r in json.loads((OUT / "screen.json").read_text())["layers"]}[bl["name"]]
    seed_res["best_layer"][SEEDS[0]] = json.loads((ROOT / screen_row["result"]).read_text())
    arr_full = np.load(CACHE / "layers" / f"{bl['cache']}.scores.f32.npy", mmap_mode="r")[:, bl["col"]].astype(np.float64)
    for s in SEEDS[1:]:
        sc = scorers.PrecomputedScorer(bl["name"], arr_full, params=dict(cache=bl["cache"], col=bl["col"],
                                                                         config=bl["config"]))
        seed_res["best_layer"][s] = evaluate.run_dev_cv(sc, seed=s, universe=u, out_dir=OUT / "compare",
                                                        tag=f"best_layer_cluster_seed{s}", allow_dirty=allow_dirty,
                                                        extra=dict(grid="phaseA_best_layer_seeds", layer=bl["name"]))
    for s in SEEDS:
        seed_res["A6_ensemble"][s] = json.loads((OUT / "ensemble" / f"ens_A6_greedy_seed{s}.json").read_text())
    for v in best.values():
        seed_res[f"BASE_{v}"] = {s: json.loads((BASE_DIR / f"{v}_cluster_seed{s}.json").read_text()) for s in SEEDS}

    # 2. §3.7 dev checks
    crit = {}
    boots = {}
    for s in SEEDS:
        boots[s] = oof_bootstrap(u, S, s)
        print(f"bootstrap seed {s}: " + ", ".join(f"{k} {v['diff']:+.3f} [{v['ci_low']:+.3f},{v['ci_high']:+.3f}]"
                                                for k, v in boots[s]["diff"].items()), flush=True)
    for c in cands:
        crit[c] = {}
        for v in best.values():
            b = f"BASE_{v}"
            fc = np.array(seed_res[c][SEEDS[0]]["summary"]["ap_ht"]["folds"])
            fb = np.array(seed_res[b][SEEDS[0]]["summary"]["ap_ht"]["folds"])
            means = {s: (seed_res[c][s]["summary"]["ap_ht"]["mean"], seed_res[b][s]["summary"]["ap_ht"]["mean"]) for s in SEEDS}
            ci = {s: boots[s]["diff"][f"{c}-{b}"] for s in SEEDS}
            sm_c, sm_b = seed_res[c][SEEDS[0]]["summary"], seed_res[b][SEEDS[0]]["summary"]
            ni = {k: sm_c[k]["mean"] / sm_b[k]["mean"] - 1 for k in ("macro_cluster_ap", "ap_ht_no_largest_component")}
            crit[c][v] = dict(
                folds_won_seed0=int((fc > fb).sum()), fold_deltas_seed0=(fc - fb).tolist(),
                seed_means={s: dict(cand=m[0], base=m[1]) for s, m in means.items()},
                i_pass=bool((fc > fb).sum() >= 4 and all(m[0] > m[1] for m in means.values())),
                oof_delta={s: dict(diff=d["diff"], ci=[d["ci_low"], d["ci_high"]], rel=d["rel_gain"],
                                   rel_ci=[d["rel_ci_low"], d["rel_ci_high"]], p=d["p_two_sided"]) for s, d in ci.items()},
                ii_dev_pass=bool(all(d["ci_low"] > 0 for d in ci.values())),
                iii_rel={k: float(x) for k, x in ni.items()}, iii_pass=bool(all(x >= -0.05 for x in ni.values())))
    strata = {k: {m: seed_res[k][SEEDS[0]]["summary"][m]["mean"] for m in
                  ("ap_ht", "ap_ht_both_nonessential", "ap_ht_mixed", "ap_ht_both_essential", "macro_cluster_ap",
                   "ap_ht_no_largest_component", "ap_ht_hard_neg", "auroc_ht", "ht_P@R0.1", "ht_P@R0.25", "ht_P@R0.5")}
              for k in seed_res}

    # 3. runtime (fresh processes)
    rt = {k: runtime(k) for k in ("best_layer", "A6_ensemble", "gls_signed", "pearson_signed")}
    print("runtime", {k: v["total_s"] for k, v in rt.items()}, flush=True)

    # 4. controls
    X, _, genes = load_matrix("CRISPRGeneEffect")
    eng = layers.LayerEngine(X)
    from src import baselines
    Sig = baselines.sample_covariance(X)[0]
    fns = {k: (lambda ia, ib, c=c: compute_scores(c, ia, ib, eng)[0]) for k, c in cands.items()}
    fns["BASE_gls_signed"] = lambda ia, ib: baselines.gls_pairs(X, ia, ib, Sigma=Sig)["signed"]
    fns["BASE_pearson_signed"] = lambda ia, ib: baselines.pearson_pairs(X, ia, ib)["signed"]
    orc = or_control(fns, eng, genes)
    chc = chrom_control(u, S, genes)

    out = dict(candidates={k: dict(label=c["label"], members=[m["name"] for m in c["members"]], zavg=c["zavg"])
                           for k, c in cands.items()},
               baselines=best, criteria_dev=crit, bootstrap={s: dict(diff=b["diff"], ap=b["ap"], n_oof_pairs=b["n_oof_pairs"],
                                                                      n_oof_pos=b["n_oof_pos"], n_clusters=b["n_clusters"],
                                                                      split_fingerprint=b["split_fingerprint"])
                                                            for s, b in boots.items()},
               strata_seed0=strata, ap_ht_by_seed={k: {s: r[s]["summary"]["ap_ht"]["mean"] for s in SEEDS}
                                                   for k, r in seed_res.items()},
               runtime=rt, or_control=orc, chrom_control=chc,
               notes=["dev only; IFT untouched", "SEEDS[1..4] re-split the same dev pairs (split variance only)",
                      "candidates were selected on SEEDS[0] dev folds (screen + greedy), so dev deltas are optimistic"])
    (OUT / "compare.json").write_text(json.dumps(out, indent=1, default=str))
    evaluate.log_trial("end", tid, tag="phaseA_compare_controls", status="ok", result="results/phaseA/compare.json")
    write_md(out)


def write_md(d):
    f = lambda x: f"{x:.3f}"  # noqa: E731
    L = ["# Phase A candidates vs baselines (dev)", "", "Generated by `scripts/14_phaseA_compare.py`. DEV pairs only; IFT "
         "untouched. Candidates were selected on SEEDS[0] dev folds -> dev deltas are optimistic (selection bias); "
         "SEEDS[1..4] re-split the same pairs.", ""]
    for k, c in d["candidates"].items():
        L.append(f"- **{k}**: {c['label']}")
    L += ["", "## HT-AP (fold mean) per seed", "", "| scorer | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean |",
          "|---|" + "---|" * (len(SEEDS) + 1)]
    for k, v in d["ap_ht_by_seed"].items():
        xs = [v[s] for s in SEEDS]
        L.append(f"| {k} | " + " | ".join(f(x) for x in xs) + f" | {f(np.mean(xs))} |")
    L += ["", "## Paired cluster bootstrap ΔHT-AP, pooled OOF dev predictions (1,000 resamples, per seed)", "",
          "| comparison | " + " | ".join(f"seed {s}" for s in SEEDS) + " |", "|---|" + "---|" * len(SEEDS)]
    for key in d["bootstrap"][SEEDS[0]]["diff"]:
        L.append(f"| {key} | " + " | ".join(
            f"{b['diff'][key]['diff']:+.3f} [{b['diff'][key]['ci_low']:+.3f}, {b['diff'][key]['ci_high']:+.3f}] "
            f"({b['diff'][key]['rel_gain']:+.0%})" for b in (d["bootstrap"][s] for s in SEEDS)) + " |")
    L += ["", "## §3.7 dev-side checks (IFT gate not evaluated)", "",
          "| candidate | baseline | (i) folds won seed0 | (i) mean higher all seeds | (i) | (ii) CI>0 all seeds | "
          "(iii) Δrel macro-cluster / no-largest | (iii) |", "|---|---|---|---|---|---|---|---|"]
    for c, byb in d["criteria_dev"].items():
        for v, r in byb.items():
            allhi = all(m["cand"] > m["base"] for m in r["seed_means"].values())
            L.append(f"| {c} | {v} | {r['folds_won_seed0']}/5 | {allhi} | {r['i_pass']} | {r['ii_dev_pass']} | "
                     f"{r['iii_rel']['macro_cluster_ap']:+.1%} / {r['iii_rel']['ap_ht_no_largest_component']:+.1%} | "
                     f"{r['iii_pass']} |")
    L += ["", "## Strata (SEEDS[0], HT-AP fold mean)", "",
          "| scorer | HT-AP | both non-ess. | mixed | both ess. | macro-cluster | w/o largest | hard-neg | HT AUROC | P@R0.1 | P@R0.25 | P@R0.5 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in d["strata_seed0"].items():
        L.append(f"| {k} | " + " | ".join(f(m[x]) for x in ("ap_ht", "ap_ht_both_nonessential", "ap_ht_mixed",
                                                             "ap_ht_both_essential", "macro_cluster_ap",
                                                             "ap_ht_no_largest_component", "ap_ht_hard_neg", "auroc_ht",
                                                             "ht_P@R0.1", "ht_P@R0.25", "ht_P@R0.5")) + " |")
    L += ["", "## Runtime (fresh process, cached npy matrix -> scores for all 596,597 scorable pairs; 8-core laptop)", "",
          "| scorer | load matrix (s) | total (s) | detail |", "|---|---|---|---|"]
    for k, r in d["runtime"].items():
        det = ", ".join(f"{m['name']} {m['s']}s" for m in r.get("members", [])) or ", ".join(
            f"{x} {r[x]}s" for x in ("sigma_s", "score_s") if x in r)
        if "max_abs_diff_vs_cache" in r:
            det += f"; max |Δ| vs cache {r['max_abs_diff_vs_cache']:.1e}"
        L.append(f"| {k} | {r['load_matrix_s']} | {r['total_s']} | {det} |")
    o = d["or_control"]
    L += ["", "## Olfactory-receptor control (§3.9)", "",
          f"{o['n_or_genes']} OR genes (HGNC group 'Olfactory receptors'; {o['n_or_in_background']} in the <10%-NaN "
          f"background; median NaN fraction {o['or_nan_frac_median']:.2f}); {o['n_or_genes_labelled']} labelled; "
          f"{o['n_or_pairs']:,} OR×OR pairs. Percentile = position in the 200k random genome-wide null pairs.", "",
          f"{o['n_or_pairs_same_chrom']:,} OR×OR pairs are on the same chromosome (OR genomic clusters). Expected under "
          "background: percentiles uniform, 1% above null p99, 0.1% above p99.9.", "",
          "| scorer | OR-pair percentile q05 / q25 / q50 / q75 / q95 / q99 | mean pct | frac > null p99 | frac > null p99.9 | "
          "> p99 same chrom / diff chrom |", "|---|---|---|---|---|---|"]
    for k, r in o["by_scorer"].items():
        qq = r["or_percentile_quantiles"]
        L.append(f"| {k} | " + " / ".join(f"{qq[x]:.2f}" for x in ("0.05", "0.25", "0.5", "0.75", "0.95", "0.99"))
                 + f" | {r['mean_percentile']:.3f} | {r['frac_above_null_p99']:.4f} | {r['frac_above_null_p999']:.4f} | "
                 f"{r['frac_above_null_p99_same_chrom']:.4f} / {r['frac_above_null_p99_diff_chrom']:.4f} |")
    c = d["chrom_control"]
    L += ["", "## Chromosome-proximity control (HGNC cytoband; dev pairs)", "",
          f"Same-chromosome share: positives {c['frac_same_chrom_pos']:.3f}, negatives {c['frac_same_chrom_neg']:.3f}; "
          f"same arm: {c['frac_same_arm_pos']:.3f} / {c['frac_same_arm_neg']:.3f}.", "",
          "| scorer | pooled HT-AP all | excl. same chrom | excl. same arm | only same chrom | fold-mean all | fold-mean excl. same chrom |",
          "|---|---|---|---|---|---|---|"]
    for k, r in c["by_scorer"].items():
        L.append(f"| {k} | {f(r['pooled_ap_ht_all'])} | {f(r['pooled_ap_ht_excl_same_chrom'])} | "
                 f"{f(r['pooled_ap_ht_excl_same_arm'])} | {f(r['pooled_ap_ht_only_same_chrom'])} | "
                 f"{f(r['fold_mean_ap_ht_all'])} | {f(r['fold_mean_ap_ht_excl_same_chrom'])} |")
    (OUT / "COMPARE.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--runtime-child", default=None)
    ap.add_argument("--md-only", action="store_true")
    a = ap.parse_args()
    if a.runtime_child:
        runtime_child(a.runtime_child)
    elif a.md_only:
        write_md(json.loads((OUT / "compare.json").read_text(), object_hook=lambda d: {
            (int(k) if k.isdigit() else k): v for k, v in d.items()}))
    else:
        main(a.allow_dirty)
