#!/usr/bin/env python
"""Report tables T1-T7 -> results/report/tables.md.

Reads ONLY committed result files (JSON sidecars / summaries under results/); it never loads
datasets/ or data_cache/ and never recomputes a metric from data.  Anything that is not in
those files is printed as "n/a (not in results)".  The only arithmetic done here is
averaging / summing / differencing numbers that are already reported (marked "derived").

Run:  ./.env/bin/python scripts/40_report_tables.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
NA = "n/a (not in results)"
SEEDS = [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- helpers
_cache: dict[str, dict] = {}


def J(rel: str):
    """Load a committed JSON result file (path relative to results/)."""
    if rel not in _cache:
        with open(R / rel) as fh:
            _cache[rel] = json.load(fh)
    return _cache[rel]


def exists(rel: str) -> bool:
    return (R / rel).exists()


def f3(x, plus=False):
    if x is None:
        return NA
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:+.3f}" if plus else f"{x:.3f}"


def prev(x):
    """prevalences are ~0.005, so print 4 decimals (the only exception to 3-decimal rounding)."""
    return NA if x is None else f"{x:.4f}"


def sec(x):
    return NA if x is None else f"{x:.2f}"


def ci(d):
    """+0.155 [+0.065, +0.246] from a bootstrap diff record."""
    if d is None:
        return NA
    return f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def apci(d):
    if d is None:
        return NA
    return f"{d['ap']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"


def pm(m, s):
    return f"{m:.3f} ± {s:.3f}"


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def run_summary(rel):
    return J(rel)["summary"]


# --------------------------------------------------------------------------- T1
def t1():
    cs = J("splits/clusters_summary.json")
    pr = J("predict/union_single.run.json")["counts"]
    ift = J("ift/look01_2026-10-27.json")
    pw = J("splits/power_ift.json")
    dev_ref = J("baselines_v2/gls_signed_cluster_seed0.json")
    p = cs["pairs"]
    pos = sum(p[k]["pos"] for k in p)
    neg = sum(p[k]["neg"] for k in p)
    n = pos + neg
    st = cs["cluster_stats"]
    unsc = cs["unscorable_pairs"]
    un_pos = sum(unsc[k]["pos"] for k in unsc)
    un_neg = sum(unsc[k]["neg"] for k in unsc)
    rows = [
        ["scorable pairs (mapped proteins), total (derived: dev+IFT+cross)", f"{n:,}"],
        ["  cross-check: n_mapped from predict_pairs run record", f"{pr['n_mapped']:,}"],
        ["positives (derived)", f"{pos:,}"],
        ["negatives (derived)", f"{neg:,}"],
        ["prevalence (derived: pos / pairs)", prev(pos / n)],
        ["proteins mapped / clusters (components -> clusters)", f"{st['n_proteins']:,} / {st['n_clusters']} ({st['n_components']} components; giant component split into {cs['giant_component']['n_communities']} Louvain communities)"],
        ["max cluster size / singleton clusters", f"{st['max_cluster_size']} / {st['n_singleton_clusters']}"],
        ["dev universe: pairs / pos / neg / proteins / clusters",
         f"{dev_ref['counts']['n_pairs']:,} / {dev_ref['counts']['n_pos']:,} / {dev_ref['counts']['n_neg']:,} / {dev_ref['counts']['n_proteins']} / {dev_ref['counts']['n_clusters']}"],
        ["dev prevalence (natural)", prev(dev_ref["pooled"]["prevalence"])],
        ["dev HT prevalence (fold-weighted)", prev(dev_ref["pooled"]["ht_prevalence"])],
        ["IFT universe: pairs / pos / neg / proteins / clusters",
         f"{ift['counts']['ift_pos'] + ift['counts']['ift_neg']:,} / {ift['counts']['ift_pos']:,} / {ift['counts']['ift_neg']:,} / {cs['proteins']['ift']} / {ift['counts']['ift_clusters']}"],
        ["IFT prevalence (natural)", prev(ift["metrics"]["zavg_BEF"]["prevalence"])],
        ["IFT HT prevalence", prev(ift["metrics"]["zavg_BEF"]["ht_prevalence"])],
        ["IFT fraction of positives / protein fraction f_IFT", f"{f3(cs['ift_frac_of_positives'])} / {f3(cs['protein_fraction_ift'])}"],
        ["cross (dev x IFT protein) pairs, in neither universe: pos / neg", f"{p['cross']['pos']:,} / {p['cross']['neg']:,}"],
        ["IFT Kish n_eff (clusters) / dev n_eff", f"{f3(cs['ift_draw']['kish_neff_ift'])} / {f3(cs['ift_draw']['kish_neff_dev'])}"],
        ["IFT largest cluster share of positives", f3(cs["ift_max_cluster_share_of_positives"])],
        ["unmapped proteins (dev / IFT)", f"{cs['unmapped_proteins']['dev']} / {cs['unmapped_proteins']['ift']}"],
        ["unscorable (unmapped) pairs dev: pos / neg", f"{unsc['dev']['pos']} / {unsc['dev']['neg']:,}"],
        ["unscorable (unmapped) pairs IFT: pos / neg", f"{unsc['ift']['pos']} / {unsc['ift']['neg']:,}"],
        ["unscorable (unmapped) pairs cross: pos / neg", f"{unsc['cross']['pos']} / {unsc['cross']['neg']:,}"],
        ["unscorable pairs total (derived) pos / neg", f"{un_pos} / {un_neg:,}"],
        ["unmapped pairs in predict_pairs run (all train pairs)", f"{pr['n_unmapped']:,} of {pr['n_pairs']:,}"],
        ["IFT power check (dev only): P(pass) at +0.20 true gain, rho=0.9 / rho=0.483",
         f"{f3(pw['settings']['rho=0.900']['power']['0.2']['p_pass'])} / {f3(pw['settings']['rho=0.483']['power']['0.2']['p_pass'])}"],
    ]
    src = ("Source: results/splits/clusters_summary.json (pairs, clusters, unmapped), results/splits/power_ift.json, "
           "results/baselines_v2/gls_signed_cluster_seed0.json (dev counts, prevalence), results/ift/look01_2026-10-27.json "
           "(IFT counts), results/predict/union_single.run.json (mapped/unmapped totals).")
    return "## T1. Data / split summary\n\n" + src + "\n\n" + table(["item", "value"], rows)


# --------------------------------------------------------------------------- T2
BASE_VARIANTS = [
    "pearson_signed", "pearson_abs", "spearman_signed", "spearman_abs",
    "gls_signed", "gls_abs", "gls_or_signed", "gls_or_abs",
    "gls_impute_signed", "gls_impute_abs", "gls_impute_or_signed", "gls_impute_or_abs",
]


def t2():
    best = J("baselines_v2/best_variants.json")["best_variants"]
    best_set = set(best.values())
    rows = []
    for v in BASE_VARIANTS:
        rel = f"baselines_v2/{v}_cluster_seed0.json"
        if not exists(rel):
            rows.append([v, NA, NA, NA, NA, NA])
            continue
        d = J(rel)
        s = d["summary"]["ap_ht"]
        seeds = []
        for sd in SEEDS:
            r2 = f"baselines_v2/{v}_cluster_seed{sd}.json"
            seeds.append(J(r2)["summary"]["ap_ht"]["mean"] if exists(r2) else None)
        five = mean(seeds) if all(x is not None for x in seeds) else None
        rt = d["runtime_s"]
        runtime = rt["setup"] + rt["score_all_dev"]
        rows.append([
            v + (" **(best)**" if v in best_set else ""),
            pm(s["mean"], s["sd"]),
            f3(five) if five is not None else NA,
            f3(d["pooled"]["ap_raw"]),
            f3(d["pooled"]["ap_ht"]),
            sec(runtime),
        ])
    t = table(["variant", "HT-AP mean ± sd over folds (seed 0)", "5-seed mean HT-AP", "pooled natural AP (raw, all dev pairs)",
               "pooled HT-AP (seed 0)", "runtime s (setup + score all dev pairs)"], rows)
    # secondary splits for the best three
    rows2 = []
    for v in sorted(best_set):
        r = [v]
        for sch in ["cluster", "component", "gene", "random"]:
            rel = f"baselines_v2/{v}_{sch}_seed0.json"
            r.append(pm(J(rel)["summary"]["ap_ht"]["mean"], J(rel)["summary"]["ap_ht"]["sd"]) if exists(rel) else NA)
        rows2.append(r)
    t_b = table(["best variant", "cluster (primary)", "strict component", "protein-disjoint (gene)", "random pair (leaky)"], rows2)
    src = ("Source: results/baselines_v2/<variant>_cluster_seed{0..4}.json (5-seed mean only exists for the three selected variants; "
           "others are n/a), results/baselines_v2/best_variants.json. Runtime = wall clock on the 8-core laptop for dev pairs only.")
    return ("## T2. Dev-CV baselines (cluster-grouped 5-fold CV)\n\n" + src + "\n\n" + t +
            "\n\nSecondary split schemes (seed 0, HT-AP fold mean ± sd). Source: results/baselines_v2/<variant>_{cluster,component,gene,random}_seed0.json.\n\n" + t_b)


# --------------------------------------------------------------------------- T3
def v2_5seed(tag):
    """5-seed means from phaseB_v2/runs + compare.json for tag like 'ref_B'."""
    ftag = tag.replace("+", "-")  # file names use '-' for '+'
    sm = []
    for sd in SEEDS:
        rel = f"phaseB_v2/runs/{ftag}_cluster_seed{sd}.json"
        sm.append(run_summary(rel) if exists(rel) else None)
    ok = [s for s in sm if s]
    cmp_ = J("phaseB_v2/compare.json")["seeds"]
    nt10 = [cmp_[str(sd)]["no_top10"]["ap"].get(tag, {}).get("ap") for sd in SEEDS]
    nt10_f = mean_or_none(nt10)
    return {
        "n": len(ok),
        "ht": mean_or_none([s["ap_ht"]["mean"] for s in ok]),
        "macro": mean_or_none([s["macro_cluster_ap"]["mean"] for s in ok]),
        "nt10": nt10_f,
        "fitscore": mean_or_none([mean(J(f"phaseB_v2/runs/{ftag}_cluster_seed{sd}.json")["runtime_s"]["fit_score_per_fold"]) for sd in SEEDS if exists(f"phaseB_v2/runs/{ftag}_cluster_seed{sd}.json")]),
    }


def phasec_5seed(tag):
    sm = []
    for sd in SEEDS:
        rel = f"phaseC/{tag}_cluster_seed{sd}.json"
        sm.append(J(rel) if exists(rel) else None)
    ok = [s for s in sm if s]
    cmp_ = J("phaseC/compare.json")["seeds"]
    nt = [cmp_[str(sd)]["no_top10"]["ap"].get(tag, {}).get("ap") for sd in SEEDS]
    return {
        "n": len(ok),
        "ht": mean_or_none([s["summary"]["ap_ht"]["mean"] for s in ok]),
        "macro": mean_or_none([s["summary"]["macro_cluster_ap"]["mean"] for s in ok]),
        "nt10": mean_or_none(nt),
        "fitscore": mean_or_none([mean(s["runtime_s"]["fit_score_per_fold"]) for s in ok]),
    }


def t3():
    ac = J("phaseA/compare.json")["runtime"]
    gr = J("phaseA_graph/runtime.json")
    rb = J("phaseB_v2/runtime.json")
    pz = J("predict/union_zavg_B_E_F.run.json")["runtime"]["scoring"]["components_s"]
    # best stackers by 5-seed HT-AP
    lr = max(["lr_L7", "lr_L7+genebin"], key=lambda t: v2_5seed(t)["ht"])
    lg = max(["lgbm_L7", "lgbm_L7+genebin"], key=lambda t: v2_5seed(t)["ht"])
    # best Phase C = best 5-seed learned config (only L1line has 5 seeds)
    pc_tags = sorted({re.sub(r"_cluster_seed\d\.json$", "", p.name) for p in (R / "phaseC").glob("L*_cluster_seed*.json")})
    pc_5 = {t: phasec_5seed(t) for t in pc_tags if phasec_5seed(t)["n"] == 5}
    pc_best = max(pc_5, key=lambda t: pc_5[t]["ht"])
    feat7 = rb["feat7"]["total_s"]
    heat = gr["rowz[cos_heat1[knn_union_lin_q0.25[spec_k20_t0.1_std]]]"]["total_s"]
    lap = gr["emb_lap512[knn_union_lin_q0.25[spec_k20_t0.1_std]]"]["total_s"]
    spec = [
        ("Pearson (signed)", "ref_PE", sec(ac["pearson_signed"]["total_s"]), "phaseA/compare.json"),
        ("Spearman (signed)", "ref_SP", NA, ""),
        ("GLS (signed)", "ref_G", sec(ac["gls_signed"]["total_s"]), "phaseA/compare.json"),
        ("F = spec_k20_t0.1_std (first-order, std + whitened)", "ref_F", f"{sec(pz['F'])} (component, incl. matrix prep)", "predict/union_zavg_B_E_F.run.json"),
        ("B = rowz[socos_q0.25] (second-order, row-z)", "ref_B", sec(rb["ref_B"]["total_s"]), "phaseB_v2/runtime.json"),
        ("E = emb_lap512 (Laplacian eigenmap)", "ref_E", sec(lap), "phaseA_graph/runtime.json"),
        ("H = rowz[cos_heat1] (diffusion)", "ref_H", sec(heat), "phaseA_graph/runtime.json"),
        ("zavg_BEF (final)", "zavg_BEF", sec(rb["zavg_BEF"]["total_s"]), "phaseB_v2/runtime.json"),
    ]
    rows = []
    for name, tag, rt, _ in spec:
        m = v2_5seed(tag)
        rows.append([name, f"{m['n']}-seed mean" if m["n"] == 5 else "seed 0", f3(m["ht"]), f3(m["macro"]), f3(m["nt10"]), rt])
    for lbl, tag in [(f"best LR stacker ({lr})", lr), (f"best LGBM stacker ({lg})", lg)]:
        m = v2_5seed(tag)
        rows.append([lbl, f"{m['n']}-seed mean" if m["n"] == 5 else "seed 0", f3(m["ht"]), f3(m["macro"]), f3(m["nt10"]),
                     f"{sec(feat7)} (7 input layers) + {sec(m['fitscore'])} fit+score per fold"])
    m = pc_5[pc_best]
    rows.append([f"best Phase C ({pc_best}, learned metric)", "5-seed mean", f3(m["ht"]), f3(m["macro"]), f3(m["nt10"]),
                 f"{sec(m['fitscore'])} fit+score per fold (CV record)"])
    t = table(["method", "HT-AP basis", "HT-AP (dev cluster CV)", "macro-cluster AP (fold mean)",
               "AP w/o top-10 clusters (pooled OOF, HT)", "runtime s (label-free inputs, all 596,597 scorable pairs, 8-core laptop)"], rows)
    src = ("Source: results/phaseB_v2/runs/<tag>_cluster_seed{0..4}.json (HT-AP, macro: mean over seeds of fold means), "
           "results/phaseB_v2/compare.json (no-top-10 pooled-OOF HT-AP, mean over seeds), results/phaseC/L*_cluster_seed*.json and "
           "results/phaseC/compare.json (Phase C), runtimes from results/phaseA/compare.json, results/phaseA_graph/runtime.json, "
           "results/phaseB_v2/runtime.json, results/predict/union_zavg_B_E_F.run.json. Spearman all-pairs runtime is not in the results "
           "(dev-only 0.6 s is in T2).")
    return "## T3. Method progression on dev\n\n" + src + "\n\n" + t


# --------------------------------------------------------------------------- T4
def t4():
    cmp_ = J("phaseB_v2/compare.json")["seeds"]
    pairs = [
        ("zavg_BEF − GLS signed", "zavg_BEF-ref_G"),
        ("zavg_BEF − Spearman signed", "zavg_BEF-ref_SP"),
        ("zavg_BEF − Pearson signed", "zavg_BEF-ref_PE"),
        ("B − GLS signed", "ref_B-ref_G"),
        ("B − Spearman signed", "ref_B-ref_SP"),
        ("B − Pearson signed", "ref_B-ref_PE"),
        ("zavg_BEF − B", "zavg_BEF-ref_B"),
    ]
    rows = []
    for lbl, key in pairs:
        r = [lbl]
        pos = 0
        for sd in SEEDS:
            d = cmp_[str(sd)]["all"]["diff"].get(key)
            r.append(ci(d))
            pos += int(d is not None and d["ci_low"] > 0)
        r.append(f"{pos}/5")
        rows.append(r)
    t = table(["comparison (ΔHT-AP, pooled OOF)"] + [f"seed {s}" for s in SEEDS] + ["seeds with CI > 0"], rows)
    # no-top10: only zavg_BEF-B has a CI in the results
    nrows = []
    key = "zavg_BEF-ref_B"
    nrows.append(["zavg_BEF − B (pairs not touching top-10 clusters)"] + [ci(cmp_[str(sd)]["no_top10"]["diff"].get(key)) for sd in SEEDS])
    for tag, lbl in [("zavg_BEF", "zavg_BEF"), ("ref_B", "B"), ("ref_G", "GLS"), ("ref_SP", "Spearman"), ("ref_PE", "Pearson")]:
        nrows.append([f"pooled-OOF HT-AP without top-10 clusters: {lbl}"] +
                     [apci(cmp_[str(sd)]["no_top10"]["ap"].get(tag)) for sd in SEEDS])
    t2_ = table(["quantity"] + [f"seed {s}" for s in SEEDS], nrows)
    src = ("Source: results/phaseB_v2/compare.json (paired cluster bootstrap on pooled out-of-fold predictions of the 5 CV test folds, 1,000 resamples "
           "over 190 dev clusters, 95% CI, per seed).")
    return ("## T4. Δ vs baselines on dev (pooled-OOF cluster bootstrap CI per seed)\n\n" + src + "\n\n" + t +
            "\n\nCaveat check (pairs not touching the 10 clusters with the most OOF positives). CIs for Δ vs the baselines on this subset are not in the results; "
            "only absolute pooled APs and Δ vs B are.\n\n" + t2_)


# --------------------------------------------------------------------------- T5
def t5():
    d = J("ift/look01_2026-10-27.json")
    m = d["metrics"]
    order = [("zavg_BEF", "zavg_BEF (primary)"), ("B", "B (secondary)"), ("gls_signed", "GLS signed"),
             ("spearman_signed", "Spearman signed"), ("pearson_signed", "Pearson signed")]
    hdr = ["method", "HT-AP [95% CI]", "raw AP", "AUROC (HT)", "macro-cluster AP", "AP w/o largest component",
           "AP incl. unmapped (min-score)", "P@R0.1", "P@R0.25", "P@R0.5", "non-ess / mixed / both-ess strata AP"]
    rows = []
    for k, lbl in order:
        x = m[k]
        rows.append([lbl, apci(d["bootstrap"]["ap"][k]), f3(x["ap_raw"]), f3(x["auroc_ht"]), f3(x["macro_cluster_ap"]),
                     f3(x["ap_ht_no_largest_component"]), f3(x["ap_ht_with_unmapped_min_score"]),
                     f3(x["ht_P@R0.1"]), f3(x["ht_P@R0.25"]), f3(x["ht_P@R0.5"]),
                     f"{f3(x['ap_ht_both_nonessential'])} / {f3(x['ap_ht_mixed'])} / {f3(x['ap_ht_both_essential'])}"])
    t = table(hdr, rows)
    x = m["zavg_BEF"]
    cnt = (f"IFT: {x['n_pos']} pos / {x['n_neg']:,} neg, {d['counts']['ift_clusters']} clusters ({x['n_clusters_macro']} with positives for macro AP); "
           f"unmapped pairs added for 'incl. unmapped': {x['n_unmapped_pos']} pos / {x['n_unmapped_neg']:,} neg; "
           f"stratum sizes (pos): non-ess {x['n_pos_both_nonessential']}, mixed {x['n_pos_mixed']}, both-ess {x['n_pos_both_essential']}; "
           f"no-largest-component positives {x['n_pos_no_largest_component']}; IFT HT prevalence {prev(x['ht_prevalence'])}.")
    # deltas
    rows2 = []
    for key, lbl in [("zavg_BEF-gls_signed", "zavg_BEF − GLS"), ("zavg_BEF-spearman_signed", "zavg_BEF − Spearman"),
                     ("zavg_BEF-pearson_signed", "zavg_BEF − Pearson"), ("B-gls_signed", "B − GLS"),
                     ("B-spearman_signed", "B − Spearman"), ("B-pearson_signed", "B − Pearson"), ("B-zavg_BEF", "B − zavg_BEF")]:
        b = d["bootstrap"]["diff"][key]
        vp = d["verdict"]["pairs"].get(key)
        if vp:
            ni = vp["non_inferiority"]
            gate = "pass" if vp["passes"] else "FAIL"
            ni_s = (f"macro {f3(ni['macro_cluster_ap']['candidate'])} vs {f3(ni['macro_cluster_ap']['baseline'])} "
                    f"({'ok' if ni['macro_cluster_ap']['passed'] else 'FAIL'}); "
                    f"no-largest {f3(ni['ap_ht_no_largest_component']['candidate'])} vs {f3(ni['ap_ht_no_largest_component']['baseline'])} "
                    f"({'ok' if ni['ap_ht_no_largest_component']['passed'] else 'FAIL'})")
        else:
            gate, ni_s = "info only", "—"
        rows2.append([lbl, ci(b), f"{b['rel_gain'] * 100:+.1f}% [{b['rel_ci_low'] * 100:+.1f}, {b['rel_ci_high'] * 100:+.1f}]",
                      f3(b["p_two_sided"]), gate, ni_s])
    t2_ = table(["Δ HT-AP", "Δ [95% cluster-bootstrap CI]", "relative gain [CI] (%)", "p (two-sided)", "v3.1 gate (>= +10% point est. and non-inferiority)",
                 "non-inferiority detail (>= 0.95 x baseline)"], rows2)
    v = d["verdict"]["candidates"]
    vrows = [[k, x2["order"], x2["role"], x2["status"],
              ", ".join(f"{b}: {'pass' if ok else 'fail'}" for b, ok in x2["passes_vs"].items()),
              x2["clearly_beats_ift"], x2["confirmed"]] for k, x2 in v.items()]
    t3_ = table(["candidate", "order", "role", "status", "gate per baseline", "clearly beats (IFT)", "confirmed"], vrows)
    verdict = (f"Verdict (fixed-sequence, D3/v3.1 §3.7): primary zavg_BEF CONFIRMED (clears all three baselines); "
               f"secondary B not confirmed (fails GLS macro non-inferiority: {f3(m['B']['macro_cluster_ap'])} vs {f3(m['gls_signed']['macro_cluster_ap'])}). "
               f"Rule: {d['verdict']['rule']}")
    src = "Source: results/ift/look01_2026-10-27.json (single pre-registered IFT look #1; metrics, bootstrap, verdict)."
    return ("## T5. IFT look #1\n\n" + src + "\n\n" + t + "\n\n" + cnt + "\n\n" + t2_ + "\n\n" + t3_ + "\n\n" + verdict)


# --------------------------------------------------------------------------- T6
def t6():
    cmp_v2 = J("phaseB_v2/compare.json")["seeds"]
    pc = J("phaseC/compare.json")["seeds"]
    rows = []

    # Phase C
    d = pc["0"]["all"]["diff"]
    p5 = phasec_5seed("L1line_rowzso")
    r5 = v2_5seed("ref_B")
    rows.append(["Phase C: L1 cell-line-weight metric, second-order (rowzso)", "supervised metric learning",
                 f"5-seed HT-AP {f3(p5['ht'])} vs init {f3(r5['ht'])}; seed-0 pooled-OOF Δ vs init {ci(d['L1line_rowzso-ref_best'])}",
                 "phaseC/L1line_rowzso_cluster_seed{0..4}.json, phaseC/compare.json"])
    rows.append(["Phase C: L2 low-rank d=64 InfoNCE, second-order", "supervised metric learning",
                 f"seed-0 pooled-OOF Δ vs init {ci(d['L2d64nce_rowzso-ref_best'])}", "phaseC/compare.json"])
    c = J("phaseC/control_shuffle_train_L2d64_rowzso_cluster_seed0.json")
    rows.append(["Phase C: training-label shuffle control (L2 d=64, rowzso)", "negative control",
                 f"HT-AP {f3(c['mean_ap_ht'])} vs unshuffled init {f3(J('phaseC/identity_rowzso_cluster_seed0.json')['summary']['ap_ht']['mean'])} (shuffled labels learn nothing; stays at init)",
                 "phaseC/control_shuffle_train_L2d64_rowzso_cluster_seed0.json"])

    # supervised stacking
    for tag, lbl in [("lgbm_L7", "LightGBM stacker (L7 layers)"), ("lr_L7", "logistic-regression stacker (L7 layers)")]:
        m = v2_5seed(tag)
        dd = [cmp_v2[str(sd)]["all"]["diff"][f"{tag}-zavg_BEF"] for sd in SEEDS]
        rows.append([f"Phase B v2: {lbl}", "supervised stacking",
                     f"5-seed HT-AP {f3(m['ht'])} vs zavg_BEF {f3(v2_5seed('zavg_BEF')['ht'])}; Δ vs zavg_BEF seed 0 {ci(dd[0])} (CI incl. 0 in {sum(1 for x in dd if x['ci_low'] <= 0 <= x['ci_high'])}/5 seeds)",
                     "phaseB_v2/runs/*.json, phaseB_v2/compare.json"])
    b = J("phaseB/bootstrap_cluster_seed0.json")["diff"]["lgbm_prov12-ref_zmean_prov12"]
    rows.append(["Phase B v1: LightGBM over 12 a-priori layers vs label-free z-mean of same 12", "supervised stacking",
                 f"Δ {ci(b)} (HT-AP {f3(J('phaseB/lgbm_prov12_cluster_seed0.json')['summary']['ap_ht']['mean'])} vs {f3(J('phaseB/ref_zmean_prov12_cluster_seed0.json')['summary']['ap_ht']['mean'])})",
                 "phaseB/bootstrap_cluster_seed0.json"])
    # shuffle controls
    sh = J("phaseB/control_shuffle_train_lgbm_base3_cluster_seed0.json")
    shn = J("phaseB/control_shuffle_rep_train_lgbmnm_base3_cluster.json")
    rows.append(["Phase B v1: all-label shuffle, monotone-constrained LGBM", "negative control",
                 f"HT-AP {f3(sh['mean_ap_ht'])} vs HT prevalence {prev(sh['mean_prevalence'])} (monotone constraint encodes 'similar = likely'; recorded finding)",
                 "phaseB/control_shuffle_train_lgbm_base3_cluster_seed0.json"])
    rows.append(["Phase B v1: all-label shuffle, unconstrained LGBM", "negative control",
                 f"mean AP {f3(shn['mean_ap_ht'])} vs HT prevalence {prev(shn['mean_prevalence'])} (5 permutations x 5 folds)",
                 "phaseB/control_shuffle_rep_train_lgbmnm_base3_cluster.json"])
    shh = J("phaseB_v2/controls/control_shuffle_held_lgbm_L7_cluster_seed0.json")
    rows.append(["Phase B v2: held-fold label shuffle (LGBM L7)", "negative control",
                 f"HT-AP {f3(shh['mean_ap_ht'])} vs HT prevalence {prev(shh['mean_prevalence'])}",
                 "phaseB_v2/controls/control_shuffle_held_lgbm_L7_cluster_seed0.json"])

    # gene features
    rows.append(["Gene-level statistics only (LGBM, binned)", "gene features",
                 f"HT-AP {f3(J('phaseB/lgbm_genebin_cluster_seed0.json')['summary']['ap_ht']['mean'])} (HT prevalence ≈ 0.009; no protein memorisation on cluster-disjoint folds)",
                 "phaseB/lgbm_genebin_cluster_seed0.json"])
    rows.append(["Phase B v2: add gene-bin features to LGBM L7", "gene features",
                 f"5-seed HT-AP {f3(v2_5seed('lgbm_L7+genebin')['ht'])} vs without {f3(v2_5seed('lgbm_L7')['ht'])}",
                 "phaseB_v2/runs/lgbm_L7*_cluster_seed*.json"])

    # ensembles
    ens = J("phaseA/ensemble.json")
    held = mean(n_["held_out_ap_ht"] for n_ in ens["nested"])
    single = mean(n_["best_single_inner_held"] for n_ in ens["nested"])
    rows.append(["Phase A: greedy label-free z-average ensemble, nested-by-fold check", "ensembles",
                 f"held-out HT-AP {f3(held)} vs best single layer {f3(single)} (seed 0; selection-biased dev gain disappears)",
                 "phaseA/ensemble.json"])
    rows.append(["Phase B v2: zavg_BEHF (adds H) vs zavg_BEF", "ensembles",
                 f"5-seed HT-AP {f3(v2_5seed('zavg_BEHF')['ht'])} vs {f3(v2_5seed('zavg_BEF')['ht'])}",
                 "phaseB_v2/runs/zavg_BEHF*.json, zavg_BEF*.json"])
    nest = J("phaseB_v2/nested.json")
    rows.append(["Phase B v2: nested-by-fold selection among 4 z-averages (mean over seeds)", "ensembles",
                 f"held-out HT-AP {f3(mean(nest[str(sd)]['zavg']['nested_mean_held_out'] for sd in SEEDS))}",
                 "phaseB_v2/nested.json"])

    # graph variants
    g = J("phaseA_graph/compare.json")
    key = "emb_lap512[knn_union_lin_q0.25[spec_k20_t0.1_std]]-rowz[socos_q0.25[spec_k20_t0.1_std]]"
    dl = [g[str(sd)]["diff"][key] for sd in SEEDS]
    rows.append(["A7: Laplacian-eigenmap embedding E replaces B (pooled-OOF Δ, seed 0 / 5-seed range)", "graph variants",
                 f"{ci(dl[0])}; Δ range {min(x['diff'] for x in dl):+.3f} to {max(x['diff'] for x in dl):+.3f}; CI incl. 0 in all {sum(1 for x in dl if x['ci_low'] <= 0 <= x['ci_high'])}/5 seeds",
                 "phaseA_graph/compare.json"])
    rows.append(["A7: nested-by-fold selection over 67 graph layers", "graph variants",
                 f"held-out HT-AP {f3(J('phaseA_graph/nested.json')['all']['nested_mean_held_out'])} vs current best {f3(J('phaseA_graph/nested.json')['current_best_fold_mean'])} (seed 0)",
                 "phaseA_graph/nested.json"])
    layers = J("phaseA_graph/screen.json")["layers"]
    tom = max((l for l in layers if "tom_adj" in l["name"]), key=lambda l: l["ap_ht"])
    rows.append(["A7: best TOM/WGCNA layer (any weighting)", "graph variants", f"HT-AP {f3(tom['ap_ht'])} vs B {f3(J('phaseA_graph/nested.json')['current_best_fold_mean'])} (seed 0)",
                 "phaseA_graph/screen.json"])

    # controls
    comp = J("phaseA/compare.json")
    orc = comp["or_control"]["by_scorer"]
    rows.append(["Olfactory-receptor control (B; 53,956 OR×OR pairs, expect 1%)", "OR control (partial fail)",
                 f"{orc['best_layer']['frac_above_null_p99'] * 100:.1f}% above null p99 (GLS {orc['BASE_gls_signed']['frac_above_null_p99'] * 100:.1f}%); "
                 f"same-chromosome pairs {orc['best_layer']['frac_above_null_p99_same_chrom'] * 100:.1f}% vs different {orc['best_layer']['frac_above_null_p99_diff_chrom'] * 100:.1f}%",
                 "phaseA/compare.json"])
    cc = comp["chrom_control"]["by_scorer"]["best_layer"]
    rows.append(["Chromosome-proximity control (B)", "chromosome control (passes)",
                 f"pooled HT-AP {f3(cc['pooled_ap_ht_all'])} all pairs vs {f3(cc['pooled_ap_ht_excl_same_chrom'])} excluding same-chromosome pairs (same-chromosome-only {f3(cc['pooled_ap_ht_only_same_chrom'])})",
                 "phaseA/compare.json"])
    rows.append(["Chromosome-only LightGBM", "chromosome control (passes)",
                 f"HT-AP {f3(J('phaseB/lgbm_chrom_cluster_seed0.json')['summary']['ap_ht']['mean'])} (≈ prevalence)", "phaseB/lgbm_chrom_cluster_seed0.json"])
    vb = J("verify/phaseA_best.json")["negative_control_gene_shuffle"]
    rows.append(["Gene-label shuffle control (independent re-implementation of B)", "negative control",
                 f"HT-AP {f3(vb['ht_ap']['mean'])} vs HT prevalence {prev(vb['ht_prevalence_mean'])}", "verify/phaseA_best.json"])
    vz = J("verify/zavg_BEF.json")["negative_control_gene_shuffle"]
    vs_ = sorted(vz)
    rows.append([f"Gene-label shuffle control for zavg_BEF ({len(vs_)} seeds)", "negative control",
                 f"HT-AP {', '.join(f3(vz[k]['ht_ap']) for k in vs_)} vs prevalence {prev(vz[vs_[0]]['ht_prevalence'])}", "verify/zavg_BEF.json"])
    t = table(["item", "category", "one number", "file"], rows)
    return ("## T6. Negative results and ablations\n\nSource: files named in the last column (all under results/); every row is a committed number, "
            "no recomputation.\n\n" + t)


# --------------------------------------------------------------------------- T7
def t7():
    items = [
        ("`single` (B = rowz[socos_q0.25])", "predict/union_single.run.json"),
        ("`zavg:B,E`", "predict/union_zavg_B_E.run.json"),
        ("`zavg:B,E,F` (final method)", "predict/union_zavg_B_E_F.run.json"),
        ("`gls`", "predict/union_gls.run.json"),
        ("`gls --components` (also computes GLS null)", "predict/union_gls_components.run.json"),
    ]
    rows = []
    for lbl, rel in items:
        r = J(rel)["runtime"]
        comps = ", ".join(f"{k} {v:.2f}" for k, v in r["scoring"]["components_s"].items())
        rows.append([lbl, sec(r["total_s"]), sec(r["matrix_load_s"]), comps, f3(r["peak_rss_gb"])])
    rows.append(["`pearson`, `spearman`, `E`, `F`, `H` alone (predict_pairs)", NA, NA, NA, NA])
    t = table(["method", "total s (raw CSV -> scores)", "matrix load + hash s", "scoring by component s", "peak RSS GB"], rows)
    n = J("predict/union_single.run.json")["counts"]
    gr = J("phaseA_graph/runtime.json")["rowz[cos_heat1[knn_union_lin_q0.25[spec_k20_t0.1_std]]]"]["total_s"]
    src = (f"Source: results/predict/union_*.run.json (run records of scripts/predict_pairs.py on the union of both train pair files: "
           f"{n['n_pairs']:,} pairs, {n['n_mapped']:,} mapped, 8-core arm64 laptop, input = raw CSV). Component times include genome-wide prep and null. "
           f"E, F and H alone and Pearson/Spearman through predict_pairs were not run, so they are n/a; H as a layer costs {sec(gr)} s from cached matrix "
           f"(results/phaseA_graph/runtime.json), E {sec(J('phaseA_graph/runtime.json')['emb_lap512[knn_union_lin_q0.25[spec_k20_t0.1_std]]']['total_s'])} s.")
    return "## T7. End-to-end runtimes from raw CSV (predict_pairs)\n\n" + src + "\n\n" + t


# --------------------------------------------------------------------------- main
def main():
    out = [
        "# Report tables",
        "",
        "Generated by `scripts/40_report_tables.py` from committed files under `results/` only (no datasets/ or data_cache/ access, no recomputation). "
        "Numbers are rounded to 3 decimals (runtimes to 2; prevalences to 4 because they are ~0.005). HT-AP = Horvitz-Thompson-weighted average precision (primary metric); "
        "dev = cluster-grouped 5-fold CV on the dev universe; IFT = held-out independent final test split. 'n/a (not in results)' = not in the committed files.",
        "",
        t1(), "", t2(), "", t3(), "", t4(), "", t5(), "", t6(), "", t7(), "",
    ]
    dest = R / "report" / "tables.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out))
    print(f"wrote {dest.relative_to(ROOT)} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
