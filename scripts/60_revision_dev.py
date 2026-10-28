"""Manuscript revision round 1 (manuscript/review_round1.md): dev-only analyses. IFT pairs are never loaded here.

Stages
  controls  (M9) olfactory-receptor (OR) and chromosome-proximity controls for the FINAL method zavg_BEF, with B and
            GLS signed recomputed by the same code. The control functions are imported unchanged from
            scripts/14_phaseA_compare.py (``or_control``, ``chrom_control``), i.e. exactly the protocol of
            results/phaseA/compare.json:
              OR: all genome-wide OR x OR gene pairs (HGNC group 'Olfactory receptors'), fraction above the 99th
                  (99.9th) percentile of the scorer on the fixed 200k-pair genome-wide null (src.layers.null_pairs),
                  split same chromosome / different chromosome (HGNC location);
              chromosome: pooled HT-AP on all dev pairs (HT weights, f = dev protein fraction) vs excluding
                  same-chromosome / same-arm pairs; and SEEDS[0] fold-mean HT-AP.
            Genome-wide scores of zavg_BEF / B come from src.predict.PairScorer (frozen configs; null moments asserted
            equal to the manifests), zavg_BEF = mean of manifest-null z of B, E, F. GLS: baselines.gls_pairs with the
            sample covariance, as in Phase A. Dev scores for the chromosome control: frozen cached candidates
            (src.scorers.make_candidate) and the GLS baseline scorer.
  onion     (B1) PC-removed Pearson reference rows ("onion" normalisation, Boyle 2018 / Hassan 2023): cached Phase-A
            layers spec_k5_tinf, spec_k20_tinf (tau = inf: no whitening, Pearson after removing the top-k PCs of the
            line x line second-moment matrix) and their gene-standardised versions spec_k{5,20}_tinf_std
            (results/layers/manifest.json). Each: src.evaluate.run_dev_cv, cluster CV, SEEDS (logged trials);
            pooled-OOF metrics incl. AP without the top-10 clusters (pooled_metrics of scripts/21_phaseB_v2.py);
            paired cluster bootstrap (1,000, seed = split seed) of pooled-OOF ΔHT-AP vs zavg_BEF (and vs GLS),
            exactly as stage 'compare' of scripts/21_phaseB_v2.py. zavg_BEF / GLS fold metrics are read from
            results/phaseB_v2/runs/.
            The best of the four by 5-seed mean dev HT-AP is named in onion.json as the single post-hoc IFT reference
            (scripts/final_eval.py --posthoc-reference); this is a reference row, not a candidate.

Output: results/revision/{controls,onion}.json (+ onion_runs/*.json).
Usage: ./.env/bin/python scripts/60_revision_dev.py --stages controls,onion [--allow-dirty]
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from src import evaluate, metrics, scorers  # noqa: E402
from src.config import N_FOLDS, RESULTS, ROOT, SEEDS  # noqa: E402

OUT = RESULTS / "revision"
PB2 = RESULTS / "phaseB_v2" / "runs"
N_BOOT = 1000
ONION = ["spec_k5_tinf", "spec_k20_tinf", "spec_k5_tinf_std", "spec_k20_tinf_std"]
ONION_TAG = "screen_v1"                       # results/layers/manifest.json == data_cache/layers/screen_v1.*
BEF = ("B", "E", "F")


def _load_script(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, ROOT / "scripts" / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PA = _load_script("14_phaseA_compare.py", "phaseA_compare")
PB = _load_script("21_phaseB_v2.py", "phaseB_v2")


# ------------------------------------------------------------------------------------------ M9 controls
def stage_controls(u, allow_dirty):
    from src import baselines, data, predict
    tid = uuid.uuid4().hex[:12]
    evaluate.log_trial("start", tid, tag="revision_M9_or_chrom_controls", universe="dev", seed=SEEDS[0],
                       scorer=dict(name="OR + chromosome controls: zavg_BEF, B, gls_signed", label_free=True),
                       git=evaluate.git_state(), allow_dirty=allow_dirty, extra=dict(grid="revision_controls"))
    t0 = time.perf_counter()
    try:
        X, _, genes = data.load_matrix("CRISPRGeneEffect")
        sc = predict.PairScorer(X, genes, keep_graph=True)
        Sig = baselines.sample_covariance(X)[0]

        def zavg(ia, ib):
            return np.mean([sc.z(c, sc.component(c, ia, ib)) for c in BEF], axis=0)

        fns = {"zavg_BEF": zavg,
               "B": lambda ia, ib: sc.component("B", ia, ib),
               "BASE_gls_signed": lambda ia, ib: baselines.gls_pairs(X, ia, ib, Sigma=Sig)["signed"]}
        orc = PA.or_control(fns, sc.eng, genes)
        orc["null_moment_checks"] = {c: {k: sc.moments[c].get(k) for k in ("rel_diff_mean", "rel_diff_sd")}
                                     for c in sc.moments}
        pid = u.col("pair_id")
        S = {"zavg_BEF": scorers.make_candidate("zavg_BEF").arr[pid], "B": scorers.make_candidate("B").arr[pid]}
        g = scorers.make_baseline("gls_signed", scorers.real_matrix()).bind(u)
        S["BASE_gls_signed"] = g.score(np.arange(len(u)))
        chc = PA.chrom_control(u, S, genes)
        # sanity: B / GLS reproduce results/phaseA/compare.json (same code, same data)
        ref = json.loads((RESULTS / "phaseA" / "compare.json").read_text())
        repro = {}
        for k, rk in (("B", "best_layer"), ("BASE_gls_signed", "BASE_gls_signed")):
            a, b = orc["by_scorer"][k], ref["or_control"]["by_scorer"][rk]
            repro[f"or_{k}"] = {m: [a[m], b[m]] for m in ("frac_above_null_p99", "frac_above_null_p99_same_chrom",
                                                         "frac_above_null_p99_diff_chrom")}
            c1, c2 = chc["by_scorer"][k], ref["chrom_control"]["by_scorer"].get(rk)
            if c2:
                repro[f"chrom_{k}"] = {m: [c1[m], c2[m]] for m in ("pooled_ap_ht_all", "pooled_ap_ht_excl_same_chrom")}
    except BaseException as e:
        evaluate.log_trial("end", tid, tag="revision_M9_or_chrom_controls", status="error", error=repr(e)[:500])
        raise
    out = dict(what="M9: OR and chromosome-proximity controls for zavg_BEF (final method), B and GLS signed; "
                    "functions or_control / chrom_control of scripts/14_phaseA_compare.py unchanged",
               or_control=orc, chrom_control=chc, reproduces_phaseA="[this run, results/phaseA/compare.json]",
               phaseA_reproduction=repro, runtime_s=round(time.perf_counter() - t0, 1),
               run_record=evaluate.run_record(seed=SEEDS[0], allow_dirty=allow_dirty, universe=u.name))
    (OUT / "controls.json").write_text(json.dumps(out, indent=1, default=evaluate._json_default))
    evaluate.log_trial("end", tid, tag="revision_M9_or_chrom_controls", status="ok", result="results/revision/controls.json")
    for k, r in orc["by_scorer"].items():
        print(f"OR {k}: >p99 {r['frac_above_null_p99']:.4f} same {r['frac_above_null_p99_same_chrom']:.4f} "
              f"diff {r['frac_above_null_p99_diff_chrom']:.4f}", flush=True)
    for k, r in chc["by_scorer"].items():
        print(f"chrom {k}: all {r['pooled_ap_ht_all']:.4f} excl {r['pooled_ap_ht_excl_same_chrom']:.4f}", flush=True)


# ------------------------------------------------------------------------------------------ B1 onion reference
def onion_scores():
    from src.supervised import load_layer_cache
    M, man, cols = load_layer_cache(ONION_TAG)
    out = {}
    for n in ONION:
        c = cols[n]
        cfg = c["config"]
        assert cfg["kind"] == "spectral" and cfg["tau"] == "inf" and cfg["center"] == "gene", cfg
        out[n] = (np.asarray(M[:, c["col"]], dtype=np.float64), dict(cache=ONION_TAG, col=c["col"], config=cfg))
    return out


def stage_onion(u, allow_dirty):
    runs_dir = OUT / "onion_runs"
    pid = u.col("pair_id")
    lay = onion_scores()
    res = {n: {} for n in ONION}
    oof = {n: {} for n in ONION}
    for s in SEEDS:
        sp = evaluate.make_dev_split(u, "cluster", s)
        te = sp.test_fold >= 0
        for n, (arr, params) in lay.items():
            tag = f"onion_{n}_cluster_seed{s}"
            r = evaluate.run_dev_cv(scorers.PrecomputedScorer(n, arr, params=params), seed=s, universe=u,
                                    out_dir=runs_dir, tag=tag, allow_dirty=allow_dirty,
                                    extra=dict(grid="revision_B1_onion", layer=n))
            o = np.full(len(u), np.nan)
            o[te] = arr[pid][te]
            oof[n][s] = o
            pm = PB.pooled_metrics(u, o)
            res[n][s] = dict(ap_ht=r["summary"]["ap_ht"]["mean"], ap_ht_sd=r["summary"]["ap_ht"]["sd"],
                             macro_cluster_ap=r["summary"]["macro_cluster_ap"]["mean"],
                             ap_ht_no_largest_component=r["summary"]["ap_ht_no_largest_component"]["mean"],
                             pooled_ap_ht=pm["ap_ht"], pooled_ap_ht_no_top10=pm["ap_ht_no_top10"],
                             result=r["path"] and str(Path(r["path"]).relative_to(ROOT)))
            print(f"{tag:40s} HT-AP {res[n][s]['ap_ht']:.3f} macro {res[n][s]['macro_cluster_ap']:.3f} "
                  f"no-top10 {res[n][s]['pooled_ap_ht_no_top10']:.3f}", flush=True)
    # references from Phase B v2 (same splits, same evaluator)
    refs = {}
    for name, entry in (("zavg_BEF", "zavg_BEF"), ("gls_signed", "ref_G")):
        refs[name] = {}
        for s in SEEDS:
            d = json.loads((PB2 / f"{entry}_cluster_seed{s}.json").read_text())
            refs[name][s] = dict(ap_ht=d["summary"]["ap_ht"]["mean"], macro_cluster_ap=d["summary"]["macro_cluster_ap"]["mean"],
                                 ap_ht_no_largest_component=d["summary"]["ap_ht_no_largest_component"]["mean"],
                                 pooled_ap_ht=d["phaseB_v2"]["pooled_oof"]["ap_ht"],
                                 pooled_ap_ht_no_top10=d["phaseB_v2"]["pooled_oof"]["ap_ht_no_top10"],
                                 result=str((PB2 / f"{entry}_cluster_seed{s}.json").relative_to(ROOT)))
    # paired cluster bootstrap on pooled OOF (as scripts/21_phaseB_v2.py stage_compare)
    tid = uuid.uuid4().hex[:12]
    evaluate.log_trial("start", tid, tag="revision_B1_onion_bootstrap", universe="dev", seed=list(SEEDS),
                       git=evaluate.git_state(), allow_dirty=allow_dirty,
                       scorer=dict(name="pooled-OOF paired cluster bootstrap: onion layers vs zavg_BEF, GLS",
                                   label_free=True), extra=dict(grid="revision_B1_onion"))
    zb = scorers.make_candidate("zavg_BEF").arr[pid]
    gls = scorers.make_baseline("gls_signed", scorers.real_matrix()).bind(u).score(np.arange(len(u)))
    boots = {}
    for s in SEEDS:
        sp = evaluate.make_dev_split(u, "cluster", s)
        idx = np.flatnonzero(sp.test_fold >= 0)
        for n in ONION:
            assert np.array_equal(np.flatnonzero(np.isfinite(oof[n][s])), idx)
        Sx = {**{n: oof[n][s][idx] for n in ONION}, "zavg_BEF": zb[idx], "gls_signed": gls[idx]}
        y = u.y[idx]
        ca, cb = u.col("cluster_a")[idx], u.col("cluster_b")[idx]
        w = metrics.ht_weights(ca, cb, 1.0 / N_FOLDS)
        diffs = [(n, "zavg_BEF") for n in ONION] + [(n, "gls_signed") for n in ONION]
        b = metrics.cluster_bootstrap(y, Sx, ca, cb, n_boot=N_BOOT, seed=s, diffs=diffs, base_w=w)
        # check the pooled zavg_BEF OOF equals the Phase B v2 value
        assert abs(b["ap"]["zavg_BEF"]["ap"] - refs["zavg_BEF"][s]["pooled_ap_ht"]) < 1e-9
        boots[s] = dict(diff=b["diff"], ap=b["ap"], n_pairs=int(len(idx)), n_pos=int(y.sum()),
                        n_clusters=b["n_clusters"], split_fingerprint=sp.fingerprint())
        print(f"seed {s}: " + ", ".join(f"{k} {v['diff']:+.3f} [{v['ci_low']:+.3f},{v['ci_high']:+.3f}]"
                                        for k, v in b["diff"].items() if k.endswith("zavg_BEF")), flush=True)
    evaluate.log_trial("end", tid, tag="revision_B1_onion_bootstrap", status="ok", result="results/revision/onion.json")

    def m5(d, k):
        return float(np.mean([d[s][k] for s in SEEDS]))

    keys = ("ap_ht", "macro_cluster_ap", "ap_ht_no_largest_component", "pooled_ap_ht", "pooled_ap_ht_no_top10")
    summary = {n: {k: m5(res[n], k) for k in keys} for n in ONION}
    summary.update({n: {k: m5(refs[n], k) for k in keys} for n in refs})
    best = max(ONION, key=lambda n: summary[n]["ap_ht"])
    out = dict(what="B1: PC-removed Pearson ('onion') reference rows, dev cluster CV, 5 seeds; not candidates",
               layers={n: lay[n][1] for n in ONION}, per_seed=res, references=refs, summary_5seed_mean=summary,
               bootstrap=boots,
               delta_vs_zavg_BEF={n: {s: {k: boots[s]["diff"][f"{n}-zavg_BEF"][k] for k in ("diff", "ci_low", "ci_high")}
                                      for s in SEEDS} for n in ONION},
               delta_vs_gls={n: {s: {k: boots[s]["diff"][f"{n}-gls_signed"][k] for k in ("diff", "ci_low", "ci_high")}
                                 for s in SEEDS} for n in ONION},
               best_by_dev_ht_ap=best,
               notes=["HT-AP, macro, no-largest = fold mean (cluster CV); pooled / no-top10 = pooled OOF of the 5 test "
                      "folds (HT weights 1 within / K between clusters), as Phase B v2",
                      "bootstrap: 1,000 resamples over dev clusters, seed = split seed; Δ = layer − reference",
                      "best_by_dev_ht_ap is the single post-hoc IFT reference (look '2-posthoc'); no selection of the "
                      "final method"],
               run_record=evaluate.run_record(seed=list(SEEDS), allow_dirty=allow_dirty, universe=u.name))
    (OUT / "onion.json").write_text(json.dumps(out, indent=1, default=evaluate._json_default))
    print("best", best, summary[best], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="controls,onion")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    subprocess.run(["sh", str(ROOT / "scripts" / "check_data.sh")], check=True, stdout=subprocess.DEVNULL)
    evaluate.check_clean(a.allow_dirty)
    OUT.mkdir(parents=True, exist_ok=True)
    from src.universe import load_dev_universe
    u = load_dev_universe()
    st = a.stages.split(",")
    if "onion" in st:
        stage_onion(u, a.allow_dirty)
    if "controls" in st:
        stage_controls(u, a.allow_dirty)


if __name__ == "__main__":
    main()
