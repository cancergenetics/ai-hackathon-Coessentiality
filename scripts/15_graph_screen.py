"""Phase A A7 screen: graph / diffusion / embedding layers (graph_v1 cache) on DEV, vs the current best and GLS.

Stages (all DEV pairs only via src.evaluate / src.universe.load_dev_universe; IFT never loaded):
  screen   every graph_v1 column, run_dev_cv SEEDS[0] (one logged trial each; results/phaseA_graph/screen/)
  seeds    top 3 new layers (by SEEDS[0] HT-AP; reproduction columns excluded) on SEEDS[1..4] (logged)
  nested   selection-optimism check (SEEDS[0]): for each outer fold k pick the layer with the best mean HT-AP on the
           other 4 folds and score fold k -- (a) over the whole graph_v1 screen, (b) within the best layer's family
           (hyper-parameters of that family); logged as trials (no new scores)
  compare  paired cluster bootstrap (1,000; seed = fold seed) of ΔHT-AP on pooled out-of-fold dev predictions, per
           seed: top 3 vs the current best (rowz[socos_q0.25[spec_k20_t0.1_std]], screen_v2) and vs GLS signed
  runtime  fresh subprocess, end to end from the cached gene-effect matrix to scores for all 596,597 scorable pairs
  readme   results/phaseA_graph/README.md (generated)
Output: results/phaseA_graph/graph_screen.json (+ per-stage JSON), README.md.

Usage: ./.env/bin/python scripts/15_graph_screen.py [--stages screen,seeds,nested,compare,runtime,readme] [--allow-dirty]
"""
import argparse
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from src import evaluate, metrics, scorers  # noqa: E402
from src.config import CACHE, N_FOLDS, RESULTS, ROOT, SEEDS  # noqa: E402

RES = RESULTS / "phaseA_graph"
TAGS = ("graph_v1", "graph_v2")       # graph_v2 = round 2 (chosen after the graph_v1 screen; disclosed)
CUR = "rowz[socos_q0.25[spec_k20_t0.1_std]]"
GLS = "gls_signed"
N_BOOT = 1000
N_REPRO = 2                        # columns 0-1 reproduce the current best family (excluded from top-k / nested)
KEYS = ("ap_ht", "ap_raw", "auroc_ht", "macro_cluster_ap", "ap_ht_no_largest_component", "ap_ht_hard_neg",
        "ap_ht_both_nonessential", "ap_ht_mixed", "ap_ht_both_essential", "ht_P@R0.1", "ht_P@R0.25")


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", name).strip("_")


def manifest(tag):
    return json.loads((RES / f"{tag}.manifest.json").read_text())


_SC = {}


def cache(tag):
    from src.universe import scorable_table
    P = np.load(CACHE / "layers" / f"{tag}.pairs.npz")
    if "sc" not in _SC:
        _SC["sc"] = scorable_table()
    sc = _SC["sc"]
    if not (np.array_equal(P["ia"], sc.ia.values) and np.array_equal(P["ib"], sc.ib.values)):
        raise RuntimeError(f"{tag} cache not aligned to the scorable pair table")
    return np.load(CACHE / "layers" / f"{tag}.scores.f32.npy", mmap_mode="r")


def cur_best_scores():
    m2 = json.loads((RESULTS / "layers" / "screen_v2.manifest.json").read_text())
    col = {c["name"]: c["col"] for c in m2["columns"]}[CUR]
    return np.asarray(np.load(CACHE / "layers" / "screen_v2.scores.f32.npy", mmap_mode="r")[:, col], dtype=np.float64)


def family(cfg):
    if cfg["kind"] != "graph":
        return "other"
    if cfg["graph"]["kind"] == "soft":
        return "soft"
    if cfg["score"] == "tom":
        return "tom"
    if cfg["score"] == "emb":
        return "emb"
    if cfg["op"] == "adj":
        return "so"
    return "diffusion"


def summarise(res, name, extra=None):
    s, p = res["summary"], res.get("pooled", {})
    row = dict(name=name, trial_id=res.get("trial_id"), ap_ht_sd=s["ap_ht"]["sd"], ap_ht_folds=s["ap_ht"]["folds"],
               pooled_ap_raw=p.get("ap_raw"), pooled_ap_ht=p.get("ap_ht"), pooled_auroc_ht=p.get("auroc_ht"))
    for k in KEYS:
        row[k] = s[k]["mean"]
    return row | (extra or {})


def run_cv(u, name, arr, seed, out, allow_dirty, extra):
    tg = f"{safe(name)}_cluster_seed{seed}"
    prev = out / f"{tg}.json"
    if prev.exists():                                   # resume: reuse the logged result (no duplicate trial)
        res = json.loads(prev.read_text())
        if res["scorer"]["name"] != name:
            raise RuntimeError(f"stale {prev}")
        return res
    sc = scorers.PrecomputedScorer(name, arr, params=extra)
    return evaluate.run_dev_cv(sc, seed=seed, universe=u, out_dir=out, tag=tg, allow_dirty=allow_dirty, extra=extra)


# ------------------------------------------------------------------------------------------------ stages
def stage_screen(u, allow_dirty):
    rows, fp = [], None
    for tag in TAGS:
        if not (RES / f"{tag}.manifest.json").exists():
            continue
        M = cache(tag)
        for c in manifest(tag)["columns"]:
            arr = np.asarray(M[:, c["col"]], dtype=np.float64)
            if not np.isfinite(arr).all():
                raise ValueError(f"{c['name']}: non-finite")
            res = run_cv(u, c["name"], arr, SEEDS[0], RES / "screen", allow_dirty,
                         dict(grid=f"phaseA_{tag}", cache=tag, col=c["col"], config=c["config"]))
            fp = fp or res["split"]["fingerprint"]
            assert res["split"]["fingerprint"] == fp
            rows.append(summarise(res, c["name"], dict(cache=tag, col=c["col"], config=c["config"],
                                                        family=family(c["config"]),
                                                        repro=tag == "graph_v1" and c["col"] < N_REPRO,
                                                        runtime_cache_s=c["runtime_pairs_incl_prep_s"])))
        print(f"{c['name']:70s} HT-AP {rows[-1]['ap_ht']:.4f} ± {rows[-1]['ap_ht_sd']:.3f} pooled "
              f"{rows[-1]['pooled_ap_raw']:.4f}", flush=True)
    ref = {}
    scr = {r["name"]: r for r in json.loads((RESULTS / "phaseA" / "screen.json").read_text())["layers"]}
    ref[CUR] = scr[CUR]
    for b in json.loads((RESULTS / "phaseA" / "screen.json").read_text())["baselines"]:
        ref[b["name"]] = b
        assert b["split_fingerprint"] == fp
    out = dict(seed=SEEDS[0], split_fingerprint=fp, n_layers=len(rows), layers=rows, reference=ref)
    (RES / "screen.json").write_text(json.dumps(out, indent=1))
    return out


def top_layers(n=3):
    rows = [r for r in json.loads((RES / "screen.json").read_text())["layers"] if not r["repro"]]
    return sorted(rows, key=lambda r: -r["ap_ht"])[:n]


def candidates():
    """Top 3 (SEEDS[0] HT-AP) + the best non-embedding graph layer (family representative)."""
    rows = [r for r in json.loads((RES / "screen.json").read_text())["layers"] if not r["repro"] and r["family"] != "emb"]
    return top_layers() + [max(rows, key=lambda r: r["ap_ht"])]


def stage_seeds(u, allow_dirty):
    out = {}
    for r in candidates():
        out[r["name"]] = {SEEDS[0]: r["ap_ht"]}
        arr = np.asarray(cache(r["cache"])[:, r["col"]], dtype=np.float64)
        for s in SEEDS[1:]:
            res = run_cv(u, r["name"], arr, s, RES / "seeds", allow_dirty,
                         dict(grid="phaseA_graph_top3_seeds", cache=r["cache"], col=r["col"]))
            out[r["name"]][s] = res["summary"]["ap_ht"]["mean"]
        print(r["name"], out[r["name"]], flush=True)
    scr = {x["name"]: x for x in json.loads((RESULTS / "phaseA" / "screen.json").read_text())["layers"]}
    out[CUR] = {SEEDS[0]: scr[CUR]["ap_ht"]}
    for s in SEEDS[1:]:
        out[CUR][s] = json.loads((RESULTS / "phaseA" / "compare" / f"best_layer_cluster_seed{s}.json").read_text()
                                 )["summary"]["ap_ht"]["mean"]
    out["BASE_" + GLS] = {s: json.loads((RESULTS / "baselines_v2" / f"{GLS}_cluster_seed{s}.json").read_text()
                                        )["summary"]["ap_ht"]["mean"] for s in SEEDS}
    (RES / "seeds.json").write_text(json.dumps(out, indent=1))
    return out


def _nested(rows, label, fp):
    F = np.array([r["ap_ht_folds"] for r in rows])          # layers x folds
    recs = []
    for k in range(N_FOLDS):
        inner = [i for i in range(N_FOLDS) if i != k]
        j = int(np.argmax(F[:, inner].mean(1)))
        tid = uuid.uuid4().hex[:12]
        evaluate.log_trial("start", tid, tag=f"graph_nested_{label}_outer{k}_seed{SEEDS[0]}", universe="dev",
                           seed=SEEDS[0], scheme="cluster", git=evaluate.git_state(),
                           scorer=dict(name=f"argmax over {len(rows)} graph layers ({label}, nested)", label_free=True),
                           extra=dict(grid="phaseA_graph_nested", outer_fold=k))
        rec = dict(outer_fold=k, selected=rows[j]["name"], inner_ap_ht=float(F[j, inner].mean()),
                   held_out_ap_ht=float(F[j, k]))
        evaluate.log_trial("end", tid, tag=f"graph_nested_{label}_outer{k}_seed{SEEDS[0]}", status="ok_not_written",
                           split_fingerprint=fp, primary=dict(metric="ap_ht", held_out_fold=k, value=rec["held_out_ap_ht"]),
                           result=rec)
        recs.append(rec)
    best = int(np.argmax(F.mean(1)))
    return dict(label=label, n_candidates=len(rows), outer=recs,
                nested_mean_held_out=float(np.mean([r["held_out_ap_ht"] for r in recs])),
                selected_in_sample=dict(name=rows[best]["name"], ap_ht=float(F[best].mean())))


def stage_nested():
    d = json.loads((RES / "screen.json").read_text())
    rows = [r for r in d["layers"] if not r["repro"]]
    top = max(rows, key=lambda r: r["ap_ht"])
    fam = [r for r in rows if r["family"] == top["family"]]
    cur_folds = d["reference"][CUR]["ap_ht_folds"]
    out = dict(all=_nested(rows, "all", d["split_fingerprint"]),
               family=_nested(fam, f"family_{top['family']}", d["split_fingerprint"]),
               current_best_fold_mean=float(np.mean(cur_folds)), current_best_folds=cur_folds,
               note="current best was itself selected on these SEEDS[0] folds in earlier screens (not nested)")
    for k in ("all", "family"):
        print(k, out[k]["nested_mean_held_out"], out[k]["selected_in_sample"], flush=True)
    (RES / "nested.json").write_text(json.dumps(out, indent=1))
    return out


def stage_compare(u):
    pid = u.col("pair_id")
    tops = candidates()
    S = {r["name"]: np.asarray(cache(r["cache"])[pid, r["col"]], dtype=np.float64) for r in tops}
    S[CUR] = cur_best_scores()[pid]
    S["BASE_" + GLS] = scorers.make_baseline(GLS, scorers.real_matrix()).bind(u).score(np.arange(len(u)))
    diffs = [(r["name"], CUR) for r in tops] + [(r["name"], "BASE_" + GLS) for r in tops] + [(CUR, "BASE_" + GLS)]
    tid = uuid.uuid4().hex[:12]
    evaluate.log_trial("start", tid, tag="graph_compare_bootstrap", universe="dev", seed=SEEDS, git=evaluate.git_state(),
                       scorer=dict(name="graph top3 vs current best / GLS (paired OOF cluster bootstrap)",
                                   label_free=True), extra=dict(grid="phaseA_graph_compare"))
    out = {}
    for s in SEEDS:
        sp = evaluate.make_dev_split(u, "cluster", s)
        oof = np.flatnonzero(sp.test_fold >= 0)
        ca, cb = u.col("cluster_a")[oof], u.col("cluster_b")[oof]
        w = metrics.ht_weights(ca, cb, 1.0 / N_FOLDS)
        b = metrics.cluster_bootstrap(u.y[oof], {k: v[oof] for k, v in S.items()}, ca, cb, n_boot=N_BOOT, seed=s,
                                      diffs=diffs, base_w=w)
        out[s] = dict(ap=b["ap"], diff=b["diff"], n_clusters=b["n_clusters"], n_oof_pairs=int(len(oof)),
                      split_fingerprint=sp.fingerprint())
        print(f"seed {s}: " + "; ".join(f"{k} {v['diff']:+.3f} [{v['ci_low']:+.3f},{v['ci_high']:+.3f}]"
                                       for k, v in b["diff"].items()), flush=True)
    evaluate.log_trial("end", tid, tag="graph_compare_bootstrap", status="ok", result="results/phaseA_graph/compare.json")
    (RES / "compare.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def runtime_child(arg):
    from src import data, layers
    from src.layers_graph import GraphEngine, as_config
    tag, col = arg.split(":")
    c = manifest(tag)["columns"][int(col)]
    P = np.load(CACHE / "layers" / f"{tag}.pairs.npz")
    ia, ib = P["ia"], P["ib"]
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    X, _, _ = data.load_matrix("CRISPRGeneEffect")
    out = dict(name=c["name"], n_pairs=int(len(ia)), load_matrix_s=round(time.perf_counter() - t0, 2))
    t0 = time.perf_counter()
    ge = GraphEngine(layers.LayerEngine(X))
    s = ge.compute(as_config(c["config"]), ia, ib)
    out["layer_s"] = round(time.perf_counter() - t0, 2)
    out["total_s"] = round(time.perf_counter() - t_all, 2)
    out["max_abs_diff_vs_cache"] = float(np.max(np.abs(s - np.load(CACHE / "layers" / f"{tag}.scores.f32.npy",
                                                                     mmap_mode="r")[:, int(col)])))
    import resource
    out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    print("RUNTIME_JSON " + json.dumps(out))


def stage_runtime():
    out = json.loads((RES / "runtime.json").read_text()) if (RES / "runtime.json").exists() else {}
    for r in candidates():
        if r["name"] in out:
            continue
        p = subprocess.run([sys.executable, __file__, "--runtime-child", f"{r['cache']}:{r['col']}"], capture_output=True, text=True,
                           cwd=ROOT)
        line = [l for l in p.stdout.splitlines() if l.startswith("RUNTIME_JSON ")]
        if p.returncode or not line:
            raise RuntimeError(p.stderr[-2000:])
        out[r["name"]] = json.loads(line[-1][len("RUNTIME_JSON "):])
        print(out[r["name"]], flush=True)
    out["reference"] = json.loads((RESULTS / "phaseA" / "compare.json").read_text())["runtime"]
    (RES / "runtime.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------------ README
def f3(x):
    return "–" if x is None or not np.isfinite(x) else f"{x:.3f}"


def write_readme():
    d = json.loads((RES / "screen.json").read_text())
    man = manifest("graph_v1")
    L = ["# Phase A A7 — kNN-graph / diffusion / spectral-embedding layers (dev)", "",
         "Generated by `scripts/15_graph_screen.py` from the `graph_v1` cache (`scripts/15_build_graph_layers.py`, "
         "`src/layers_graph.py`). DEV pairs only; §3.3 cluster-grouped 5-fold CV, SEEDS[0] (split fingerprint "
         f"`{d['split_fingerprint'][:12]}`, same as Phase A). HT-AP = mean ± sd over the 5 test folds; pooled AP = AP over "
         "all dev pairs at natural prevalence; macro = macro per-cluster AP; AUROC = HT AUROC (fold mean). All layers "
         "label-free (graphs built genome-wide from CRISPRGeneEffect only). Base unless noted: spec_k20_t0.1_std; "
         "q = % of background genes kept as neighbours (q0.25 = 44).", "",
         f"Cache build: {man['runtime']['total_s']:.0f} s for {len(man['columns'])} columns (peak RSS "
         f"{man['runtime']['peak_rss_gb']} GB); reproduction of the current best inside the cache: max |Δ| "
         + ", ".join(f"{v:.1e}" for v in man["reproduction_max_abs_diff_vs_screen_v2"].values()) + ".", "",
         "## Reference (same split)", "", "| layer | HT-AP mean ± sd | pooled AP | macro-cluster AP | HT AUROC |",
         "|---|---|---|---|---|"]
    for k, r in d["reference"].items():
        L.append(f"| {k} | {r['ap_ht']:.3f} ± {r['ap_ht_sd']:.3f} | {f3(r['pooled_ap_raw'])} | "
                 f"{f3(r['macro_cluster_ap'])} | {f3(r['auroc_ht'])} |")
    L += ["", "## All graph layers (graph_v1 + graph_v2) sorted by HT-AP (SEEDS[0])", "",
          "Round 2 (cache graph_v2, marked R2) was chosen after seeing the graph_v1 results (disclosed).", "",
          "| rank | layer | family | HT-AP mean ± sd | pooled AP | macro-cluster AP | HT AUROC | AP non-ess / mixed / ess | "
          "cache build (s) |", "|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(sorted(d["layers"], key=lambda r: -r["ap_ht"]), 1):
        L.append(f"| {i} | {r['name']}{' (repro)' if r['repro'] else ''}{' (R2)' if r['cache'] == 'graph_v2' else ''} | {r['family']} | {r['ap_ht']:.3f} ± "
                 f"{r['ap_ht_sd']:.3f} | {f3(r['pooled_ap_raw'])} | {f3(r['macro_cluster_ap'])} | {f3(r['auroc_ht'])} | "
                 f"{f3(r['ap_ht_both_nonessential'])} / {f3(r['ap_ht_mixed'])} / {f3(r['ap_ht_both_essential'])} | "
                 f"{r['runtime_cache_s']:.0f} |")
    if (RES / "seeds.json").exists():
        sd = json.loads((RES / "seeds.json").read_text())
        L += ["", "## Top 3 + references on all 5 seeds (HT-AP fold mean; SEEDS[1..4] re-split the same dev pairs)", "",
              "| layer | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean |", "|---|" + "---|" * (len(SEEDS) + 1)]
        for k, v in sd.items():
            vals = [v[str(s)] if str(s) in v else v[s] for s in SEEDS]
            L.append(f"| {k} | " + " | ".join(f3(x) for x in vals) + f" | {f3(np.mean(vals))} |")
    if (RES / "nested.json").exists():
        n = json.loads((RES / "nested.json").read_text())
        L += ["", "## Nested selection check (SEEDS[0]; select on 4 folds, score the 5th)", ""]
        for k in ("all", "family"):
            x = n[k]
            L += [f"**{x['label']}** ({x['n_candidates']} candidates): nested mean held-out HT-AP "
                  f"**{x['nested_mean_held_out']:.3f}** vs in-sample best {x['selected_in_sample']['name']} "
                  f"{x['selected_in_sample']['ap_ht']:.3f} (optimism "
                  f"{x['selected_in_sample']['ap_ht'] - x['nested_mean_held_out']:+.3f}); current best "
                  f"{n['current_best_fold_mean']:.3f} (not nested).", "",
                  "| outer fold | selected on other 4 | inner HT-AP | held-out HT-AP | current best on fold |",
                  "|---|---|---|---|---|"]
            for r in x["outer"]:
                L.append(f"| {r['outer_fold']} | {r['selected']} | {r['inner_ap_ht']:.3f} | {r['held_out_ap_ht']:.3f} | "
                         f"{n['current_best_folds'][r['outer_fold']]:.3f} |")
            L.append("")
    if (RES / "compare.json").exists():
        c = json.loads((RES / "compare.json").read_text())
        keys = list(next(iter(c.values()))["diff"].keys())
        L += ["## Paired cluster bootstrap ΔHT-AP, pooled OOF dev predictions (1,000 resamples, per seed)", "",
              "| comparison | " + " | ".join(f"seed {s}" for s in c) + " |", "|---|" + "---|" * len(c)]
        for k in keys:
            L.append(f"| {k} | " + " | ".join(f"{v['diff'][k]['diff']:+.3f} [{v['diff'][k]['ci_low']:+.3f}, "
                                              f"{v['diff'][k]['ci_high']:+.3f}]" for v in c.values()) + " |")
    if (RES / "runtime.json").exists():
        rt = json.loads((RES / "runtime.json").read_text())
        L += ["", "## Runtime (fresh process; cached npy matrix -> scores for all 596,597 scorable pairs; 8-core laptop)",
              "", "| layer | total (s) | layer (s) | peak RSS (GB) | max |Δ| vs cache |", "|---|---|---|---|---|"]
        for k, v in rt.items():
            if k != "reference":
                L.append(f"| {k} | {v['total_s']} | {v['layer_s']} | {v['peak_rss_gb']} | {v['max_abs_diff_vs_cache']:.1e} |")
        for k, v in rt["reference"].items():
            L.append(f"| {k} (Phase A reference) | {v['total_s']} | – | – | – |")
    notes = RES / "notes.md"
    if notes.exists():
        L += ["", notes.read_text().strip()]
    (RES / "README.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="screen,seeds,nested,compare,runtime,readme")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--runtime-child")
    a = ap.parse_args()
    if a.runtime_child is not None:
        return runtime_child(a.runtime_child)
    subprocess.run(["sh", str(ROOT / "scripts" / "check_data.sh")], check=True, stdout=subprocess.DEVNULL)
    st = a.stages.split(",")
    u = None
    if any(s in st for s in ("screen", "seeds", "compare")):
        from src.universe import load_dev_universe
        u = load_dev_universe()
    if "screen" in st:
        stage_screen(u, a.allow_dirty)
    if "seeds" in st:
        stage_seeds(u, a.allow_dirty)
    if "nested" in st:
        stage_nested()
    if "compare" in st:
        stage_compare(u)
    if "runtime" in st:
        stage_runtime()
    if "readme" in st:
        write_readme()


if __name__ == "__main__":
    main()
