"""A6 label-free ensemble: z-averaging of cached Phase-A layers (DESIGN v3.1 §4 A6), DEV pairs only.

z-scores use each layer's FROZEN genome-wide null moments from the cache manifests (200k random background gene
pairs, seed SEEDS[0]; results/layers/manifest.json and results/layers/screen_v2.manifest.json) -- never moments of
evaluated pairs or folds. Ensemble score = unweighted mean of the member z-scores.

1. Greedy forward selection on the SEEDS[0] cluster-CV folds: start from the best single layer of the screen
   (results/phaseA/screen.json), at each step add the layer (from all cached layers) that maximises mean fold HT-AP
   of the equal-weight z-average; stop at 6 layers or when the gain < 0.002. Every candidate ensemble is one logged
   trial (src.evaluate.run_dev_cv, write=False -> trials.jsonl 'ok_not_written').
2. Selected ensemble and simple rule-based ensembles evaluated on SEEDS[0..4] (written JSON per seed; SEEDS[1..4] are
   a split-variance check, the selection used SEEDS[0] folds only):
     E_base3  best spectral layer that has cached rowz and socos_q0.5 layers + those two (same base)
     E_fam3   best layer of each family: spectral / A3 (rowz, mr, mrbal) / A4 second-order
     E_top3   top-3 single layers of the screen
3. Selection-optimism check (nested): for each SEEDS[0] fold k, rerun the greedy on the other 4 folds and score the
   chosen ensemble on fold k (one logged trial per outer fold).
Output: results/phaseA/ensemble.json, results/phaseA/ensemble/*.json, results/phaseA/ENSEMBLE.md.

Usage: ./.env/bin/python scripts/13_label_free_ensemble.py [--allow-dirty]
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

from src import evaluate, metrics  # noqa: E402
from src.config import CACHE, N_FOLDS, RESULTS, ROOT, SEEDS  # noqa: E402
from src.scorers import Scorer  # noqa: E402

OUT = RESULTS / "phaseA"
MANIFESTS = {"screen_v1": RESULTS / "layers" / "manifest.json",
             "screen_v2": RESULTS / "layers" / "screen_v2.manifest.json"}
MAX_LAYERS, MIN_GAIN = 6, 0.002


class DevArrayScorer(Scorer):
    """Label-free scorer over a fixed per-row score array of the bound (dev) universe."""

    def __init__(self, name, arr, params=None):
        super().__init__()
        self.name, self.arr, self.params = name, np.asarray(arr, dtype=np.float64), dict(params or {})

    def bind(self, universe):
        super().bind(universe)
        if len(universe) != len(self.arr):
            raise ValueError("array not aligned to universe")
        return self

    def score(self, idx):
        return self.arr[np.asarray(idx)]


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.=+-]+", "_", name).strip("_")


def load_z(u):
    """(Z dev x L float32, columns meta) for all cached layers, z-scored with the frozen null moments."""
    pid = u.col("pair_id")
    blocks, meta = [], []
    for tag, mf in MANIFESTS.items():
        if not mf.exists():
            continue
        m = json.loads(mf.read_text())
        M = np.load(CACHE / "layers" / f"{tag}.scores.f32.npy", mmap_mode="r")
        D = np.asarray(M[pid], dtype=np.float64)
        for c in m["columns"]:
            D[:, c["col"]] = (D[:, c["col"]] - c["null_mean"]) / c["null_sd"]
            meta.append(dict(cache=tag, col=c["col"], name=c["name"], config=c["config"],
                             null_mean=c["null_mean"], null_sd=c["null_sd"]))
        blocks.append(D.astype(np.float32))
    return np.concatenate(blocks, 1), meta


def ens_name(members, meta):
    return "zavg[" + "+".join(meta[j]["name"] for j in members) + "]"


def evaluate_ensemble(u, Z, meta, members, seed, write, allow_dirty, grid, n_boot=0):
    arr = Z[:, members].astype(np.float64).mean(1)
    name = ens_name(members, meta)
    sc = DevArrayScorer(name, arr, params=dict(kind="zavg", weights="equal", null="frozen genome-wide (manifest)",
                                               members=[dict(cache=meta[j]["cache"], col=meta[j]["col"],
                                                             name=meta[j]["name"]) for j in members]))
    tag = f"ens_{grid}_seed{seed}"
    return evaluate.run_dev_cv(sc, seed=seed, universe=u, out_dir=OUT / "ensemble", tag=tag, write=write,
                               allow_dirty=allow_dirty, n_boot_pooled=n_boot, extra=dict(grid=grid))


def greedy(u, Z, meta, start, allow_dirty):
    sel, path = [start], []
    screen = {r["name"]: r for r in json.loads((OUT / "screen.json").read_text())["layers"]}
    cur = screen[meta[start]["name"]]["ap_ht"]
    path.append(dict(step=1, added=meta[start]["name"], ap_ht=cur, gain=None, n_candidates=len(meta)))
    while len(sel) < MAX_LAYERS:
        best, best_ap, t0 = None, -1, time.perf_counter()
        for j in range(len(meta)):
            if j in sel:
                continue
            r = evaluate_ensemble(u, Z, meta, sel + [j], SEEDS[0], False, allow_dirty, f"A6_greedy_step{len(sel) + 1}")
            ap = r["summary"]["ap_ht"]["mean"]
            if ap > best_ap:
                best, best_ap = j, ap
        gain = best_ap - cur
        path.append(dict(step=len(sel) + 1, added=meta[best]["name"], ap_ht=best_ap, gain=gain,
                         accepted=bool(gain >= MIN_GAIN), n_candidates=len(meta) - len(sel),
                         runtime_s=round(time.perf_counter() - t0, 1)))
        print(f"step {len(sel) + 1}: +{meta[best]['name']} -> {best_ap:.4f} (gain {gain:+.4f})", flush=True)
        if gain < MIN_GAIN:
            break
        sel.append(best)
        cur = best_ap
    return sel, path


def fold_cache(u, seed):
    sp = evaluate.make_dev_split(u, "cluster", seed)
    out = []
    for k in range(N_FOLDS):
        te = np.flatnonzero(sp.test_mask(k))
        out.append((te, u.y[te], evaluate.design_weights(u, te, "cluster")))
    return out, sp.fingerprint()


def fast_fold_ap(arr, folds):
    return [metrics.weighted_ap(y, arr[te], w) for te, y, w in folds]


def nested_check(u, Z, meta):
    """Greedy on 4 folds, score on the 5th (SEEDS[0]); one logged trial per outer fold."""
    folds, fp = fold_cache(u, SEEDS[0])
    single = np.array([fast_fold_ap(Z[:, j].astype(np.float64), folds) for j in range(Z.shape[1])])   # L x K
    out = []
    for k in range(N_FOLDS):
        inner = [i for i in range(N_FOLDS) if i != k]
        tid = uuid.uuid4().hex[:12]
        evaluate.log_trial("start", tid, tag=f"A6_nested_outer{k}_seed{SEEDS[0]}", universe="dev", seed=SEEDS[0],
                           scheme="cluster", scorer=dict(name="zavg greedy (nested)", label_free=True),
                           extra=dict(grid="A6_nested", outer_fold=k), git=evaluate.git_state())
        sel = [int(np.argmax(single[:, inner].mean(1)))]
        cur = single[sel[0], inner].mean()
        while len(sel) < MAX_LAYERS:
            base = Z[:, sel].astype(np.float64).sum(1)
            best, best_ap = None, -1
            for j in range(Z.shape[1]):
                if j in sel:
                    continue
                arr = (base + Z[:, j]) / (len(sel) + 1)
                ap = float(np.mean([metrics.weighted_ap(folds[i][1], arr[folds[i][0]], folds[i][2]) for i in inner]))
                if ap > best_ap:
                    best, best_ap = j, ap
            if best_ap - cur < MIN_GAIN:
                break
            sel.append(best)
            cur = best_ap
        arr = Z[:, sel].astype(np.float64).mean(1)
        te, y, w = folds[k]
        held = metrics.weighted_ap(y, arr[te], w)
        rec = dict(outer_fold=k, members=[meta[j]["name"] for j in sel], inner_ap_ht=cur, held_out_ap_ht=held,
                   best_single_inner=meta[int(np.argmax(single[:, inner].mean(1)))]["name"],
                   best_single_inner_held=float(single[int(np.argmax(single[:, inner].mean(1))), k]))
        evaluate.log_trial("end", tid, tag=f"A6_nested_outer{k}_seed{SEEDS[0]}", status="ok_not_written",
                           split_fingerprint=fp, primary=dict(metric="ap_ht", held_out_fold=k, value=held), result=rec)
        out.append(rec)
        print(f"nested outer {k}: held {held:.4f} (inner {cur:.4f}) {rec['members']}", flush=True)
    return out


def best_by(screen, pred):
    rows = [r for r in screen if pred(r)]
    return max(rows, key=lambda r: r["ap_ht"])["name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    subprocess.run(["sh", str(ROOT / "scripts" / "check_data.sh")], check=True, stdout=subprocess.DEVNULL)
    from src.universe import load_dev_universe
    u = load_dev_universe()
    Z, meta = load_z(u)
    idx = {m["name"]: j for j, m in enumerate(meta)}
    screen = json.loads((OUT / "screen.json").read_text())["layers"]
    if {r["name"] for r in screen} != set(idx):
        raise RuntimeError("screen.json does not cover all cached layers; run 11_screen_layers.py first")
    ranked = sorted(screen, key=lambda r: -r["ap_ht"])
    t0 = time.perf_counter()
    sel, path = greedy(u, Z, meta, idx[ranked[0]["name"]], a.allow_dirty)
    t_greedy = time.perf_counter() - t0

    # rule-based ensembles (rules fixed in this docstring; names resolved from the screen)
    kind = lambda r: r["config"]["kind"]  # noqa: E731
    best_spec = best_by(screen, lambda r: kind(r) == "spectral")
    base3 = best_by(screen, lambda r: kind(r) == "spectral" and f"rowz[{r['name']}]" in idx
                    and f"socos_q0.5[{r['name']}]" in idx)
    fixed = {
        "E_base3": [base3, f"rowz[{base3}]", f"socos_q0.5[{base3}]"],
        "E_fam3": [best_spec, best_by(screen, lambda r: kind(r) in ("rowz", "mr")),
                   best_by(screen, lambda r: kind(r) in ("so", "rowzso"))],
        "E_top3": [r["name"] for r in ranked[:3]],
    }
    ens = {"A6_greedy": sel} | {k: [idx[n] for n in v] for k, v in fixed.items()}
    results = {}
    for g, mem in ens.items():
        per_seed = {}
        for s in SEEDS:
            r = evaluate_ensemble(u, Z, meta, mem, s, True, a.allow_dirty, g, n_boot=1000 if s == SEEDS[0] else 0)
            per_seed[s] = r
        s0 = per_seed[SEEDS[0]]
        results[g] = dict(members=[meta[j]["name"] for j in mem], name=ens_name(mem, meta),
                          members_meta=[meta[j] for j in mem],
                          ap_ht_seed0=dict(mean=s0["summary"]["ap_ht"]["mean"], sd=s0["summary"]["ap_ht"]["sd"],
                                           folds=s0["summary"]["ap_ht"]["folds"]),
                          ap_ht_by_seed={s: per_seed[s]["summary"]["ap_ht"]["mean"] for s in SEEDS},
                          summary_seed0={k: v["mean"] for k, v in s0["summary"].items()},
                          pooled_seed0={k: s0["pooled"][k] for k in ("ap_raw", "ap_ht", "auroc_ht", "ap_ht_ci")},
                          results={s: str(Path(per_seed[s]["path"]).relative_to(ROOT)) for s in SEEDS})
        print(g, results[g]["ap_ht_by_seed"], flush=True)
    nested = nested_check(u, Z, meta)
    out = dict(seed_selection=SEEDS[0], null="frozen genome-wide moments from cache manifests",
               rule=dict(max_layers=MAX_LAYERS, min_gain=MIN_GAIN, weights="equal", pool=len(meta)),
               greedy_path=path, greedy_runtime_s=round(t_greedy, 1), ensembles=results, nested=nested,
               nested_mean_held_out=float(np.mean([n["held_out_ap_ht"] for n in nested])),
               note="greedy selection on SEEDS[0] dev folds inflates the A6_greedy dev numbers; SEEDS[1..4] reuse the "
                    "same pairs (split variance only, not fresh evidence); nested = honest(er) estimate")
    (OUT / "ensemble.json").write_text(json.dumps(out, indent=1, default=str))
    write_md(out)


def write_md(d):
    L = ["# A6 label-free ensembles (z-averaging, frozen genome-wide null)", "",
         "Generated by `scripts/13_label_free_ensemble.py`. DEV pairs only. HT-AP = mean over the 5 cluster-CV test "
         "folds. Selection used SEEDS[0] folds only; SEEDS[1..4] re-split the same dev pairs (split-variance check). "
         "**Greedy selection on dev inflates A6_greedy's dev numbers**; the nested check (greedy on 4 folds, scored on "
         "the 5th) and the rule-based ensembles gauge the optimism.", "",
         "## Greedy path (SEEDS[0])", "", "| step | added | HT-AP | gain |", "|---|---|---|---|"]
    for p in d["greedy_path"]:
        g = "" if p["gain"] is None else f"{p['gain']:+.4f}" + ("" if p.get("accepted", True) else " (rejected)")
        L.append(f"| {p['step']} | {p['added']} | {p['ap_ht']:.4f} | {g} |")
    L += ["", "## Ensembles", "", "| ensemble | members | HT-AP seed0 mean ± sd | HT-AP seeds 0..4 | mean over seeds | "
          "pooled AP | pooled HT-AP [95% CI] | macro-cluster AP | AP w/o largest | AP non-ess / mixed / ess |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g, r in d["ensembles"].items():
        bs = [r["ap_ht_by_seed"][s] if s in r["ap_ht_by_seed"] else r["ap_ht_by_seed"][str(s)] for s in SEEDS]
        sm, p = r["summary_seed0"], r["pooled_seed0"]
        L.append(f"| {g} | {' + '.join(r['members'])} | {r['ap_ht_seed0']['mean']:.3f} ± {r['ap_ht_seed0']['sd']:.3f} | "
                 + ", ".join(f"{x:.3f}" for x in bs) + f" | {np.mean(bs):.3f} ± {np.std(bs, ddof=1):.3f} | "
                 f"{p['ap_raw']:.3f} | {p['ap_ht']:.3f} [{p['ap_ht_ci'][0]:.3f}, {p['ap_ht_ci'][1]:.3f}] | "
                 f"{sm['macro_cluster_ap']:.3f} | {sm['ap_ht_no_largest_component']:.3f} | "
                 f"{sm['ap_ht_both_nonessential']:.3f} / {sm['ap_ht_mixed']:.3f} / {sm['ap_ht_both_essential']:.3f} |")
    L += ["", f"## Nested selection check (SEEDS[0]; mean held-out HT-AP {d['nested_mean_held_out']:.3f})", "",
          "| outer fold | members selected on other 4 folds | inner HT-AP | held-out HT-AP | best single (inner) held-out |",
          "|---|---|---|---|---|"]
    for n in d["nested"]:
        L.append(f"| {n['outer_fold']} | {' + '.join(n['members'])} | {n['inner_ap_ht']:.3f} | {n['held_out_ap_ht']:.3f} | "
                 f"{n['best_single_inner']}: {n['best_single_inner_held']:.3f} |")
    (OUT / "ENSEMBLE.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
