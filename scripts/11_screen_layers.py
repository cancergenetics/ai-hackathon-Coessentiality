"""Phase A screening sweep (DESIGN v3.1 §4 Phase A, §3.5): every cached label-free layer on the dev CV.

For each column of the label-free layer caches (data_cache/layers/<tag>.scores.f32.npy, aligned to the canonical
scorable pair_id; built by scripts/10_build_layers.py (screen_v1) and scripts/12_build_layers_v2.py (screen_v2)),
runs ``src.evaluate.run_dev_cv`` with the fast path (``PrecomputedScorer``) on the DEV universe, §3.3 cluster-grouped
5-fold CV, SEEDS[0] (one logged trial per layer in results/trials.jsonl; per-layer JSON in results/phaseA/screen/).

Reported per layer: HT-weighted AP mean ± sd over folds (primary), pooled natural-prevalence AP over all dev pairs,
HT AUROC (fold mean), HT AP per essentiality stratum (fold mean), macro per-cluster AP (fold mean), AP without the
largest component, pooled HT AP. Reference rows: best baseline variants (results/baselines_v2, same split).
Output: results/phaseA/screen.json and results/phaseA/README.md (generated; trend tables included).

IFT pairs are never loaded (src.universe.load_dev_universe only).

Usage: ./.env/bin/python scripts/11_screen_layers.py [--tags screen_v1,screen_v2] [--readme-only] [--allow-dirty]
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from src import evaluate, scorers  # noqa: E402
from src.config import CACHE, RESULTS, ROOT, SEEDS  # noqa: E402

OUT = RESULTS / "phaseA"
LAYER_DIR = CACHE / "layers"
MANIFESTS = {"screen_v1": RESULTS / "layers" / "manifest.json",
             "screen_v2": RESULTS / "layers" / "screen_v2.manifest.json"}
BASE_DIR = RESULTS / "baselines_v2"
KEYS = ("ap_ht", "ap_raw", "auroc_ht", "macro_cluster_ap", "ap_ht_no_largest_component", "ap_ht_hard_neg",
        "ap_ht_both_nonessential", "ap_ht_mixed", "ap_ht_both_essential", "ht_P@R0.1", "ht_P@R0.25")


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", name).strip("_")


def manifest(tag):
    return json.loads(MANIFESTS[tag].read_text())


def load_cache(tag):
    """(scores memmap, columns) with alignment to the canonical scorable table asserted."""
    from src.universe import scorable_table
    m = manifest(tag)
    P = np.load(LAYER_DIR / f"{tag}.pairs.npz", allow_pickle=False)
    sc = scorable_table()
    if not (np.array_equal(P["ia"], sc.ia.values) and np.array_equal(P["ib"], sc.ib.values)):
        raise RuntimeError(f"{tag}: cache rows are not aligned to the scorable pair table")
    M = np.load(LAYER_DIR / f"{tag}.scores.f32.npy", mmap_mode="r")
    if M.shape[1] != len(m["columns"]):
        raise RuntimeError(f"{tag}: manifest/cache column mismatch")
    return M, m["columns"]


def summarise(res, name, tag, col):
    s, p = res["summary"], res["pooled"]
    row = dict(name=name, cache=tag, col=col, result=str(Path(res["path"]).relative_to(ROOT)) if res["path"] and Path(res["path"]).is_absolute() else res["path"], trial_id=res["trial_id"])
    for k in KEYS:
        row[k] = s[k]["mean"]
    row["ap_ht_sd"] = s["ap_ht"]["sd"]
    row["ap_ht_folds"] = s["ap_ht"]["folds"]
    row["pooled_ap_raw"] = p["ap_raw"]
    row["pooled_ap_ht"] = p["ap_ht"]
    row["pooled_auroc_ht"] = p["auroc_ht"]
    return row


def baseline_rows():
    best = json.loads((BASE_DIR / "best_variants.json").read_text())["best_variants"]
    rows = []
    for fam, v in best.items():
        r = json.loads((BASE_DIR / f"{v}_cluster_seed{SEEDS[0]}.json").read_text())
        row = summarise(dict(summary=r["summary"], pooled=r["pooled"], path=f"results/baselines_v2/{v}_cluster_seed0.json",
                             trial_id=r["trial_id"]), f"BASELINE {v}", "baselines_v2", None)
        row["split_fingerprint"] = r["split"]["fingerprint"]
        rows.append(row)
    return rows


def run(tags, allow_dirty):
    subprocess.run(["sh", str(ROOT / "scripts" / "check_data.sh")], check=True, stdout=subprocess.DEVNULL)
    from src.universe import load_dev_universe
    u = load_dev_universe()
    rows = []
    fp = None
    for tag in tags:
        M, cols = load_cache(tag)
        for c in cols:
            t0 = time.perf_counter()
            arr = np.asarray(M[:, c["col"]], dtype=np.float64)
            if not np.isfinite(arr).all():
                raise ValueError(f"{c['name']}: non-finite scores")
            sc = scorers.PrecomputedScorer(c["name"], arr, params=dict(cache=tag, col=c["col"], config=c["config"]))
            tg = f"{safe(c['name'])}_cluster_seed{SEEDS[0]}"
            prev = OUT / "screen" / f"{tg}.json"
            if prev.exists():                         # resume: reuse the logged result (no duplicate trial)
                res = json.loads(prev.read_text())
                res["path"] = str(prev)
                if res["scorer"]["params"].get("cache") != tag or res["scorer"]["name"] != c["name"]:
                    raise RuntimeError(f"stale result {prev}")
            else:
                res = evaluate.run_dev_cv(sc, seed=SEEDS[0], universe=u, out_dir=OUT / "screen", tag=tg,
                                          allow_dirty=allow_dirty, extra=dict(grid=f"phaseA_{tag}", cache=tag,
                                                                              col=c["col"]))
            fp = fp or res["split"]["fingerprint"]
            assert res["split"]["fingerprint"] == fp
            rows.append(summarise(res, c["name"], tag, c["col"]) | dict(config=c["config"]))
            print(f"{c['name']:45s} HT-AP {rows[-1]['ap_ht']:.4f} ± {rows[-1]['ap_ht_sd']:.3f}  "
                  f"pooled {rows[-1]['pooled_ap_raw']:.4f}  ({time.perf_counter() - t0:.1f}s)", flush=True)
    base = baseline_rows()
    for b in base:
        if b["split_fingerprint"] != fp:
            raise RuntimeError("baseline split differs from screening split")
    out = dict(seed=SEEDS[0], scheme="cluster", split_fingerprint=fp, caches=tags, n_layers=len(rows),
               primary="ap_ht (mean over 5 cluster-CV test folds)", layers=rows, baselines=base,
               note="dev pairs only; one logged trial per layer (results/trials.jsonl, grid=phaseA_<cache>)")
    (OUT / "screen.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------------------------- README
def f3(x):
    return "–" if x is None or not np.isfinite(x) else f"{x:.3f}"


def label(cfg):
    if cfg["kind"] == "spectral":
        return cfg
    return cfg["base"]


def spec_name(b):
    t = "lw" if b["shrink"] == "lw" else b["tau"]
    return f"spec_k{b['k']}_t{t}" + ("_line" if b["center"] == "line" else "") + ("_std" if b["standardize"] else "")


def trend_tables(rows):
    by = {r["name"]: r for r in rows}
    L = []
    spec = [r for r in rows if r["config"]["kind"] == "spectral" and r["config"]["center"] == "gene"
            and not r["config"]["standardize"]]
    ks = sorted({r["config"]["k"] for r in spec})
    taus = ["0", "0.01", "0.1", "1", "10", "inf", "lw"]
    L += ["### A1 spectral filters (gene-centred, unstandardised): HT-AP by k (rows) × τ (columns)", "",
          "τ is relative to the mean eigenvalue (0 = GLS whitening on range(Σ), inf = Pearson after top-k PC removal, "
          "lw = Ledoit–Wolf, τ≈0.055).", "",
          "| k | " + " | ".join(f"τ={t}" for t in taus) + " |", "|---|" + "---|" * len(taus)]
    for k in ks:
        cells = []
        for t in taus:
            n = f"spec_k{k}_t{t}"
            cells.append(f3(by[n]["ap_ht"]) if n in by else "–")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    std = [r for r in rows if r["config"]["kind"] == "spectral" and (r["config"]["standardize"]
                                                                     or r["config"]["center"] == "line")]
    if std:
        L += ["", "Standardised-Σ / line-centred variants (vs the same k, τ unstandardised):", "",
              "| layer | HT-AP | unstandardised / gene-centred counterpart |", "|---|---|---|"]
        for r in std:
            c = dict(r["config"])
            twin = f"spec_k{c['k']}_t{'lw' if c['shrink'] == 'lw' else c['tau']}"
            L.append(f"| {r['name']} | {f3(r['ap_ht'])} | {twin}: {f3(by[twin]['ap_ht']) if twin in by else '–'} |")
    derived = [r for r in rows if r["config"]["kind"] != "spectral" and r["config"]["base"]["kind"] == "spectral"]
    bases = []
    for r in derived:
        b = spec_name(r["config"]["base"])
        if b not in bases:
            bases.append(b)
    if derived:
        cols = ["base", "rowz", "mr", "mrbal"] + [f"so{m}_q{q}" for m in ("cos", "jaccard") for q in ("0.25", "0.5", "1", "2")]
        L += ["", "### A3/A4 derived layers: HT-AP by base spectral layer", "",
              "rowz = genome-wide row-z balancing; mr / mrbal = mutual rank on raw / balanced similarity; "
              "so{cos,jaccard}_q = second-order neighbour overlap with the top q% background genes (balanced). "
              "Δ columns are not shown; compare with `base`.", "",
              "| base | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for b in bases:
            cells = [f3(by[b]["ap_ht"]) if b in by else "–"]
            for c in cols[1:]:
                if c == "rowz":
                    n = f"rowz[{b}]"
                elif c in ("mr", "mrbal"):
                    n = f"{c}[{b}]"
                else:
                    n = f"{c}[{b}]"
                cells.append(f3(by[n]["ap_ht"]) if n in by else "–")
            L.append(f"| {b} | " + " | ".join(cells) + " |")
        other = [r for r in rows if r["config"]["kind"] != "spectral" and not re.match(r"^(rowz|mr|mrbal|so(cos|jaccard)_q[0-9.]+)\[spec_[^\[\]]+\]$",
                                                      r["name"])]
        if other:
            L += ["", "Other derived layers (row-z of second-order, screen_v2):", "", "| layer | HT-AP | pooled AP | second-order input HT-AP |", "|---|---|---|---|"]
            for r in sorted(other, key=lambda r: -r["ap_ht"]):
                inner = r["name"][5:-1] if r["name"].startswith("rowz[") else None
                L.append(f"| {r['name']} | {f3(r['ap_ht'])} | {f3(r['pooled_ap_raw'])} | "
                         f"{f3(by[inner]['ap_ht']) if inner in by else '–'} |")
    return L


NOTES_FILE = OUT / "screen_notes.md"


def write_readme():
    d = json.loads((OUT / "screen.json").read_text())
    rows = sorted(d["layers"], key=lambda r: -r["ap_ht"])
    head = ("| rank | layer | cache | HT-AP mean ± sd | pooled AP | HT AUROC | AP both non-ess. | AP mixed | AP both ess. | "
            "macro-cluster AP | AP w/o largest | pooled HT-AP |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"

    def line(rank, r):
        return (f"| {rank} | {r['name']} | {r['cache']} | {r['ap_ht']:.3f} ± {r['ap_ht_sd']:.3f} | {f3(r['pooled_ap_raw'])} | "
                f"{f3(r['auroc_ht'])} | {f3(r['ap_ht_both_nonessential'])} | {f3(r['ap_ht_mixed'])} | "
                f"{f3(r['ap_ht_both_essential'])} | {f3(r['macro_cluster_ap'])} | {f3(r['ap_ht_no_largest_component'])} | "
                f"{f3(r['pooled_ap_ht'])} |")

    L = ["# Phase A — label-free layer screen (dev CV)", "",
         "Generated by `scripts/11_screen_layers.py` (tables; do not edit by hand). DEV pairs only, §3.3 cluster-grouped "
         f"5-fold CV, SEEDS[0] (split fingerprint `{d['split_fingerprint'][:12]}`, identical to the baselines). "
         f"{d['n_layers']} layers from caches {', '.join(d['caches'])}. Primary = HT-weighted AP, mean ± sd over the 5 "
         "test folds. `pooled AP` = AP over all 322,528 dev pairs at natural prevalence; HT AUROC, essentiality-stratum "
         "APs, macro per-cluster AP and AP without the largest component are fold means (HT-weighted); `pooled HT-AP` "
         "= HT AP over all dev pairs (between-cluster weight 1/f). Per-layer JSON: `results/phaseA/screen/`; each layer "
         "is one logged trial in `results/trials.jsonl`.", "",
         "## Baselines (best dev variants, reference)", "", "| rank | layer | cache | HT-AP mean ± sd | pooled AP | HT AUROC | "
         "AP both non-ess. | AP mixed | AP both ess. | macro-cluster AP | AP w/o largest | pooled HT-AP |", sep]
    for b in d["baselines"]:
        L.append(line("ref", b))
    L += ["", "## All layers sorted by primary", "", head, sep]
    for i, r in enumerate(rows, 1):
        L.append(line(i, r))
    L += ["", "## Trend tables (HT-AP, fold mean)", ""] + trend_tables(d["layers"])
    if NOTES_FILE.exists():
        L += ["", NOTES_FILE.read_text().strip()]
    (OUT / "README.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="screen_v1")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--readme-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.readme_only:
        run([t for t in a.tags.split(",") if t], a.allow_dirty)
    write_readme()


if __name__ == "__main__":
    main()
