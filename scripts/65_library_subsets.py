"""Post-hoc check (after the report and the single IFT look): does restricting the input to Avana-screened cell lines
help the FINAL method? Dev-only; IFT pairs are never loaded here.

The round-1 data review (docs/reviews/round1_data_biology.md) answered this only with Pearson and a single pass.
Here zavg_BEF and the GLS baseline are recomputed from scratch on three row subsets of CRISPRGeneEffect (26Q3):
  all         all 1,361 models (the published input)
  avana_flag  models whose CRISPRConfounders library one-hot includes Avana (1,048; includes multi-library models)
  avana_only  models screened with Avana and no other library (827)
Library flags come from src.data.model_metadata. zavg_BEF uses the ported implementation (mvp/src/coessential/method.py,
frozen Config: k = 20, tau = 0.1, q = 0.25%, d = 512); BG genes, the covariance, neighbour sets, graph and the 200k
null are all re-estimated on each subset (K = round(0.25% x |BG|)). GLS: scorers.make_baseline('gls_signed') on the
subset rows (sample covariance, NaN marginalised, as for the published baseline).

Genes with no observation in a subset (non-Avana-library genes in avana_only) cannot be scored; their pairs get the
subset's minimum score. Two evaluations:
  native   each subset scores what it can (the real cost of restricting)
  matched  pairs touching any gene unscorable in some subset get the minimum score in EVERY subset, so the three
           inputs are compared on the same scorable pairs
Each (subset, scorer, evaluation) runs src.evaluate.run_dev_cv over SEEDS (cluster scheme; logged trials), plus a
paired cluster bootstrap (1,000, seed 0) of pooled dev HT-AP differences vs the 'all' input.

Output: results/library_subsets/{summary.json, runs/*.json}; dev scores cached in data_cache/library_subsets/.
Usage: ./.env/bin/python scripts/65_library_subsets.py [--stages score,eval]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mvp" / "src"))
import numpy as np  # noqa: E402

from src import evaluate, metrics, scorers  # noqa: E402
from src.config import CACHE, RESULTS, SEEDS  # noqa: E402
from src.data import load_matrix, model_metadata  # noqa: E402
from src.universe import load_dev_universe  # noqa: E402
from coessential.method import Config, CoessentialityModel  # noqa: E402

OUT = RESULTS / "library_subsets"
SCORES = CACHE / "library_subsets"
SUBSETS = ("all", "avana_flag", "avana_only")
SCORERS = ("zavg_BEF", "gls_signed")
N_BOOT = 1000


def subset_rows(samples):
    lib = model_metadata(samples).libraries
    return {"all": np.ones(len(lib), bool), "avana_flag": lib.str.contains("Avana").values,
            "avana_only": (lib == "Avana").values}


def score_stage(u):
    X, samples, genes = load_matrix("CRISPRGeneEffect", mmap=True)
    ia, ib = u.col("ia"), u.col("ib")
    SCORES.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, rows in subset_rows(samples).items():
        t0 = time.perf_counter()
        Xs = np.asarray(X[rows], dtype=np.float64)
        keep = ~np.isnan(Xs).all(0)
        ok = keep[ia] & keep[ib]
        col = np.cumsum(keep) - 1                      # full gene index -> column of Xs[:, keep]
        model = CoessentialityModel(Xs[:, keep], Config())
        z, _ = model.zavg(col[ia[ok]], col[ib[ok]])
        zavg = np.full(len(u), np.nan)
        zavg[ok] = z
        gls = scorers.make_baseline("gls_signed", scorers._Matrix(Xs, genes)).bind(u).score(np.arange(len(u)))
        np.savez(SCORES / f"{name}.npz", zavg_BEF=zavg, gls_signed=gls, scorable=ok)
        info[name] = dict(n_models=int(rows.sum()), n_genes_unobserved=int((~keep).sum()), n_bg=int(model.n_bg),
                          K=int(model.K), n_dev_pairs_unscorable=int((~ok).sum()),
                          n_dev_pos_unscorable=int(u.y[~ok].sum()), gls_nan=int(np.isnan(gls).sum()),
                          runtime_s=round(time.perf_counter() - t0, 1))
        print(name, info[name], flush=True)
        del model, Xs
    (SCORES / "info.json").write_text(json.dumps(info, indent=1))
    return info


def filled(s, bad):
    s = s.copy()
    s[bad | np.isnan(s)] = np.nanmin(s[~bad])
    return s


def eval_stage(u, allow_dirty):
    info = json.loads((SCORES / "info.json").read_text())
    S = {n: dict(np.load(SCORES / f"{n}.npz")) for n in SUBSETS}
    unscorable_any = ~np.logical_and.reduce([S[n]["scorable"] for n in SUBSETS])
    pid = u.col("pair_id")
    out = dict(subsets=info, n_dev_pairs=len(u), n_dev_pos=int(u.y.sum()),
               matched_excluded_pairs=int(unscorable_any.sum()), matched_excluded_pos=int(u.y[unscorable_any].sum()),
               runs={}, bootstrap={})
    for ev in ("native", "matched"):
        boot_scores = {}
        for sc in SCORERS:
            for n in SUBSETS:
                bad = ~S[n]["scorable"] if ev == "native" else unscorable_any
                s = filled(S[n][sc], bad)
                boot_scores[f"{sc}|{n}"] = s
                arr = np.full(int(pid.max()) + 1, np.nan)
                arr[pid] = s
                key = f"{sc}|{n}|{ev}"
                folds = []
                for seed in SEEDS:
                    r = evaluate.run_dev_cv(scorers.PrecomputedScorer(f"{sc}_{n}_{ev}", arr), seed=seed,
                                            universe=u, out_dir=OUT / "runs", allow_dirty=allow_dirty,
                                            extra=dict(script="65_library_subsets", subset=n, evaluation=ev))
                    folds.append(r["summary"])
                out["runs"][key] = {m: dict(mean=float(np.mean([f[m]["mean"] for f in folds])),
                                            sd_over_seeds=float(np.std([f[m]["mean"] for f in folds], ddof=1)))
                                    for m in ("ap_ht", "macro_cluster_ap", "auroc_ht")}
                print(key, out["runs"][key]["ap_ht"], flush=True)
        diffs = [(f"{sc}|{n}", f"{sc}|all") for sc in SCORERS for n in SUBSETS[1:]]
        diffs += [(f"zavg_BEF|{n}", f"gls_signed|{n}") for n in SUBSETS]
        b = metrics.cluster_bootstrap(u.y, boot_scores, u.col("cluster_a"), u.col("cluster_b"), f=u.f,
                                      n_boot=N_BOOT, seed=SEEDS[0], diffs=diffs)
        out["bootstrap"][ev] = dict(pooled_ap_ht={k: v for k, v in b["ap"].items()}, diff=b["diff"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["bootstrap"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="score,eval")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    u = load_dev_universe()
    st = a.stages.split(",")
    if "score" in st:
        score_stage(u)
    if "eval" in st:
        eval_stage(u, a.allow_dirty)


if __name__ == "__main__":
    main()
