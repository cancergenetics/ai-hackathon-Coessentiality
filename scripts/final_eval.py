"""The ONLY entry point that touches internal-final-test (IFT) pairs (DESIGN v3.1 §3.2, §3.7, §6).

Every real run is a "look": it is appended to results/ift_looks.jsonl BEFORE scoring (status 'started') and again
when finished, and writes results/ift/look<NN>_<date>.json. At most 2 named candidates, given in their pre-registered
fixed-sequence order (first = primary; RESEARCH_TREE D3 / results/phaseB_v2/decision.json), compared against each
baseline's dev-selected best variant (results/baselines_v2/best_variants.json).

Per scorer: bind to the dev+IFT universe, fit(dev pair idx) (no-op for label-free), score(IFT pair idx).
Metrics on IFT: metrics.metric_suite with HT weights (within-cluster 1, between-cluster 1/f, f = realised fraction of
scorable proteins in IFT); paired cluster bootstrap (1,000 resamples, SEEDS[0]) of each candidate vs each baseline
(CIs and p-values are reported as information, they are not part of the gate).

Verdict (DESIGN v3.1 §3.7, IFT parts), per candidate c and baseline b:
  gate_ii_ift  = relative HT-AP gain AP_c / AP_b - 1 >= 0.10 (point estimate; IFT CI reported, not required)
  non-inferiority (iii): macro per-cluster AP and HT-AP without the largest component, each c >= 0.95 x b
                 (undefined / non-finite values fail)
  passes(c, b) = gate_ii_ift and both non-inferiority checks.
  Intersection-union: c clearly beats the baselines on the IFT iff it passes vs every baseline. The dev parts
  (i) and (ii-dev) are not recomputed here; they are recorded by reference (paths + SHA-256 of
  results/phaseB_v2/decision.json and compare.json, and the stored pooled-OOF CIs read from compare.json).
  Fixed sequence (no Holm): the primary's verdict is confirmatory; the secondary's is confirmatory only if the primary
  clearly beats all baselines, otherwise exploratory. Secondary - primary ΔHT-AP (bootstrap) is reported as info.
§3.10: HT AP also reported with the unscorable IFT pairs included at the minimum score.

--dry-run prints IFT pair/cluster counts only (no scores, no metrics, nothing appended); with --candidates it also
prints the fixed-sequence plan and its agreement with the D3 decision file.
The scoring logic (``run_final_eval``) takes the universe as an argument so it is tested on a toy universe only
(tests/test_final_eval.py).

Usage: ./.env/bin/python scripts/final_eval.py --candidates PRIMARY [SECONDARY] --what "why this look" [--allow-dirty]
       ./.env/bin/python scripts/final_eval.py --dry-run [--candidates PRIMARY [SECONDARY]]
       ./.env/bin/python scripts/final_eval.py --export-curves results/ift/lookNN_<date>.json --what "why"
         (re-plot of an existing look: same scorers, no selection; logged as look "<NN>-figure" before scoring;
          writes results/ift/lookNN_<date>_curves.json and asserts curve AP == stored HT-AP)
       ./.env/bin/python scripts/final_eval.py --posthoc-reference NAME --what "why"
         (POST-HOC reference baseline requested in manuscript review, NOT a candidate: NAME is one of
          POSTHOC_REFERENCES (PC-removed Pearson layers of the screen_v1 cache); logged as look "2-posthoc" before
          scoring; metric_suite on the IFT for NAME and the final method zavg_BEF + paired cluster bootstrap
          Δ(NAME − zavg_BEF); no verdict logic, no selection; writes results/revision/ift_posthoc_<NAME>.json)
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from src import metrics  # noqa: E402
from src.config import RESULTS, SEEDS  # noqa: E402
from src.evaluate import _json_default, check_clean, run_record  # noqa: E402

MARGIN = 0.10                     # §3.7(ii) IFT: relative HT-AP gain point estimate >= 10%
NI_RATIO = 0.95                   # §3.7(iii): non-inferiority margin -5% relative
NI_METRICS = ("macro_cluster_ap", "ap_ht_no_largest_component")
MAX_CANDIDATES = 2
LOOKS = RESULTS / "ift_looks.jsonl"
OUT = RESULTS / "ift"
BEST = RESULTS / "baselines_v2" / "best_variants.json"
DEV_FILES = {"decision": RESULTS / "phaseB_v2" / "decision.json", "compare": RESULTS / "phaseB_v2" / "compare.json",
             "readme": RESULTS / "phaseB_v2" / "README.md"}
POSTHOC_LOOK = "2-posthoc"
POSTHOC_PRIMARY = "zavg_BEF"      # final method (D4); the reference is compared against it, never selected
POSTHOC_CACHE = "screen_v1"       # results/layers/manifest.json
POSTHOC_REFERENCES = ("spec_k5_tinf", "spec_k20_tinf", "spec_k5_tinf_std", "spec_k20_tinf_std")   # review B1
POSTHOC_OUT = RESULTS / "revision"
DEV_NAME = {"zavg_BEF": "zavg_BEF", "B": "ref_B", "gls_signed": "ref_G", "spearman_signed": "ref_SP",
            "pearson_signed": "ref_PE"}          # IFT scorer name -> Phase B v2 entry name


def _append(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, default=_json_default) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def next_look(path: Path) -> int:
    if not path.exists():
        return 0
    looks = [json.loads(l)["look"] for l in path.read_text().splitlines() if l.strip()]
    looks = [x for x in looks if isinstance(x, int)]  # verification/figure re-plots ("1-verify") are not new looks
    return max(looks) + 1 if looks else 0


def ift_counts(universe) -> dict:
    side = universe.col("side")
    y = universe.y.astype(bool)
    ift = side == "ift"
    return dict(ift_pos=int((y & ift).sum()), ift_neg=int((~y & ift).sum()), dev_pos=int((y & ~ift).sum()),
                dev_neg=int((~y & ~ift).sum()),
                ift_clusters=int(len(set(universe.col("cluster_a")[ift]) | set(universe.col("cluster_b")[ift]))),
                f_ift=universe.meta.get("f_ift"))


def non_inferior(cand, base, ratio: float = NI_RATIO) -> dict:
    """§3.7(iii): cand >= ratio * base (both finite; undefined values fail)."""
    ok = cand is not None and base is not None and bool(np.isfinite(cand)) and bool(np.isfinite(base))
    return dict(candidate=cand, baseline=base, threshold=ratio * base if ok else None,
                rel=(cand / base - 1) if ok and base > 0 else None, passed=bool(ok and cand >= ratio * base))


def decide(per: dict, boot: dict, candidates: list, baselines: list) -> dict:
    """DESIGN v3.1 §3.7 IFT verdict. candidates in fixed-sequence order (first = primary)."""
    pairs, cands = {}, {}
    for c in candidates:
        for b in baselines:
            d = boot["diff"][f"{c}-{b}"]
            rel = d["rel_gain"]
            gate = bool(rel is not None and np.isfinite(rel) and rel >= MARGIN)
            ni = {m: non_inferior(per[c].get(m), per[b].get(m)) for m in NI_METRICS}
            pairs[f"{c}-{b}"] = dict(
                candidate=c, baseline=b, ap_ht_candidate=per[c]["ap_ht"], ap_ht_baseline=per[b]["ap_ht"],
                diff=d["diff"], rel_gain=rel, gate_ii_ift=gate, non_inferiority=ni,
                passes=bool(gate and all(v["passed"] for v in ni.values())),
                info=dict(ci_low=d["ci_low"], ci_high=d["ci_high"], rel_ci_low=d["rel_ci_low"],
                          rel_ci_high=d["rel_ci_high"], p_two_sided=d["p_two_sided"]))
    primary_ok = None
    for i, c in enumerate(candidates):
        beats = all(pairs[f"{c}-{b}"]["passes"] for b in baselines)
        if i == 0:
            primary_ok = beats
            status = "confirmatory"
        else:
            status = "confirmatory" if primary_ok else "exploratory"
        cands[c] = dict(order=i + 1, role="primary" if i == 0 else "secondary", status=status,
                        passes_vs={b: pairs[f"{c}-{b}"]["passes"] for b in baselines},
                        clearly_beats_ift=bool(beats), confirmed=bool(beats and status == "confirmatory"))
    return dict(rule="DESIGN v3.1 §3.7: per baseline gate_ii_ift (rel HT-AP gain >= %.2f, point estimate) and "
                     "non-inferiority (%s >= %.2f x baseline); intersection-union over baselines; dev parts (i), "
                     "(ii-dev) by reference; fixed sequence, no Holm" % (MARGIN, " and ".join(NI_METRICS), NI_RATIO),
                fixed_sequence=list(candidates), pairs=pairs, candidates=cands)


def dev_reference(files: dict = DEV_FILES, candidates=(), baselines=()) -> dict:
    """Dev evidence for §3.7 (i), (ii-dev): file paths + SHA-256, and stored pooled-OOF Δ CIs read (not recomputed)
    from compare.json for each candidate x baseline."""
    out = dict(note="dev parts (i) and (ii-dev) of §3.7 are taken from Phase B v2 (5 seeds, pooled-OOF paired cluster "
                    "bootstrap); not recomputed here", files={})
    for k, p in files.items():
        p = Path(p)
        out["files"][k] = dict(path=str(p.relative_to(RESULTS.parent)) if p.is_relative_to(RESULTS.parent) else str(p),
                               sha256=sha256_file(p) if p.exists() else None)
    dec = Path(files.get("decision", "")) if files.get("decision") else None
    if dec is not None and dec.exists():
        d = json.loads(dec.read_text())
        out["d3"] = dict(primary=d.get("primary"), secondary=d.get("secondary"), fixed_sequence=d.get("fixed_sequence"))
    cmp_ = Path(files.get("compare", "")) if files.get("compare") else None
    if cmp_ is not None and cmp_.exists():
        cmp = json.loads(cmp_.read_text())
        read = {}
        for c in candidates:
            for b in baselines:
                key = f"{DEV_NAME.get(c, c)}-{DEV_NAME.get(b, b)}"
                per_seed = {s: {k: v["all"]["diff"][key][k] for k in ("diff", "ci_low", "ci_high")}
                            for s, v in cmp["seeds"].items() if key in v["all"]["diff"]}
                read[f"{c}-{b}"] = dict(compare_key=key, per_seed=per_seed or None)
        out["pooled_oof_delta_vs_baselines"] = read
    return out


def run_final_eval(universe, candidates: dict, baselines: dict, out_dir: Path, looks_file: Path, what: str,
                   unscorable=None, n_boot: int = 1000, allow_dirty: bool = False, dev_ref: dict | None = None) -> dict:
    """universe: dev+IFT PairUniverse (side column, meta['f_ift']); candidates (fixed-sequence order, first =
    primary) / baselines: name -> unbound scorer."""
    if not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError(f"1..{MAX_CANDIDATES} candidates required")
    if set(candidates) & set(baselines):
        raise ValueError("a candidate cannot also be a baseline")
    if not baselines:
        raise ValueError("at least one baseline required")
    order = list(candidates)
    look = next_look(looks_file)
    date = dt.date.today().isoformat()
    head = dict(look=look, date=date, what=what, candidates=order, baselines=sorted(baselines),
                status="started", allow_dirty=allow_dirty)
    _append(looks_file, head)
    t0 = time.perf_counter()
    side = universe.col("side")
    dev_idx, ift_idx = np.flatnonzero(side == "dev"), np.flatnonzero(side == "ift")
    y = universe.y[ift_idx].astype(bool)
    ca, cb = universe.col("cluster_a")[ift_idx], universe.col("cluster_b")[ift_idx]
    f = float(universe.meta["f_ift"])
    w = metrics.ht_weights(ca, cb, f)
    scores, per, runtime = {}, {}, {}
    for name, sc in {**candidates, **baselines}.items():
        t1 = time.perf_counter()
        sc.bind(universe)
        sc.fit(dev_idx)
        s = np.asarray(sc.score(ift_idx), dtype=np.float64)
        runtime[name] = round(time.perf_counter() - t1, 3)
        if np.isnan(s).any():
            raise ValueError(f"{name}: NaN scores on IFT")
        scores[name] = s
        u = universe
        per[name] = metrics.metric_suite(
            y, s, w, ca, cb, u.col("comp_a")[ift_idx], u.col("comp_b")[ift_idx],
            u.col("ess_a")[ift_idx] if "ess_a" in u.pairs else None,
            u.col("ess_b")[ift_idx] if "ess_b" in u.pairs else None, largest_component=u.largest_component)
        if unscorable is not None and len(unscorable):
            yu = unscorable.label.values.astype(bool)
            fill = np.full(len(yu), s.min() - 1.0)
            per[name]["ap_ht_with_unmapped_min_score"] = metrics.weighted_ap(
                np.r_[y, yu], np.r_[s, fill], np.r_[w, np.full(len(yu), 1.0 / f)])
            per[name]["n_unmapped_pos"] = int(yu.sum())
            per[name]["n_unmapped_neg"] = int((~yu).sum())
    diffs = [(c, b) for c in order for b in baselines] + [(c, order[0]) for c in order[1:]]
    boot = metrics.cluster_bootstrap(y, scores, ca, cb, f=f, n_boot=n_boot, seed=SEEDS[0], diffs=diffs)
    verdict = decide(per, boot, order, list(baselines))
    verdict["candidate_vs_primary_info"] = {f"{c}-{order[0]}": boot["diff"][f"{c}-{order[0]}"] for c in order[1:]}
    res = dict(look=look, date=date, what=what, ht_f=f, counts=ift_counts(universe), metrics=per, bootstrap=boot,
               verdict=verdict, dev_reference=dev_ref, runtime_s=runtime,
               scorers={k: v.describe() for k, v in {**candidates, **baselines}.items()})
    res["run_record"] = run_record(seed=SEEDS[0], counts=res["counts"], runtime_s=round(time.perf_counter() - t0, 2),
                                   allow_dirty=allow_dirty)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"look{look:02d}_{date}.json"
    path.write_text(json.dumps(res, indent=1, default=_json_default))
    _append(looks_file, dict(head, status="done", result=path.name, git_commit=res["run_record"]["git_commit"],
                             verdict={k: v["passes"] for k, v in verdict["pairs"].items()}))
    return res


def ht_pr_curve(y, s, w) -> dict:
    """HT-weighted precision-recall curve with sklearn tie semantics (one point per distinct score; only the points
    where recall increases are kept, which is exactly the set entering AP = sum dR * P)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    ys, ss, ws = y[order], s[order], w[order]
    last = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    tp = np.cumsum(ws * ys)[last]
    tot = np.cumsum(ws)[last]
    rec, prec = tp / tp[-1], tp / tot
    drec = np.diff(np.r_[0.0, rec])
    keep = drec > 0
    return dict(recall=rec[keep].tolist(), precision=prec[keep].tolist(), ap_from_curve=float(np.sum(drec * prec)))


def export_curves(universe, scorers_: dict, look_file: Path, looks_file: Path, out_path: Path, what: str,
                  tol: float = 1e-9) -> dict:
    """Re-plot of an EXISTING look (no new candidates, no selection): HT-weighted PR curves on the IFT pairs for the
    scorers of ``look_file``. Appends {"look": "<N>-figure", ...} to the look log BEFORE scoring; asserts that the AP
    integrated from each curve equals the look's stored HT-AP."""
    ref = json.loads(Path(look_file).read_text())
    if set(scorers_) != set(ref["scorers"]):
        raise ValueError(f"scorers {sorted(scorers_)} differ from look {ref['look']}: {sorted(ref['scorers'])}")
    _append(looks_file, dict(look=f"{ref['look']}-figure", date=dt.date.today().isoformat(), what=what))
    side = universe.col("side")
    dev_idx, ift_idx = np.flatnonzero(side == "dev"), np.flatnonzero(side == "ift")
    y = universe.y[ift_idx].astype(bool)
    f = float(universe.meta["f_ift"])
    if abs(f - ref["ht_f"]) > 1e-12:
        raise ValueError("IFT fraction differs from the look")
    w = metrics.ht_weights(universe.col("cluster_a")[ift_idx], universe.col("cluster_b")[ift_idx], f)
    out = dict(look=ref["look"], look_file=Path(look_file).name, look_sha256=sha256_file(look_file), what=what,
               ht_f=f, n_pos=int(y.sum()), n_neg=int((~y).sum()), ht_prevalence=float(w[y].sum() / w.sum()),
               curve_note="HT-weighted (within-cluster 1, between-cluster 1/f); one point per distinct score where "
                          "recall increases (sklearn tie semantics); AP = sum dR * P", curves={})
    for name, sc in scorers_.items():
        sc.bind(universe)
        sc.fit(dev_idx)
        s = np.asarray(sc.score(ift_idx), dtype=np.float64)
        c = ht_pr_curve(y, s, w)
        c["ap_look"] = float(ref["metrics"][name]["ap_ht"])
        c["ap_weighted_ap"] = metrics.weighted_ap(y, s, w)
        if abs(c["ap_from_curve"] - c["ap_look"]) > tol or abs(c["ap_weighted_ap"] - c["ap_look"]) > tol:
            raise AssertionError(f"{name}: curve AP {c['ap_from_curve']} != look AP {c['ap_look']}")
        out["curves"][name] = c
    out["run_record"] = run_record(seed=SEEDS[0], counts=dict(n_pos=out["n_pos"], n_neg=out["n_neg"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=_json_default))
    return out


def posthoc_reference(universe, reference: dict, primary: dict, looks_file: Path, out_path: Path, what: str,
                      look_label: str = POSTHOC_LOOK, n_boot: int = 1000) -> dict:
    """POST-HOC reference baseline on the IFT (not a candidate; no verdict, no selection). reference / primary: one
    {name: unbound scorer} each. Appends {"look": look_label, "date", "what", "reference"} to the look log BEFORE
    scoring; returns metric_suite for both and the paired cluster bootstrap (SEEDS[0]) of Δ(reference − primary)."""
    if len(reference) != 1 or len(primary) != 1 or set(reference) & set(primary):
        raise ValueError("exactly one reference and one (different) primary scorer required")
    (rn, _), (pn, _) = next(iter(reference.items())), next(iter(primary.items()))
    if looks_file.exists() and any(json.loads(l).get("look") == look_label
                                   for l in looks_file.read_text().splitlines() if l.strip()):
        raise RuntimeError(f"look {look_label!r} already logged; a post-hoc reference is evaluated once")
    _append(looks_file, dict(look=look_label, date=dt.date.today().isoformat(), what=what, reference=rn))
    t0 = time.perf_counter()
    side = universe.col("side")
    dev_idx, ift_idx = np.flatnonzero(side == "dev"), np.flatnonzero(side == "ift")
    y = universe.y[ift_idx].astype(bool)
    ca, cb = universe.col("cluster_a")[ift_idx], universe.col("cluster_b")[ift_idx]
    f = float(universe.meta["f_ift"])
    w = metrics.ht_weights(ca, cb, f)
    u = universe
    scores, per = {}, {}
    for name, sc in {**reference, **primary}.items():
        sc.bind(universe)
        sc.fit(dev_idx)
        s = np.asarray(sc.score(ift_idx), dtype=np.float64)
        if np.isnan(s).any():
            raise ValueError(f"{name}: NaN scores on IFT")
        scores[name] = s
        per[name] = metrics.metric_suite(
            y, s, w, ca, cb, u.col("comp_a")[ift_idx], u.col("comp_b")[ift_idx],
            u.col("ess_a")[ift_idx] if "ess_a" in u.pairs else None,
            u.col("ess_b")[ift_idx] if "ess_b" in u.pairs else None, largest_component=u.largest_component)
    boot = metrics.cluster_bootstrap(y, scores, ca, cb, f=f, n_boot=n_boot, seed=SEEDS[0], diffs=[(rn, pn)])
    res = dict(look=look_label, date=dt.date.today().isoformat(), what=what, posthoc=True, reference=rn, primary=pn,
               note="post-hoc reference baseline requested in manuscript review; not a candidate; no verdict logic; "
                    "no selection of the final method", ht_f=f, counts=ift_counts(universe), metrics=per,
               bootstrap=boot, scorers={k: v.describe() for k, v in {**reference, **primary}.items()})
    res["run_record"] = run_record(seed=SEEDS[0], counts=res["counts"], runtime_s=round(time.perf_counter() - t0, 2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=_json_default))
    return res


def posthoc_scorer(name: str):
    """Cached PC-removed Pearson layer (label-free, computed on all scorable pairs before any split)."""
    from src import scorers
    from src.supervised import load_layer_cache
    if name not in POSTHOC_REFERENCES:
        raise SystemExit(f"--posthoc-reference must be one of {POSTHOC_REFERENCES}")
    M, _, cols = load_layer_cache(POSTHOC_CACHE)
    c = cols[name]
    if c["config"]["kind"] != "spectral" or c["config"]["tau"] != "inf":
        raise SystemExit(f"{name} is not a PC-removed Pearson layer: {c['config']}")
    return scorers.PrecomputedScorer(name, np.asarray(M[:, c["col"]], dtype=np.float64),
                                     params=dict(cache=POSTHOC_CACHE, col=c["col"], config=c["config"],
                                                 role="post-hoc reference (review B1), not a candidate"))


def check_sequence(candidates: list, decision_file: Path = DEV_FILES["decision"]) -> dict:
    """Candidates must be the D3 fixed sequence (or its primary alone), in that order."""
    d = json.loads(Path(decision_file).read_text())
    want = list(d["fixed_sequence"])
    got = [DEV_NAME.get(c, c) for c in candidates]
    return dict(d3_fixed_sequence=want, given=list(candidates), given_as_dev_names=got,
                consistent=bool(got == want[:len(got)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", default=[], help="fixed-sequence order: primary first")
    ap.add_argument("--what", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--export-curves", metavar="LOOK_JSON", default=None,
                    help="re-plot of an existing look: HT PR curves for its scorers -> <look>_curves.json")
    ap.add_argument("--posthoc-reference", metavar="NAME", default=None,
                    help=f"post-hoc reference baseline vs {POSTHOC_PRIMARY} (no verdict); one of {POSTHOC_REFERENCES}")
    a = ap.parse_args()
    from src.ift import load_ift_universe            # the only permitted import of the IFT loader
    from src import scorers
    if a.export_curves:
        if not a.what:
            raise SystemExit("--what is required for --export-curves")
        look_file = Path(a.export_curves)
        ref = json.loads(look_file.read_text())
        check_clean(a.allow_dirty)
        u, _ = load_ift_universe()
        mat = scorers.real_matrix()
        scs = {n: scorers.make_scorer(n, mat) for n in ref["scorers"]}
        out_path = look_file.with_name(look_file.stem + "_curves.json")
        res = export_curves(u, scs, look_file, LOOKS, out_path, a.what)
        print(json.dumps({k: dict(ap_from_curve=v["ap_from_curve"], ap_look=v["ap_look"], n_points=len(v["recall"]))
                          for k, v in res["curves"].items()}, indent=1))
        return
    if a.posthoc_reference:
        if not a.what:
            raise SystemExit("--what is required for --posthoc-reference")
        ref_sc = posthoc_scorer(a.posthoc_reference)
        check_clean(a.allow_dirty)
        u, _ = load_ift_universe()
        out_path = POSTHOC_OUT / f"ift_posthoc_{a.posthoc_reference}.json"
        res = posthoc_reference(u, {a.posthoc_reference: ref_sc}, {POSTHOC_PRIMARY: scorers.make_scorer(POSTHOC_PRIMARY)},
                                LOOKS, out_path, a.what, n_boot=a.n_boot)
        look1 = sorted(OUT.glob("look01_*.json"))
        if look1:                                   # consistency with the pre-registered look (same scorer, same pairs)
            ref1 = json.loads(look1[0].read_text())["metrics"][POSTHOC_PRIMARY]["ap_ht"]
            res["primary_ap_ht_look01"] = ref1
            res["primary_ap_ht_abs_diff_vs_look01"] = abs(res["metrics"][POSTHOC_PRIMARY]["ap_ht"] - ref1)
            out_path.write_text(json.dumps(res, indent=1, default=_json_default))
        d = res["bootstrap"]["diff"][f"{a.posthoc_reference}-{POSTHOC_PRIMARY}"]
        print(json.dumps(dict(ap_ht={k: v["ap_ht"] for k, v in res["metrics"].items()},
                              macro={k: v.get("macro_cluster_ap") for k, v in res["metrics"].items()},
                              delta=d, out=str(out_path)), indent=1, default=_json_default))
        return
    if len(set(a.candidates)) != len(a.candidates):
        raise SystemExit("duplicate candidate names")
    unknown = [n for n in a.candidates if n not in scorers.CANDIDATES and n not in scorers.BASELINE_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown candidates {unknown}; registered: {sorted(scorers.CANDIDATES)}")
    if a.dry_run:
        u, un = load_ift_universe()
        c = ift_counts(u)
        c.update(unscorable_ift_pos=int(un.label.sum()), unscorable_ift_neg=int((un.label == 0).sum()))
        if a.candidates:
            c["plan"] = dict(fixed_sequence=a.candidates,
                             baselines=sorted(json.loads(BEST.read_text())["best_variants"].values()),
                             d3=check_sequence(a.candidates))
        print(json.dumps(c, indent=1))
        return
    if not a.what:
        raise SystemExit("--what is required for a real look")
    if not 1 <= len(a.candidates) <= MAX_CANDIDATES:
        raise SystemExit(f"1..{MAX_CANDIDATES} candidates required")
    seq = check_sequence(a.candidates)
    if not seq["consistent"]:
        raise SystemExit(f"candidates are not the pre-registered D3 fixed sequence: {seq}")
    check_clean(a.allow_dirty)
    best = json.loads(BEST.read_text())["best_variants"]
    u, un = load_ift_universe()
    mat = scorers.real_matrix()
    cands = {n: scorers.make_scorer(n, mat) for n in a.candidates}
    bases = {n: scorers.make_scorer(n, mat) for n in best.values()}
    dref = dev_reference(candidates=a.candidates, baselines=list(bases))
    dref["d3_sequence_check"] = seq
    res = run_final_eval(u, cands, bases, OUT, LOOKS, a.what, unscorable=un, n_boot=a.n_boot,
                         allow_dirty=a.allow_dirty, dev_ref=dref)
    print(json.dumps(dict(candidates=res["verdict"]["candidates"],
                          pairs={k: {kk: v[kk] for kk in ("rel_gain", "gate_ii_ift", "passes")}
                                 for k, v in res["verdict"]["pairs"].items()}), indent=1))


if __name__ == "__main__":
    main()
