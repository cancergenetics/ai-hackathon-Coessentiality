"""scripts/final_eval.py exercised on a TOY universe only (never real IFT); DESIGN v3.1 §3.7 verdict logic on synthetic
metric dicts; frozen IFT candidate scorers (zavg_BEF, B) reproduce Phase B v2 / predict_pairs values on dev pairs
(real label-free caches, no labels used, no metrics; skipped without caches)."""
import importlib.util
import json

import numpy as np
import pytest

from src import scorers
from src.config import CACHE, RESULTS, ROOT, SEEDS
from src.toy import make_toy


def _load():
    spec = importlib.util.spec_from_file_location("final_eval", ROOT / "scripts" / "final_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fe = _load()


@pytest.fixture(scope="module")
def toy():
    u, X, _ = make_toy(seed=SEEDS[1])
    return u, X


# ------------------------------------------------------------------------------------------ end-to-end (toy)
def test_final_eval_toy(toy, tmp_path):
    u, X = toy
    mat = scorers._Matrix(X)
    looks = tmp_path / "looks.jsonl"
    looks.write_text(json.dumps(dict(look=0, date="x", what="seed")) + "\n")
    order = ["gls_signed", "gls_abs"]                       # fixed sequence, deliberately not sorted
    cands = {n: scorers.make_baseline(n, mat) for n in order}
    bases = {"pearson_signed": scorers.make_baseline("pearson_signed", mat),
             "spearman_abs": scorers.make_baseline("spearman_abs", mat)}
    import pandas as pd
    unsc = pd.DataFrame(dict(label=[1, 0, 0]))
    dref = dict(note="toy")
    res = fe.run_final_eval(u, cands, bases, tmp_path / "ift", looks, "toy test", unscorable=unsc, n_boot=100,
                            dev_ref=dref)
    recs = [json.loads(l) for l in looks.read_text().splitlines()]
    assert [r["look"] for r in recs] == [0, 1, 1] and [r.get("status") for r in recs[1:]] == ["started", "done"]
    assert recs[1]["candidates"] == order                    # pre-declared order logged before scoring
    assert (tmp_path / "ift" / recs[2]["result"]).exists()
    assert res["counts"]["ift_pos"] == int(u.y[u.col("side") == "ift"].sum())
    v = res["verdict"]
    assert v["fixed_sequence"] == order and res["dev_reference"] == dref
    assert set(v["pairs"]) == {f"{c}-{b}" for c in order for b in bases}
    assert recs[2]["verdict"] == {k: p["passes"] for k, p in v["pairs"].items()}
    p = v["pairs"]["gls_signed-pearson_signed"]
    assert p["rel_gain"] > 0 and "p_holm" not in p and "p_two_sided" in p["info"]
    m = res["metrics"]
    for k, p in v["pairs"].items():                         # gate recomputed from the reported metrics
        c, b = p["candidate"], p["baseline"]
        assert p["rel_gain"] == pytest.approx(m[c]["ap_ht"] / m[b]["ap_ht"] - 1)
        ni = all(m[c][x] >= 0.95 * m[b][x] for x in fe.NI_METRICS)
        assert p["passes"] == (p["rel_gain"] >= 0.10 and ni)
    prim = v["candidates"]["gls_signed"]
    sec = v["candidates"]["gls_abs"]
    assert prim["role"] == "primary" and prim["status"] == "confirmatory"
    assert prim["clearly_beats_ift"] == all(v["pairs"][f"gls_signed-{b}"]["passes"] for b in bases)
    assert sec["status"] == ("confirmatory" if prim["clearly_beats_ift"] else "exploratory")
    info = v["candidate_vs_primary_info"]["gls_abs-gls_signed"]
    assert info["diff"] == pytest.approx(m["gls_abs"]["ap_ht"] - m["gls_signed"]["ap_ht"])
    mm = m["gls_signed"]
    assert mm["n_unmapped_pos"] == 1 and mm["ap_ht_with_unmapped_min_score"] < mm["ap_ht"]
    assert res["ht_f"] == pytest.approx(u.meta["f_ift"])


def test_final_eval_limits(toy, tmp_path):
    u, X = toy
    mat = scorers._Matrix(X)
    mk = lambda n: scorers.make_baseline(n, mat)  # noqa: E731
    with pytest.raises(ValueError):
        fe.run_final_eval(u, {n: mk(n) for n in ("gls_signed", "gls_abs", "pearson_abs")}, {"spearman_abs": mk("spearman_abs")},
                          tmp_path, tmp_path / "l.jsonl", "x")
    with pytest.raises(ValueError):
        fe.run_final_eval(u, {"gls_signed": mk("gls_signed")}, {"gls_signed": mk("gls_signed")}, tmp_path,
                          tmp_path / "l.jsonl", "x")
    with pytest.raises(ValueError):
        fe.run_final_eval(u, {"gls_signed": mk("gls_signed")}, {}, tmp_path, tmp_path / "l.jsonl", "x")
    assert not (tmp_path / "l.jsonl").exists()


# ------------------------------------------------------------------------------------------ verdict logic (synthetic)
BASES = ["pearson_signed", "spearman_signed", "gls_signed"]


def _decide(rel, macro_c, nolarge_c, order):
    """Candidate metric values are per candidate (the same against every baseline)."""
    per = {b: dict(ap_ht=0.30, macro_cluster_ap=0.40, ap_ht_no_largest_component=0.30) for b in BASES}
    diff = {}
    for c in order:
        per[c] = dict(ap_ht=0.30 * (1 + rel[c]["gls_signed"]))
        if macro_c[c] is not None:
            per[c]["macro_cluster_ap"] = macro_c[c]
        if nolarge_c[c] is not None:
            per[c]["ap_ht_no_largest_component"] = nolarge_c[c]
        for b in BASES:
            r = rel[c][b]
            diff[f"{c}-{b}"] = dict(diff=0.30 * r, rel_gain=r, ci_low=-0.05, ci_high=0.2, rel_ci_low=-0.1,
                                    rel_ci_high=0.5, p_two_sided=0.4)   # CI includes 0, p large: not in the gate
    return fe.decide(per, dict(diff=diff), order, BASES)


def _all(x):
    return {b: x for b in BASES}


def test_decide_pass_and_fixed_sequence_confirmatory():
    v = _decide({"P": _all(0.12), "S": _all(0.30)}, {"P": 0.40, "S": 0.40}, {"P": 0.30, "S": 0.30}, ["P", "S"])
    assert all(p["passes"] for p in v["pairs"].values())        # CI including 0 does not block (v3.1)
    assert v["candidates"]["P"] == dict(order=1, role="primary", status="confirmatory", passes_vs=_all(True),
                                        clearly_beats_ift=True, confirmed=True)
    assert v["candidates"]["S"]["status"] == "confirmatory" and v["candidates"]["S"]["confirmed"]
    assert v["fixed_sequence"] == ["P", "S"]


def test_decide_primary_fails_secondary_exploratory():
    v = _decide({"P": _all(0.09), "S": _all(0.50)}, {"P": 0.5, "S": 0.5}, {"P": 0.5, "S": 0.5}, ["P", "S"])
    assert not v["candidates"]["P"]["clearly_beats_ift"] and v["candidates"]["P"]["status"] == "confirmatory"
    s = v["candidates"]["S"]
    assert s["clearly_beats_ift"] and s["status"] == "exploratory" and not s["confirmed"]
    # order matters: the same numbers with S first make S confirmatory
    v2 = _decide({"P": _all(0.09), "S": _all(0.50)}, {"P": 0.5, "S": 0.5}, {"P": 0.5, "S": 0.5}, ["S", "P"])
    assert v2["candidates"]["S"]["confirmed"] and v2["candidates"]["P"]["status"] == "confirmatory"


def test_decide_margin_boundary_and_intersection_union():
    rel = {"P": dict(pearson_signed=0.50, spearman_signed=0.10, gls_signed=0.0999)}
    v = _decide(rel, {"P": 0.4}, {"P": 0.3}, ["P"])
    assert v["pairs"]["P-spearman_signed"]["gate_ii_ift"]           # exactly 10% passes (>=)
    assert not v["pairs"]["P-gls_signed"]["gate_ii_ift"]
    assert v["candidates"]["P"]["passes_vs"] == dict(pearson_signed=True, spearman_signed=True, gls_signed=False)
    assert not v["candidates"]["P"]["clearly_beats_ift"]            # must pass vs all three
    v = _decide({"P": dict(pearson_signed=0.5, spearman_signed=np.nan, gls_signed=0.5)}, {"P": 0.4}, {"P": 0.3}, ["P"])
    assert not v["pairs"]["P-spearman_signed"]["gate_ii_ift"]       # undefined gain fails


@pytest.mark.parametrize("macro,nolarge,ok", [
    (0.95 * 0.40, 0.95 * 0.30, True),          # exactly at the -5% margin passes
    (0.95 * 0.40 - 1e-9, 0.30, False),         # just below on macro
    (0.40, 0.95 * 0.30 - 1e-9, False),         # just below on no-largest-component
    (0.20, 0.60, False),                       # better on one does not compensate the other
    (np.nan, 0.30, False),                     # undefined metric fails
    (None, 0.30, False),                       # missing metric fails
    (0.40, None, False),
])
def test_decide_non_inferiority_edges(macro, nolarge, ok):
    v = _decide({"P": _all(0.25)}, {"P": macro}, {"P": nolarge}, ["P"])
    for b in BASES:
        p = v["pairs"][f"P-{b}"]
        assert p["gate_ii_ift"] and p["passes"] == ok
    assert v["candidates"]["P"]["clearly_beats_ift"] == ok


def test_non_inferior_values():
    r = fe.non_inferior(0.38, 0.40)
    assert r["passed"] and r["threshold"] == pytest.approx(0.38) and r["rel"] == pytest.approx(-0.05)
    assert not fe.non_inferior(0.1, np.nan)["passed"] and fe.non_inferior(0.0, 0.0)["passed"]


def test_check_sequence_and_dev_reference(tmp_path):
    dec = tmp_path / "decision.json"
    dec.write_text(json.dumps(dict(primary="zavg_BEF", secondary="ref_B", fixed_sequence=["zavg_BEF", "ref_B"])))
    assert fe.check_sequence(["zavg_BEF", "B"], dec)["consistent"]
    assert fe.check_sequence(["zavg_BEF"], dec)["consistent"]
    assert not fe.check_sequence(["B", "zavg_BEF"], dec)["consistent"]
    assert not fe.check_sequence(["B"], dec)["consistent"]
    cmp = tmp_path / "compare.json"
    cmp.write_text(json.dumps(dict(seeds={"0": dict(all=dict(diff={"zavg_BEF-ref_G": dict(
        diff=0.2, ci_low=0.1, ci_high=0.3, rel_gain=0.7)}))})))
    r = fe.dev_reference(dict(decision=dec, compare=cmp), ["zavg_BEF", "B"], ["gls_signed"])
    assert r["files"]["compare"]["sha256"] == fe.sha256_file(cmp)
    assert r["d3"]["fixed_sequence"] == ["zavg_BEF", "ref_B"]
    assert r["pooled_oof_delta_vs_baselines"]["zavg_BEF-gls_signed"]["per_seed"] == {
        "0": dict(diff=0.2, ci_low=0.1, ci_high=0.3)}
    assert r["pooled_oof_delta_vs_baselines"]["B-gls_signed"]["per_seed"] is None


def test_real_sequence_is_d3():
    """The committed D3 decision file names the sequence the PI will run (zavg_BEF, then B)."""
    if not fe.DEV_FILES["decision"].exists():
        pytest.skip("no Phase B v2 decision file")
    assert fe.check_sequence(["zavg_BEF", "B"])["consistent"]


# ------------------------------------------------------------------------------------------ frozen candidates
OOF = CACHE / "phaseB_v2_oof"


def test_candidate_scorers_reproduce_phaseB_v2_and_predict():
    """zavg_BEF / B on a sample of dev pairs == the values evaluated in Phase B v2 (pooled OOF arrays, exact) and
    == src/predict.py (predict_pairs) z-average with its recorded null moments and live recompute of B, F."""
    need = [OOF / "zavg_BEF_cluster_seed0.npy", OOF / "ref_B_cluster_seed0.npy",
            RESULTS / "predict" / "union_zavg_B_E_F.run.json"] + [
        CACHE / "layers" / f"{t}.scores.f32.npy" for t in ("screen_v1", "screen_v2", "graph_v1")]
    if not all(p.exists() for p in need):
        pytest.skip("Phase B v2 OOF / layer caches not present")
    from src import predict
    from src.universe import load_dev_universe
    u = load_dev_universe()
    zs, bs = scorers.make_scorer("zavg_BEF"), scorers.make_scorer("B")
    assert zs.label_free and bs.label_free
    zs.bind(u)
    bs.bind(u)
    oz, ob = np.load(need[0]), np.load(need[1])
    rng = np.random.default_rng(SEEDS[0])
    idx = np.sort(rng.choice(np.flatnonzero(np.isfinite(oz)), 5000, replace=False))
    pid = u.col("pair_id")[idx]
    np.testing.assert_array_equal(zs.score(idx), oz[idx])              # Phase B v2 zavg_BEF (identical code path)
    np.testing.assert_array_equal(scorers.null_z("B")[pid], ob[idx])   # Phase B v2 ref_B (= null z of B)
    rawB = bs.score(idx)
    np.testing.assert_array_equal(rawB, scorers.cached_layer("B")[0][pid])
    assert np.all(np.diff(rawB[np.argsort(ob[idx])]) >= 0)             # B is monotone in its z: same ranking
    # predict_pairs: the null moments it used for zavg:B,E,F equal the candidate's, and ZAvg gives the same score
    mom = json.loads(need[2].read_text())["null"]["moments"]
    raw = {k: scorers.cached_layer(k)[0][pid] for k in "BEF"}
    for k in "BEF":
        mem = zs.params["members"][k]
        assert (mem["null_mean"], mem["null_sd"]) == (mom[k]["null_mean"], mom[k]["null_sd"])
        assert mem["layer"] == predict.COMPONENT_NAMES[k]
    Z = {k: (raw[k] - mom[k]["null_mean"]) / mom[k]["null_sd"] for k in "BEF"}
    np.testing.assert_allclose(predict.ZAvg(["B", "E", "F"]).predict(Z), oz[idx], rtol=0, atol=1e-12)
    # live label-free recompute (src/predict.py PairScorer) of B and F on 300 pairs; E: tests/test_predict.py
    X, genes, _ = predict.load_matrix(cache=True)
    ps = predict.PairScorer(X, genes)
    sub = idx[:300]
    ia, ib = u.col("ia")[sub], u.col("ib")[sub]
    live = {k: ps.component(k, ia, ib) for k in "BF"}
    np.testing.assert_allclose(live["B"], bs.score(sub), rtol=0, atol=1e-4)
    np.testing.assert_allclose(live["F"], scorers.cached_layer("F")[0][u.col("pair_id")[sub]], rtol=0, atol=1e-6)
    Zl = {k: ps.z(k, live[k]) for k in "BF"}
    Zl["E"] = Z["E"][:300]
    np.testing.assert_allclose(predict.ZAvg(["B", "E", "F"]).predict(Zl), zs.score(sub), rtol=0, atol=1e-4)


def test_export_curves_toy(toy, tmp_path):
    """Re-plot of an existing (toy) look: logged as '<N>-figure' before scoring, curve AP == the look's HT-AP,
    scorer set must equal the look's."""
    u, X = toy
    mat = scorers._Matrix(X)
    looks = tmp_path / "looks.jsonl"
    looks.write_text(json.dumps(dict(look=0, date="x", what="seed")) + "\n")
    mk = lambda: ({"gls_signed": scorers.make_baseline("gls_signed", mat)},
                  {"pearson_signed": scorers.make_baseline("pearson_signed", mat)})
    c, b = mk()
    fe.run_final_eval(u, c, b, tmp_path / "ift", looks, "toy look", n_boot=20, dev_ref={})
    look_file = next((tmp_path / "ift").glob("look01_*.json"))
    c, b = mk()
    out = fe.export_curves(u, {**c, **b}, look_file, looks, tmp_path / "curves.json", "toy figure")
    rec = json.loads(looks.read_text().splitlines()[-1])
    assert rec["look"] == "1-figure" and rec["what"] == "toy figure"
    for v in out["curves"].values():
        assert v["ap_from_curve"] == pytest.approx(v["ap_look"], abs=1e-12)
        assert np.all(np.diff(v["recall"]) > 0) and v["recall"][-1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fe.export_curves(u, mk()[0], look_file, looks, tmp_path / "c2.json", "x")


def test_posthoc_reference_toy(toy, tmp_path):
    """Post-hoc reference mode (review B1): logged as '2-posthoc' BEFORE scoring, metric_suite for reference and
    primary, bootstrap Δ(reference − primary) only, no verdict; refuses a second evaluation; names whitelisted."""
    u, X = toy
    mat = scorers._Matrix(X)
    looks = tmp_path / "looks.jsonl"
    looks.write_text(json.dumps(dict(look=1, date="x", what="seed")) + "\n")
    ref = {"pearson_signed": scorers.make_baseline("pearson_signed", mat)}
    prim = {"gls_signed": scorers.make_baseline("gls_signed", mat)}
    out = tmp_path / "rev" / "posthoc.json"
    res = fe.posthoc_reference(u, ref, prim, looks, out, "toy posthoc", n_boot=50)
    recs = [json.loads(l) for l in looks.read_text().splitlines()]
    assert recs[-1] == dict(look="2-posthoc", date=recs[-1]["date"], what="toy posthoc", reference="pearson_signed")
    assert fe.next_look(looks) == 2                                    # post-hoc record is not a numbered look
    assert out.exists() and "verdict" not in res and res["posthoc"] is True
    assert set(res["bootstrap"]["diff"]) == {"pearson_signed-gls_signed"}
    d = res["bootstrap"]["diff"]["pearson_signed-gls_signed"]["diff"]
    assert d == pytest.approx(res["metrics"]["pearson_signed"]["ap_ht"] - res["metrics"]["gls_signed"]["ap_ht"])
    with pytest.raises(RuntimeError):
        fe.posthoc_reference(u, ref, prim, looks, out, "again", n_boot=10)
    assert len(looks.read_text().splitlines()) == 2
    with pytest.raises(SystemExit):
        fe.posthoc_scorer("spec_k20_t0.1_std")                         # the method's own layer is not a reference
