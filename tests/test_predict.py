"""src/predict.py + scripts/predict_pairs.py: agreement with the cached layer / baseline scores on training pair IDs,
null moments vs manifests, raw-CSV provenance, batch / order invariance, unmapped handling, model plug-in.

Only the two ID columns of the HuMAP train pair files are read (both files concatenated and shuffled: which file a
pair came from is never used). Cached baseline scores are read by key (ia, ib, *_signed) only. No label is loaded and
no metric is computed. The module fixture builds the full genome-wide state (~3-4 min: E and H need G x G matrices).
"""
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from src import predict
from src.config import CACHE, NEG_FILE, POS_FILE, SEEDS

N_SUB = 20_000
LAYER_CACHE = {"B": ("screen_v2", 17), "E": ("graph_v1", 52), "H": ("graph_v1", 45), "F": ("screen_v1", 62)}

# needs the raw datasets and the research data_cache (not distributed); skip cleanly when absent
from src.config import DEPMAP  # noqa: E402
_NEEDED = [POS_FILE, NEG_FILE, DEPMAP / "CRISPRGeneEffect.csv", CACHE / "layers"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in _NEEDED),
                                reason="raw datasets / data_cache not present (see datasets/README.md)")
TOL = {"F": 1e-6, "B": 1e-4, "E": 1e-5, "H": 1e-4}       # float32 cache vs float64 recompute (abs)


def train_ids() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f, sep="\t", header=None, usecols=[0, 1], names=["uniprot_a", "uniprot_b"], dtype=str)
                   for f in (POS_FILE, NEG_FILE)], ignore_index=True)
    return d.sample(frac=1.0, random_state=SEEDS[0]).reset_index(drop=True)


class Lin:                                        # picklable model for the plug-in test
    components = ["B", "F"]

    def predict(self, Z):
        return 2.0 * Z["B"] - Z["F"]


@pytest.fixture(scope="module")
def env():
    X, genes, src = predict.load_matrix(cache=True)
    sc = predict.PairScorer(X, genes, keep_graph=True)
    ids = train_ids()
    sub = ids.iloc[:N_SUB].reset_index(drop=True)
    meth = predict.parse_method("zavg:F,B,E,H")
    out, rep = predict.score_pairs(sub, meth, sc, components=True)
    return dict(sc=sc, X=X, genes=genes, ids=ids, sub=sub, out=out, rep=rep, src=src)


def _cache_rows(G, ia, ib):
    P = np.load(CACHE / "layers" / "screen_v1.pairs.npz")
    key = np.minimum(P["ia"], P["ib"]) * G + np.maximum(P["ia"], P["ib"])
    order = np.argsort(key)
    q = np.minimum(ia, ib) * G + np.maximum(ia, ib)
    pos = np.searchsorted(key[order], q)
    assert (key[order][pos] == q).all()
    return order[pos]


# ------------------------------------------------------------------------------------------ provenance
def test_raw_csv_equals_cache_and_manifest(env):
    X, genes, src = predict.load_matrix(cache=False)          # raises unless np.save hash == manifest input sha256
    assert src["npy_sha256_matches_manifest"] and genes.equals(env["genes"])
    assert np.array_equal(X, env["X"], equal_nan=True)


def test_null_moments_match_manifests(env):
    sc = env["sc"]
    for c in "FBEH":
        m = sc.moments[c]
        assert m["rel_diff_mean"] <= predict.NULL_TOL and m["rel_diff_sd"] <= predict.NULL_TOL, m
        assert m["null_n"] == 200_000


def test_frozen_configs_are_manifest_configs():
    for c in "FBEH":
        cols = {x["name"]: x for x in json.loads(predict.MANIFESTS[c].read_text())["columns"]}
        assert predict.COMPONENT_NAMES[c] in cols


def test_parse_method():
    assert predict.parse_method("single").components == ["B"]
    assert predict.parse_method("gls").components == ["G"]
    assert predict.parse_method("comb_zavg_BEF").components == ["B", "E", "F"]
    m = predict.parse_method("zavg:B,E,gls")
    assert m.kind == "zavg" and m.components == ["B", "E", "G"]
    for bad in ("zavg:B,B", "zavg:Q", "comb_zavg_BX", "nope"):
        with pytest.raises(ValueError):
            predict.parse_method(bad)


# ------------------------------------------------------------------------------------------ agreement with cache
def test_mapping_reproduces_scorable_universe(env):
    """All train IDs: mapped canonical pairs == the cached scorable pair set (596,597), same rule as src/data."""
    ia, ib, _ = predict.map_ids(env["ids"], env["genes"])
    ok = (ia >= 0) & (ib >= 0) & (ia != ib)
    G = len(env["genes"])
    P = np.load(CACHE / "layers" / "screen_v1.pairs.npz")
    ref = np.sort(np.minimum(P["ia"], P["ib"]) * G + np.maximum(P["ia"], P["ib"]))
    got = np.sort(np.minimum(ia[ok], ib[ok]) * G + np.maximum(ia[ok], ib[ok]))
    assert len(got) == 596_597 and np.array_equal(got, ref)


def test_layers_match_cache(env):
    out, sub, G = env["out"], env["sub"], len(env["genes"])
    ia, ib, _ = predict.map_ids(sub, env["genes"])
    m = out.mapped.values == 1
    assert m.sum() > 0.9 * N_SUB
    rows = _cache_rows(G, ia[m], ib[m])
    sc = env["sc"]
    for c, (tag, col) in LAYER_CACHE.items():
        ref = np.load(CACHE / "layers" / f"{tag}.scores.f32.npy", mmap_mode="r")[:, col][rows].astype(np.float64)
        raw = out[f"z_{c}"].values[m] * sc.moments[c]["null_sd"] + sc.moments[c]["null_mean"]
        assert np.abs(raw - ref).max() < TOL[c], (c, np.abs(raw - ref).max())
    # method score = mean of the four z-scores
    z = out[[f"z_{c}" for c in "FBEH"]].values[m]
    assert np.allclose(out.score.values[m], z.mean(1), atol=1e-12)


def test_single_is_raw_B(env):
    out, rep = predict.score_pairs(env["sub"].iloc[:3000], predict.parse_method("single"), env["sc"])
    ref = env["out"].iloc[:3000]
    m = ref.mapped.values == 1
    mo = env["sc"].moments["B"]
    assert np.array_equal(out.mapped.values, ref.mapped.values)
    assert np.allclose(out.score.values[m], ref.z_B.values[m] * mo["null_sd"] + mo["null_mean"], atol=1e-12)


def test_baselines_match_cache(env):
    z = np.load(CACHE / "scores" / "CRISPRGeneEffect_all.npz")      # keys read: ia, ib, *_signed (no label)
    ia, ib = z["ia"], z["ib"]
    sel = np.random.default_rng(SEEDS[0]).choice(len(ia), 5000, replace=False)
    for code, key in (("P", "pearson_signed"), ("S", "spearman_signed"), ("G", "gls_signed")):
        s = env["sc"].component(code, ib[sel], ia[sel])            # reversed orientation on purpose
        assert np.allclose(s, z[key][sel], rtol=1e-10, atol=1e-10), code


# ------------------------------------------------------------------------------------------ invariance
def test_order_and_batch_invariance(env):
    sc, sub = env["sc"], env["sub"].iloc[:4000].reset_index(drop=True)
    meth = predict.parse_method("zavg:B,E,H,F,G")
    a, _ = predict.score_pairs(sub, meth, sc, components=True)
    # other batch: swapped orientation, shuffled, mixed with other train pairs + duplicates + unknown IDs
    other = env["ids"].iloc[N_SUB:N_SUB + 6000]
    sw = sub.rename(columns={"uniprot_a": "uniprot_b", "uniprot_b": "uniprot_a"})[["uniprot_a", "uniprot_b"]]
    junk = pd.DataFrame(dict(uniprot_a=["ZZZ999", "P12345"], uniprot_b=["Q00000X", "ZZZ999"]))
    mix = pd.concat([other, sw, sub.iloc[:500], junk], ignore_index=True)
    perm = np.random.default_rng(1).permutation(len(mix))
    b, _ = predict.score_pairs(mix.iloc[perm].reset_index(drop=True), meth, sc, components=True)
    b = b.iloc[np.argsort(perm)].reset_index(drop=True)
    bs = b.iloc[len(other):len(other) + len(sub)].reset_index(drop=True)
    bd = b.iloc[len(other) + len(sub):len(other) + len(sub) + 500].reset_index(drop=True)
    m = a.mapped.values == 1
    assert np.array_equal(a.mapped.values, bs.mapped.values)
    for col in ["score"] + [f"z_{c}" for c in "BEHFG"]:
        assert np.array_equal(a[col].values[m], bs[col].values[m]), col           # bit-identical
        assert np.array_equal(a[col].values[:500][m[:500]], bd[col].values[m[:500]]), col
    assert (bs.uniprot_a.values == sub.uniprot_b.values).all()                       # input orientation kept


def test_cold_engine_invariance_B(env):
    """Fresh engine, different request: genome-wide state does not depend on the request."""
    sc2 = predict.PairScorer(env["X"], env["genes"])
    sub = env["sub"].iloc[::7].iloc[::-1].reset_index(drop=True)
    ia, ib, _ = predict.map_ids(sub, env["genes"])
    ok = (ia >= 0) & (ib >= 0) & (ia != ib)
    s2 = sc2.component("B", ib[ok], ia[ok])
    s1 = env["sc"].component("B", ia[ok], ib[ok])
    assert np.array_equal(s1, s2)


# ------------------------------------------------------------------------------------------ unmapped / toy
def test_unmapped_handling(env):
    sc = env["sc"]
    good = env["out"][env["out"].mapped == 1].iloc[:5]
    toy = pd.concat([good[["uniprot_a", "uniprot_b"]],
                     pd.DataFrame(dict(uniprot_a=["NOTANID", good.uniprot_a.iloc[0], "P00000", good.uniprot_a.iloc[1]],
                                       uniprot_b=[good.uniprot_b.iloc[0], "XXXX_HUMAN", "P99999",
                                                  good.uniprot_a.iloc[1]]))], ignore_index=True)
    out, rep = predict.score_pairs(toy, predict.parse_method("comb_zavg_BE"), sc, components=True)
    assert out.mapped.tolist() == [1] * 5 + [0] * 4
    floor = out.score.values[:5].min() - 1
    assert np.all(out.score.values[5:] == floor) and rep["counts"]["unmapped_floor"] == floor
    assert out[["z_B", "z_E"]].iloc[5:].isna().all().all()
    assert rep["counts"]["unmapped_reasons"] == dict(protein_not_mapped=3, self_or_same_gene=1, undefined_score=0)
    allbad = pd.DataFrame(dict(uniprot_a=["FOO1", "FOO2"], uniprot_b=["BAR1", "BAR2"]))
    o2, _ = predict.score_pairs(allbad, predict.parse_method("single"), sc)
    assert o2.mapped.sum() == 0 and (o2.score == -1.0).all()


def test_model_plugin(env, tmp_path):
    sc, sub = env["sc"], env["sub"].iloc[:2000]
    zb, _ = predict.score_pairs(sub, predict.parse_method("zavg:B,F"), sc)
    spec = tmp_path / "m.json"
    spec.write_text(json.dumps(dict(kind="linear", components=["B", "F"], weights=[0.5, 0.5], intercept=0.0)))
    lj, _ = predict.score_pairs(sub, predict.parse_method(None, spec), sc)
    m = zb.mapped.values == 1
    assert np.allclose(lj.score.values[m], zb.score.values[m], atol=1e-12)
    pk = tmp_path / "m.pkl"
    pk.write_bytes(pickle.dumps(Lin()))
    lp, _ = predict.score_pairs(sub, predict.parse_method(None, pk), sc, components=True)
    assert np.allclose(lp.score.values[m], 2 * lp.z_B.values[m] - lp.z_F.values[m], atol=1e-12)


def _load_cli():
    import importlib.util
    from src.config import ROOT
    spec = importlib.util.spec_from_file_location("predict_pairs", ROOT / "scripts" / "predict_pairs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_toy_file(env, tmp_path):
    good = env["out"][env["out"].mapped == 1].iloc[:20]
    f = tmp_path / "toy.txt"
    lines = [f"{a}\t{b}" for a, b in zip(good.uniprot_a, good.uniprot_b)] + ["UNKNOWN1\tUNKNOWN2", "", "P1\tQ2\textra"]
    f.write_text("\n".join(lines) + "\n")
    out = tmp_path / "s.tsv"
    _load_cli().main(["--pairs", str(f), "--method", "pearson", "--out", str(out), "--cache"])
    d = pd.read_csv(out, sep="\t")
    assert list(d.columns) == ["uniprot_a", "uniprot_b", "score", "mapped"] and len(d) == 22
    assert d.mapped.tolist() == [1] * 20 + [0, 0]
    rec = json.loads((tmp_path / "s.tsv.run.json").read_text())
    assert rec["inputs"]["pairs"]["sha256"] == predict.sha256_file(f)
    assert rec["inputs"]["matrix"]["npy_sha256_matches_manifest"] and rec["git"]["commit"]
    assert rec["counts"]["n_mapped"] == 20 and rec["runtime"]["total_s"] > 0
