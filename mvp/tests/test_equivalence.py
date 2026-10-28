"""The ported implementation reproduces the research code's cached scores (optional; needs the research cache).

Set COESS_RESEARCH_CACHE to the research project's ``data_cache`` directory. It must contain
``CRISPRGeneEffect.f32.npy`` / ``CRISPRGeneEffect.genes.txt`` (the DepMap 26Q3 matrix as float32), the layer caches
``layers/{screen_v1,screen_v2,graph_v1}.{pairs.npz,scores.f32.npy,manifest.json}`` and, for the baselines,
``scores/CRISPRGeneEffect_all.npz``. Only gene-index pairs and score columns are read (no labels).
Builds the full genome-wide state (~2-3 min, ~4 GB RAM).
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from coessential.baselines import Baselines
from coessential.method import Config, CoessentialityModel

CACHE = os.environ.get("COESS_RESEARCH_CACHE")
pytestmark = pytest.mark.skipif(not CACHE or not (Path(CACHE) / "CRISPRGeneEffect.f32.npy").exists(),
                                reason="COESS_RESEARCH_CACHE not set (research cache not available)")

LAYERS = {  # component: (cache tag, layer name)
    "F": ("screen_v1", "spec_k20_t0.1_std"),
    "B": ("screen_v2", "rowz[socos_q0.25[spec_k20_t0.1_std]]"),
    "E": ("graph_v1", "emb_lap512[knn_union_lin_q0.25[spec_k20_t0.1_std]]"),
}
TOL = {"F": 1e-6, "B": 1e-4, "E": 1e-5}     # absolute, raw scale (cache stores float32)
N_SAMPLE = 5000


@pytest.fixture(scope="module")
def env():
    c = Path(CACHE)
    X = np.load(c / "CRISPRGeneEffect.f32.npy")
    G = X.shape[1]
    assert len((c / "CRISPRGeneEffect.genes.txt").read_text().split("\n")) >= G
    m = CoessentialityModel(X, Config())
    return c, X, G, m


def _rows(c, tag, G, ia, ib):
    with np.load(c / "layers" / f"{tag}.pairs.npz") as P:
        pa, pb = P["ia"], P["ib"]
    key = np.minimum(pa, pb).astype(np.int64) * G + np.maximum(pa, pb)
    order = np.argsort(key)
    q = np.minimum(ia, ib) * G + np.maximum(ia, ib)
    pos = np.searchsorted(key[order], q)
    assert (key[order][pos] == q).all()
    return order[pos]


def test_components_match_research_cache(env):
    c, X, G, m = env
    with np.load(c / "layers" / "screen_v1.pairs.npz") as P:
        ia, ib = P["ia"].astype(np.int64), P["ib"].astype(np.int64)
    sel = np.random.default_rng(0).choice(len(ia), N_SAMPLE, replace=False)
    ia, ib = ia[sel], ib[sel]
    for comp, (tag, name) in LAYERS.items():
        man = json.loads((c / "layers" / f"{tag}.manifest.json").read_text())
        col = next(x for x in man["columns"] if x["name"] == name)
        ref = np.load(c / "layers" / f"{tag}.scores.f32.npy", mmap_mode="r")[:, col["col"]]
        ref = ref[_rows(c, tag, G, ia, ib)].astype(np.float64)
        got = m.component(comp, ia, ib)
        assert np.abs(got - ref).max() < TOL[comp], (comp, np.abs(got - ref).max())
        mu, sd = m.null_moments(comp)
        assert abs(mu - col["null_mean"]) / col["null_sd"] < 1e-4 and abs(sd - col["null_sd"]) / col["null_sd"] < 1e-4


def test_baselines_match_research_cache(env):
    c, X, G, _ = env
    with np.load(c / "scores" / "CRISPRGeneEffect_all.npz") as z:
        ia, ib = z["ia"], z["ib"]
        sel = np.random.default_rng(0).choice(len(ia), 3000, replace=False)
        ref = {k: z[f"{k}_signed"][sel] for k in ("pearson", "spearman", "gls")}
    bl = Baselines(X)
    for k, r in ref.items():
        assert np.allclose(bl.score(k, ib[sel], ia[sel]), r, rtol=1e-8, atol=1e-8), k
