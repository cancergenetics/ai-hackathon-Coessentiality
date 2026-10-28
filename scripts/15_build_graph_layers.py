"""Build the graph_v1 label-free layer cache (Phase A A7: kNN graph / diffusion / spectral embedding; src/layers_graph.py).

Same contract as scripts/10 and 12: every layer is label-free (CRISPRGeneEffect + gene indices only), scored for all
596,597 scorable pairs aligned to the canonical pair_id, with null z-moments on the fixed 200k genome-wide BG pairs
(seed SEEDS[0]). Output: data_cache/layers/graph_v1.{scores.f32.npy,pairs.npz,manifest.json} and
results/phaseA_graph/graph_v1.manifest.json (per-column config, rationale, null moments, runtimes).

The grid was fixed before any graph layer was evaluated on dev; the choice of the base (spec_k20_t0.1_std, plus k=0/5)
and of q ~ 0.25 was informed by the earlier Phase-A screens (dev-selected, disclosed). Columns 0-1 reproduce the
current best (socos_q0.25 and its row-z) inside this cache as an exactness check (max |diff| vs screen_v2 recorded).

Round 2 (``--round2``; cache graph_v2, 7 columns): chosen AFTER the graph_v1 SEEDS[0] screen (dev-selected, disclosed):
the Laplacian-eigenmap embedding was best and still improving at the largest dimension (512).

Usage: ./.env/bin/python scripts/15_build_graph_layers.py [--round2]
"""
import hashlib
import json
import logging
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from src import data, layers  # noqa: E402
from src.config import CACHE, RESULTS, ROOT  # noqa: E402
from src.layers import Spectral  # noqa: E402
from src.layers_graph import GraphEngine, GraphLayer, KNNGraph, SoftGraph  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("15b")
OUT = CACHE / "layers"
RES = RESULTS / "phaseA_graph"
TAG = "graph_v1"

S20, S0, S5 = (Spectral(k, 0.1, standardize=True) for k in (20, 0, 5))


def kg(sym, w="lin", q=0.25, base=S20):
    return KNNGraph(base, q, sym, w)


UL = kg("union")
R_REPRO = "reproduction of the current best (socos_q0.25 / rowz) inside this cache"
R_SYM = "kNN symmetrisation: does reciprocity (union / mutual / avg) beat directed top-K neighbour sets?"
R_W = "edge weights: binary vs linear vs soft-threshold power of the balanced z (WGCNA-like within the kNN set)"
R_Q = "neighbourhood size for symmetric graphs (mutual pruning shrinks the effective K)"
R_TOM = "topological overlap (WGCNA) on kNN / dense soft-threshold adjacency"
R_DIF = "diffusion on the symmetric-normalised union kNN graph: multi-step / restart / heat kernels"
R_EMB = "spectral embedding of the union kNN graph (Laplacian eigenmaps / diffusion map), cosine; dims sweep"
R_BASE = "first-order base sensitivity (k = 0 / 5 PCs removed instead of 20)"

# (core layer, rationale, rowz too?)
CORES = [
    (GraphLayer(kg("dir")), R_REPRO, True),
    (GraphLayer(kg("union")), R_SYM, True),
    (GraphLayer(kg("mutual")), R_SYM, True),
    (GraphLayer(kg("avg")), R_SYM, True),
    (GraphLayer(kg("dir", "bin")), R_W, True),
    (GraphLayer(kg("dir", "p4")), R_W, True),
    (GraphLayer(kg("dir", "p8")), R_W, True),
    (GraphLayer(kg("union", "p4")), R_W, True),
    (GraphLayer(kg("union", q=0.5)), R_Q, True),
    (GraphLayer(kg("mutual", q=0.5)), R_Q, True),
    (GraphLayer(kg("mutual", q=1.0)), R_Q, True),
    (GraphLayer(kg("union", "bin"), "adj", "tom"), R_TOM, True),
    (GraphLayer(kg("mutual", "bin", 0.5), "adj", "tom"), R_TOM, True),
    (GraphLayer(UL, "adj", "tom"), R_TOM, True),
    (GraphLayer(SoftGraph(S20, 6.0), "adj", "tom"), R_TOM, True),
    (GraphLayer(SoftGraph(S20, 6.0), "adj", "cos"), R_TOM + " (cosine of dense soft-threshold rows)", True),
    (GraphLayer(UL, "pow1", "cos"), R_DIF + " (1 step = degree-normalised second order)", True),
    (GraphLayer(UL, "pow2", "kern"), R_DIF, True),
    (GraphLayer(UL, "pow2", "cos"), R_DIF, True),
    (GraphLayer(UL, "pow3", "cos"), R_DIF, True),
    (GraphLayer(UL, "ppr0.5", "kern"), R_DIF, True),
    (GraphLayer(UL, "ppr0.5", "cos"), R_DIF, True),
    (GraphLayer(UL, "heat1", "cos"), R_DIF, True),
    (GraphLayer(UL, "heat3", "cos"), R_DIF, True),
    (GraphLayer(kg("mutual", q=0.5), "pow2", "cos"), R_DIF + " (mutual graph)", True),
    (GraphLayer(UL, "lap32", "emb"), R_EMB, False),
    (GraphLayer(UL, "lap128", "emb"), R_EMB, False),
    (GraphLayer(UL, "lap512", "emb"), R_EMB, False),
    (GraphLayer(UL, "dmap512t4", "emb"), R_EMB, False),
    (GraphLayer(kg("union", base=S0)), R_BASE, True),
    (GraphLayer(kg("union", base=S5)), R_BASE, True),
    (GraphLayer(kg("union", base=S0), "ppr0.5", "cos"), R_BASE, True),
    (GraphLayer(kg("union", base=S5), "ppr0.5", "cos"), R_BASE, True),
]
R_R2 = ("round 2 (after the graph_v1 SEEDS[0] screen: emb_lap512 best, AP still rising with d 32 -> 128 -> 512): "
         "d = 1024, row-z of embeddings, denser graph, k = 0 base")
UL5 = kg("union", q=0.5)
CORES_V2 = [
    (GraphLayer(UL, "lap512", "emb", True), R_R2, False),
    (GraphLayer(UL, "lap1024", "emb"), R_R2, True),
    (GraphLayer(UL5, "lap512", "emb"), R_R2, True),
    (GraphLayer(kg("union", base=S0), "lap512", "emb"), R_R2, True),
]
if "--round2" in sys.argv:
    TAG, CORES = "graph_v2", CORES_V2
GRID = []
for core, why, rz in CORES:
    GRID.append((core, why))
    if rz:
        GRID.append((GraphLayer(core.graph, core.op, core.score, True), why + " + genome-wide row-z"))

REPRO = {"score_so": "socos_q0.25[spec_k20_t0.1_std]", "rowz_so": "rowz[socos_q0.25[spec_k20_t0.1_std]]"}


def sha256(path=None, arrays=()):
    h = hashlib.sha256()
    if path is not None:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main():
    t_all = time.perf_counter()
    names = [c.name for c, _ in GRID]
    assert len(set(names)) == len(names)
    pairs, _ = data.load_pairs()
    sc = pairs.loc[pairs.scorable, ["a", "b", "ia", "ib"]].reset_index(drop=True)   # labels dropped here
    ia, ib = sc.ia.values.astype(np.int64), sc.ib.values.astype(np.int64)
    P1 = np.load(OUT / "screen_v1.pairs.npz")
    assert np.array_equal(P1["ia"], ia) and np.array_equal(P1["ib"], ib)
    eng = layers.get_engine("CRISPRGeneEffect")
    ge = GraphEngine(eng)
    na, nb = layers.null_pairs(eng)
    assert np.array_equal(na, P1["null_a"]) and np.array_equal(nb, P1["null_b"])
    M = np.full((len(sc), len(GRID)), np.nan, dtype=np.float32)
    cols = []
    cur_base, cur_graph = None, None
    for j, (cfg, why) in enumerate(GRID):
        b = cfg.graph.base
        if b != cur_base:
            eng.release()
            ge.release()
            cur_base = b
        if cfg.graph != cur_graph:
            ge._emb = {}
            cur_graph = cfg.graph
        t0 = time.perf_counter()
        s = ge.compute(cfg, ia, ib)
        t_pairs = time.perf_counter() - t0
        t0 = time.perf_counter()
        mo = layers.moments(ge.compute(cfg, na, nb))
        t_null = time.perf_counter() - t0
        M[:, j] = s
        cols.append(dict(col=j, name=cfg.name, config=cfg.to_dict(), rationale=why, **mo,
                         n_nonfinite=int((~np.isfinite(s)).sum()), runtime_pairs_incl_prep_s=round(t_pairs, 3),
                         runtime_null_s=round(t_null, 3)))
        log.info("%-60s %.1fs (null %.1fs) null mean %.4g sd %.4g", cfg.name, t_pairs, t_null, mo["null_mean"],
                 mo["null_sd"])
    # exactness check vs screen_v2 (graph_v1 columns 0, 1) / vs graph_v1 (round 2: lap512 recomputed)
    if TAG == "graph_v1":
        m2 = json.loads((RESULTS / "layers" / "screen_v2.manifest.json").read_text())
        c2 = {c["name"]: c["col"] for c in m2["columns"]}
        S2 = np.load(OUT / "screen_v2.scores.f32.npy", mmap_mode="r")
        repro = {names[0]: float(np.max(np.abs(M[:, 0] - S2[:, c2[REPRO["score_so"]]]))),
                 names[1]: float(np.max(np.abs(M[:, 1] - S2[:, c2[REPRO["rowz_so"]]])))}
    else:
        repro = {}
    log.info("reproduction max |diff| vs screen_v2: %s", repro)
    f_scores, f_pairs = OUT / f"{TAG}.scores.f32.npy", OUT / f"{TAG}.pairs.npz"
    np.save(f_scores, M)
    np.savez(f_pairs, ia=ia, ib=ib, null_a=na, null_b=nb)
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout  # noqa
    v1 = json.loads((RESULTS / "layers" / "manifest.json").read_text())
    manifest = dict(
        tag=TAG, created=time.strftime("%Y-%m-%dT%H:%M:%S"), script="scripts/15_build_graph_layers.py",
        git=dict(commit=run("rev-parse", "HEAD").strip(),
                 dirty_files=[l[3:] for l in run("status", "--porcelain", "--", "src", "scripts").splitlines() if l]),
        label_free="layer scores use only CRISPRGeneEffect and gene indices; no label is read. Grid fixed before any "
                   "graph layer was evaluated; base / q informed by earlier Phase-A dev screens (disclosed).",
        reproduction_max_abs_diff_vs_screen_v2=repro,
        input=v1["input"], preprocessing=dict(v1["preprocessing"], eig=eng.info, graph_timing=ge.timing),
        pairs=dict(n=len(sc), sha256_ia_ib=sha256(arrays=(ia, ib)), file=str(f_pairs.relative_to(ROOT)),
                   same_as="screen_v1"),
        null=v1["null"],
        output=dict(file=str(f_scores.relative_to(ROOT)), shape=list(M.shape), dtype="float32", sha256=sha256(f_scores)),
        runtime=dict(total_s=round(time.perf_counter() - t_all, 1),
                     peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)),
        columns=cols)
    txt = json.dumps(manifest, indent=1)
    RES.mkdir(parents=True, exist_ok=True)
    (OUT / f"{TAG}.manifest.json").write_text(txt)
    (RES / f"{TAG}.manifest.json").write_text(txt)
    log.info("wrote %s %s in %.0fs", f_scores, M.shape, manifest["runtime"]["total_s"])


if __name__ == "__main__":
    main()
