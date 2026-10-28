"""Build the screen_v2 label-free layer cache: a small set of extra layers chosen from the screen_v1 dev trends.

Same contract as scripts/10_build_layers.py (label-free, all 596,597 scorable pairs, aligned to the canonical
pair_id; null z-moments on the fixed 200k genome-wide BG pairs, seed SEEDS[0]). Output:
data_cache/layers/screen_v2.{scores.f32.npy,pairs.npz,manifest.json} and results/layers/screen_v2.manifest.json.
results/layers/manifest.json (screen_v1) is NOT touched.

The layers were chosen AFTER looking at the screen_v1 dev results (results/phaseA/README.md), so they are dev-selected
(disclosed); every one has a rationale below.

Usage: ./.env/bin/python scripts/12_build_layers_v2.py
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

from src import data, layers, layers_ext  # noqa: E402
from src.config import CACHE, RESULTS, ROOT, SEEDS  # noqa: E402
from src.layers import MutualRank, RowZ, SecondOrder, Spectral  # noqa: E402
from src.layers_ext import RowZSO  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("12")
OUT = CACHE / "layers"
TAG = "screen_v2"

S20 = Spectral(20, 0.1, standardize=True)       # best screen_v1 layer
S0 = Spectral(0, 0.1, standardize=True)         # best k=0 standardised layer (k-insensitive at tau=0.1)
U20 = Spectral(20, 0.1)                          # best unstandardised base for second-order (socos_q0.5)
U5 = Spectral(5, 0.1)

R_TAU = ("standardised Sigma beat unstandardised at tau=0.1 and 1 (0.36 vs 0.30); grid had only 0.1/1/inf -> "
         "locate the tau optimum below/around 0.1")
R_A3 = "rowz / mutual rank gave +0.01-0.03 on every unstandardised base; not yet tried on the (better) standardised base"
R_SO = ("socos_q0.5 was the best derived layer on every whitened base (+0.05-0.07 over base) and AP fell with q "
        "(0.5 > 1 > 2) -> q0.5 on the standardised bases, and q0.25 to test the q trend")
R_RZSO = "rowz helped spectral layers; second-order rows are not balanced genome-wide -> rowz of the best socos layer"

GRID = [
    (Spectral(0, 0.01, standardize=True), R_TAU),
    (Spectral(0, 0.03, standardize=True), R_TAU),
    (Spectral(0, 0.3, standardize=True), R_TAU),
    (Spectral(0, None, shrink="lw", standardize=True), R_TAU + " (Ledoit-Wolf on the standardised matrix)"),
    (Spectral(20, 0.03, standardize=True), R_TAU + " (k=20)"),
    (Spectral(20, 0.3, standardize=True), R_TAU + " (k=20)"),
    (RowZ(S20), R_A3),
    (MutualRank(S20, True), R_A3),
    (RowZ(S0), R_A3),
    (SecondOrder(S20, 0.5, "cos"), R_SO),
    (SecondOrder(S20, 0.25, "cos"), R_SO),
    (SecondOrder(S0, 0.5, "cos"), R_SO),
    (SecondOrder(S0, 0.25, "cos"), R_SO),
    (SecondOrder(U20, 0.25, "cos"), R_SO + " (unstandardised base)"),
    (SecondOrder(U5, 0.25, "cos"), R_SO + " (unstandardised base)"),
    (RowZSO(SecondOrder(S20, 0.5, "cos")), R_RZSO),
    (RowZSO(SecondOrder(U20, 0.5, "cos")), R_RZSO),
    # round 2 (after screening round 1 of this cache): rowz of socos_q0.5 on the std base was best (0.429, +0.04 over
    # its input) and q0.25 beat q0.5 on every base -> combine (rowz of q0.25) and test the q trend once more (q0.1)
    (RowZSO(SecondOrder(S20, 0.25, "cos")), R_RZSO2 := "round 2: rowz(SO) best and q0.25 > q0.5 -> rowz of q0.25"),
    (SecondOrder(S20, 0.1, "cos"), "round 2: q trend (0.25 > 0.5 > 1 > 2) -> q0.1"),
    (RowZSO(SecondOrder(S20, 0.1, "cos")), "round 2: q trend -> rowz of q0.1"),
]


def sha256(path=None, arrays=()):
    h = hashlib.sha256()
    if path is not None:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def base_of(cfg):
    while not isinstance(cfg, Spectral):
        cfg = cfg.base
    return cfg


def main():
    t_all = time.perf_counter()
    v1 = json.loads((RESULTS / "layers" / "manifest.json").read_text())
    names_v1 = {c["name"] for c in v1["columns"]}
    for cfg, _ in GRID:
        if cfg.name in names_v1:
            raise ValueError(f"{cfg.name} already in screen_v1")
    pairs, _ = data.load_pairs()
    sc = pairs.loc[pairs.scorable, ["a", "b", "ia", "ib"]].reset_index(drop=True)   # labels dropped here
    ia, ib = sc.ia.values.astype(np.int64), sc.ib.values.astype(np.int64)
    P1 = np.load(OUT / "screen_v1.pairs.npz")
    assert np.array_equal(P1["ia"], ia) and np.array_equal(P1["ib"], ib)
    eng = layers.get_engine("CRISPRGeneEffect")
    na, nb = layers.null_pairs(eng)
    assert np.array_equal(na, P1["null_a"]) and np.array_equal(nb, P1["null_b"])
    M = np.full((len(sc), len(GRID)), np.nan, dtype=np.float32)
    cols = []
    order = sorted(range(len(GRID)), key=lambda j: (base_of(GRID[j][0]).name, j))   # group by base (cache reuse)
    cur = None
    for j in order:
        cfg, why = GRID[j]
        b = base_of(cfg)
        if b != cur:
            eng.release()
            layers_ext.release()
            cur = b
        t0 = time.perf_counter()
        s = layers_ext.compute(cfg, ia, ib, eng)
        t_pairs = time.perf_counter() - t0
        t0 = time.perf_counter()
        mo = layers.moments(layers_ext.compute(cfg, na, nb, eng))
        t_null = time.perf_counter() - t0
        M[:, j] = s
        cols.append(dict(col=j, name=cfg.name, config=cfg.to_dict(), rationale=why, **mo,
                         n_nonfinite=int((~np.isfinite(s)).sum()), runtime_pairs_incl_prep_s=round(t_pairs, 3),
                         runtime_null_s=round(t_null, 3)))
        log.info("%-45s pairs(+prep) %.1fs null %.1fs  null mean %.4g sd %.4g", cfg.name, t_pairs, t_null,
                 mo["null_mean"], mo["null_sd"])
    cols.sort(key=lambda c: c["col"])
    f_scores, f_pairs = OUT / f"{TAG}.scores.f32.npy", OUT / f"{TAG}.pairs.npz"
    np.save(f_scores, M)
    np.savez(f_pairs, ia=ia, ib=ib, null_a=na, null_b=nb)
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout  # noqa
    manifest = dict(
        tag=TAG, created=time.strftime("%Y-%m-%dT%H:%M:%S"), script="scripts/12_build_layers_v2.py",
        git=dict(commit=run("rev-parse", "HEAD").strip(),
                 dirty_files=[l[3:] for l in run("status", "--porcelain", "--", "src", "scripts").splitlines() if l]),
        label_free="layer scores use only CRISPRGeneEffect and gene indices; no label is read. Layer CHOICE was "
                   "informed by the screen_v1 dev results (dev-selected; see rationale per column). Columns 0-16 = round 1 (chosen from screen_v1), 17-19 = round 2 (chosen from screen_v1 + round 1).",
        selection_basis="results/phaseA/README.md (screen_v1, SEEDS[0] dev CV)",
        input=v1["input"], preprocessing=dict(v1["preprocessing"], eig=eng.info),
        pairs=dict(n=len(sc), sha256_ia_ib=sha256(arrays=(ia, ib)), file=str(f_pairs.relative_to(ROOT)),
                   same_as="screen_v1"),
        null=v1["null"],
        output=dict(file=str(f_scores.relative_to(ROOT)), shape=list(M.shape), dtype="float32", sha256=sha256(f_scores)),
        runtime=dict(total_s=round(time.perf_counter() - t_all, 1),
                     peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)),
        columns=cols)
    txt = json.dumps(manifest, indent=1)
    (OUT / f"{TAG}.manifest.json").write_text(txt)
    (RESULTS / "layers" / f"{TAG}.manifest.json").write_text(txt)
    log.info("wrote %s %s in %.0fs", f_scores, M.shape, manifest["runtime"]["total_s"])


if __name__ == "__main__":
    main()
