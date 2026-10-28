"""Build the Phase-A screening cache of label-free similarity layers (DESIGN v3 §4 A1/A3/A4/A6).

For every scorable pair of the project pair table (src/data.py; only gene indices are read, never labels) computes
each layer of ``layers.screening_grid()`` and stores an (n_pairs x n_layers) float32 matrix in
data_cache/layers/ (gitignored). Each layer is also scored on the fixed genome-wide null sample
(200k random background gene pairs, seed SEEDS[0]) to freeze its z-moments. The JSON manifest (config, null
moments and runtimes per column, input/output hashes, git commit) goes to results/layers/manifest.json.

Usage: ./.env/bin/python scripts/10_build_layers.py [--tag screen_v1] [--limit N]   (--limit: smoke test only)
"""
import argparse
import hashlib
import json
import logging
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from src import data, layers  # noqa: E402
from src.config import CACHE, RESULTS, ROOT, SEEDS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("10")
OUT = CACHE / "layers"


def sha256(path=None, arrays=()):
    h = hashlib.sha256()
    if path is not None:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def git_state():
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout  # noqa
    dirty = [l[3:] for l in run("status", "--porcelain", "--", "src", "scripts", "tests").splitlines() if l]
    return dict(commit=run("rev-parse", "HEAD").strip(), dirty_files=dirty,
                layer_code_dirty=any(f in dirty for f in ("src/layers.py", "scripts/10_build_layers.py")))


def scorable_gene_pairs():
    pairs, _ = data.load_pairs()
    sc = pairs.loc[pairs.scorable, ["a", "b", "ia", "ib"]].reset_index(drop=True)   # labels dropped here
    return sc


def main(tag, limit):
    t_all = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    (RESULTS / "layers").mkdir(parents=True, exist_ok=True)
    sc = scorable_gene_pairs()
    if limit:
        sc = sc.iloc[:limit]
    ia, ib = sc.ia.values.astype(np.int64), sc.ib.values.astype(np.int64)
    t0 = time.perf_counter()
    eng = layers.get_engine("CRISPRGeneEffect")
    t_engine = time.perf_counter() - t0
    na, nb = layers.null_pairs(eng)
    grid, bases = layers.screening_grid()
    log.info("%d pairs (%d genes), %d layers, %d null pairs, BG %d genes", len(sc), len(np.unique(np.r_[ia, ib])),
             len(grid), len(na), eng.n_bg)

    M = np.full((len(sc), len(grid)), np.nan, dtype=np.float32)
    cols, base_prep = [None] * len(grid), {}
    order = [i for i, c in enumerate(grid) if isinstance(c, layers.Spectral)]
    for b in bases:                                    # derived layers grouped by base, so caches are reused
        order += [i for i, c in enumerate(grid) if not isinstance(c, layers.Spectral) and c.base == b]
    assert sorted(order) == list(range(len(grid)))
    current_base = None
    for j in order:
        cfg = grid[j]
        prep = 0.0
        if isinstance(cfg, layers.Spectral):
            eng.release()
            t0 = time.perf_counter()
            eng.spectral(cfg)
            prep = time.perf_counter() - t0
        elif cfg.base != current_base:
            eng.release()
            current_base = cfg.base
            tp = {}
            t0 = time.perf_counter()
            eng.spectral(cfg.base)
            tp["spectral_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            eng.row_stats(cfg.base)
            tp["row_stats_s"] = time.perf_counter() - t0
            for bal in (False, True):
                t0 = time.perf_counter()
                eng.neighbours(cfg.base, bal, eng.n_neighbours(2.0))
                tp[f"neighbours_{'balanced' if bal else 'raw'}_s"] = time.perf_counter() - t0
            base_prep[cfg.base.name] = {k: round(v, 3) for k, v in tp.items()}
        t0 = time.perf_counter()
        s = eng.compute(cfg, ia, ib)
        t_pairs = time.perf_counter() - t0
        t0 = time.perf_counter()
        mo = layers.moments(eng.compute(cfg, na, nb))
        t_null = time.perf_counter() - t0
        M[:, j] = s
        cols[j] = dict(col=j, name=cfg.name, config=cfg.to_dict(), **mo,
                       n_nonfinite=int((~np.isfinite(s)).sum()),
                       runtime_prep_s=round(prep, 3), runtime_pairs_s=round(t_pairs, 3),
                       runtime_null_s=round(t_null, 3))
        log.info("%-45s prep %.2fs pairs %.2fs null %.2fs  null mean %.4g sd %.4g", cfg.name, prep, t_pairs, t_null,
                 mo["null_mean"], mo["null_sd"])
    eng.release()

    f_scores, f_pairs = OUT / f"{tag}.scores.f32.npy", OUT / f"{tag}.pairs.npz"
    np.save(f_scores, M)
    np.savez(f_pairs, ia=ia, ib=ib, a=sc.a.values.astype(str), b=sc.b.values.astype(str),
             null_a=na, null_b=nb)
    _, _, genes = data.load_matrix("CRISPRGeneEffect", mmap=True)
    manifest = dict(
        tag=tag, created=time.strftime("%Y-%m-%dT%H:%M:%S"), script="scripts/10_build_layers.py",
        args=dict(limit=limit), git=git_state(),
        env=dict(python=platform.python_version(), numpy=np.__version__, machine=platform.machine()),
        label_free=("layer scores use only CRISPRGeneEffect and gene indices; the pair table is read for (a, b, ia, "
                    "ib, scorable) to choose rows; no label is read"),
        input=dict(matrix="CRISPRGeneEffect", shape=[eng.n, eng.G],
                   sha256=sha256(CACHE / "CRISPRGeneEffect.f32.npy"), n_genes_listed=len(genes)),
        preprocessing=dict(bg_max_nan=eng.max_nan, n_background_genes=eng.n_bg, block=eng.block,
                           imputation="per-gene observed mean (all genes); BG genes only for Sigma / row moments / "
                                      "ranks / neighbours / null", eig=eng.info),
        pairs=dict(n=len(sc), n_genes=int(len(np.unique(np.r_[ia, ib]))),
                   sha256_ia_ib=sha256(arrays=(ia, ib)), file=str(f_pairs.relative_to(ROOT))),
        null=dict(n=len(na), seed=SEEDS[0], sampling="uniform BG x BG, a != b, with replacement",
                  sha256=sha256(arrays=(na, nb))),
        output=dict(file=str(f_scores.relative_to(ROOT)), shape=list(M.shape), dtype="float32",
                    sha256=sha256(f_scores)),
        runtime=dict(engine_init_s=round(t_engine, 3), base_prep=base_prep,
                     total_s=round(time.perf_counter() - t_all, 1),
                     peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)),
        columns=cols)
    txt = json.dumps(manifest, indent=1)
    (OUT / f"{tag}.manifest.json").write_text(txt)
    if not limit:
        (RESULTS / "layers" / "manifest.json").write_text(txt)
    log.info("wrote %s %s in %.0fs", f_scores, M.shape, manifest["runtime"]["total_s"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="screen_v1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.tag if not a.limit else f"{a.tag}_smoke", a.limit)
