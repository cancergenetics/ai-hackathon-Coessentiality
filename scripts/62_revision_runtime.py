"""Revision M2: raw-CSV -> scores runtimes through scripts/predict_pairs.py (no labels, no metrics).

Input: the union of both train pair ID files (datasets/HuMAP/Train_{positive,negative}_pairs.txt, first two columns
only, i.e. IDs without labels), shuffled with SEEDS[0]; written to data_cache/revision/ (HuMAP-derived pair lists are
never committed; its SHA-256 is recorded). Every method is a fresh `predict_pairs.py` process reading the raw
CRISPRGeneEffect.csv, run one after another on the same machine state, wrapped in `/usr/bin/time -l` (wall clock
incl. interpreter start + maximum resident set size). The raw CSV is read once beforehand (page-cache warm-up) so
that no run pays a cold-disk penalty the others do not.

Methods (same order every time): pearson, spearman, E, F (single components alone), zavg:B,E,F (final method), gls.
Output: results/revision/runtime.json, run records results/revision/runtime/<method>.run.json; score TSVs stay in
data_cache/revision/ (pair lists).

Usage: ./.env/bin/python scripts/62_revision_runtime.py
"""
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from src.config import CACHE, ROOT, SEEDS  # noqa: E402
from src.predict import RAW_CSV, sha256_file  # noqa: E402

OUT = ROOT / "results" / "revision"
REC = OUT / "runtime"
WORK = CACHE / "revision"
METHODS = ["pearson", "spearman", "E", "F", "zavg:B,E,F", "gls"]
HUMAP = ROOT / "datasets" / "HuMAP"


def union_file() -> Path:
    path = WORK / "train_union_ids.tsv"
    if not path.exists():
        rows = []
        for f in ("Train_positive_pairs.txt", "Train_negative_pairs.txt"):
            for line in (HUMAP / f).read_text().splitlines():
                if line.strip():
                    a, b = line.split("\t")[:2]
                    rows.append(f"{a.strip()}\t{b.strip()}")
        order = np.random.default_rng(SEEDS[0]).permutation(len(rows))
        WORK.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows[i] for i in order) + "\n")
    return path


def safe(m):
    return re.sub(r"[^A-Za-z0-9]+", "_", m).strip("_")


def run_one(method, pairs):
    out = WORK / f"union_{safe(method)}.tsv"
    rec = REC / f"union_{safe(method)}.run.json"
    cmd = ["/usr/bin/time", "-l", sys.executable, str(ROOT / "scripts" / "predict_pairs.py"), "--pairs", str(pairs),
           "--method", method, "--out", str(out), "--record", str(rec)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    m = re.search(r"(\d+)\s+maximum resident set size", r.stderr)
    real = re.search(r"([\d.]+)\s+real", r.stderr)
    tool = json.loads(rec.read_text())
    return dict(method=method, wall_s_process=round(wall, 2), time_real_s=float(real.group(1)) if real else None,
                peak_rss_gb_time_l=round(int(m.group(1)) / 1e9, 3) if m else None,
                tool_total_s=tool["runtime"]["total_s"], matrix_load_s=tool["runtime"]["matrix_load_s"],
                engine_init_s=tool["runtime"]["engine_init_s"], components_s=tool["runtime"]["scoring"]["components_s"],
                map_s=tool["runtime"]["scoring"]["map_s"], tool_peak_rss_gb=tool["runtime"]["peak_rss_gb"],
                n_pairs=tool["counts"]["n_pairs"], n_mapped=tool["counts"]["n_mapped"],
                null_moment_checks={c: dict(rel_diff_mean=v.get("rel_diff_mean"), rel_diff_sd=v.get("rel_diff_sd"))
                                    for c, v in tool["null"]["moments"].items()},
                record=str(rec.relative_to(ROOT)), output_sha256=tool["output"]["sha256"])


def main():
    REC.mkdir(parents=True, exist_ok=True)
    pairs = union_file()
    t0 = time.perf_counter()
    with open(RAW_CSV, "rb") as fh:                       # page-cache warm-up (not timed as part of any run)
        while fh.read(1 << 24):
            pass
    warm = round(time.perf_counter() - t0, 2)
    load = os.getloadavg()
    runs = []
    for m in METHODS:
        r = run_one(m, pairs)
        r["loadavg_before"] = [round(x, 2) for x in load]
        runs.append(r)
        print(json.dumps({k: r[k] for k in ("method", "wall_s_process", "tool_total_s", "peak_rss_gb_time_l",
                                            "components_s")}), flush=True)
        load = os.getloadavg()
    out = dict(
        what="raw CRISPRGeneEffect.csv -> scores for the union of both train pair ID files, scripts/predict_pairs.py, "
             "one fresh process per method, sequential, same machine state",
        input=dict(pairs="data_cache/revision/train_union_ids.tsv (not committed: HuMAP-derived)",
                   pairs_sha256=sha256_file(pairs), n_pairs=runs[0]["n_pairs"], n_mapped=runs[0]["n_mapped"],
                   raw_csv=str(RAW_CSV.relative_to(ROOT)), raw_csv_warmup_read_s=warm),
        columns=dict(wall_s_process="wall clock of the whole process incl. interpreter start + imports (Python "
                                    "perf_counter around subprocess)",
                     time_real_s="/usr/bin/time -l real", peak_rss_gb_time_l="/usr/bin/time -l maximum resident set "
                                                                             "size (bytes / 1e9)",
                     tool_total_s="predict.run() total (matrix load + hash, scoring, writing TSV), as in "
                                  "results/predict/*.run.json"),
        machine=dict(platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count(),
                     python=platform.python_version(), numpy=np.__version__,
                     note="8-core arm64 laptop; desktop apps (browser, IDE) running, load average recorded per run"),
        runs={r["method"]: r for r in runs})
    (OUT / "runtime.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
