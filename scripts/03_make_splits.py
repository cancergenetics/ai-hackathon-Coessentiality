"""DESIGN v2 §3.1-3.4: clusters, stratified IFT/dev protein assignment, dev CV fold assignments.
# NOTE: must be run with PYTHONHASHSEED=0 (src/clusters.build_all enforces it): PYTHONHASHSEED=0 python scripts/03_make_splits.py

Writes
  data_cache/splits/protein_assignment.csv        (gitignored; regenerated from SEEDS[0])
  results/splits/protein_assignment.sha256        (committed; verified by src.clusters.load_assignment)
  results/splits/clusters_summary.json            (counts only: clusters, IFT/dev/cross pair counts, strata, giant
                                                   component fractions, Kish n_eff, dev CV fold sizes and
                                                   fingerprints for SEEDS 0..4 and the secondary schemes)
No metrics are computed here. Refuses to overwrite an existing assignment whose hash differs from the committed
one unless --force.

Usage: ./.env/bin/python scripts/03_make_splits.py [--force]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from src import clusters as C  # noqa: E402
from src.config import CACHE, ROOT, SEEDS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    subprocess.run(["sh", str(ROOT / "scripts" / "check_data.sh")], check=True, stdout=subprocess.DEVNULL)
    allp = pd.read_pickle(CACHE / "pairs.pkl")
    assign, summ = C.build_all(allp, seed=SEEDS[0])
    b = C._csv_bytes(assign)
    new = C.sha256_bytes(b)
    if C.SHA_FILE.exists():
        old = C.SHA_FILE.read_text().split()[0]
        if old != new and not a.force:
            raise SystemExit(f"assignment hash would change ({old[:12]} -> {new[:12]}); use --force deliberately")
    C.write_assignment(assign)
    summ["assignment_sha256"] = new
    summ["params"] = dict(max_cluster_size=C.MAX_CLUSTER_SIZE, ift_fraction=C.IFT_FRACTION, n_top=C.N_TOP,
                          min_neff_ratio=C.MIN_NEFF_RATIO, seed=SEEDS[0], louvain="networkx louvain_communities, resolution 1")

    from src.evaluate import make_dev_split
    from src.universe import load_dev_universe
    u = load_dev_universe()
    summ["dev_universe"] = dict(n_pairs=len(u), n_pos=int(u.y.sum()), f_dev=u.f, largest_component=u.largest_component)
    cv = {}
    for scheme in ("cluster", "component", "gene", "random"):
        for seed in SEEDS:
            sp = make_dev_split(u, scheme, seed)
            cv[f"{scheme}_seed{seed}"] = dict(fingerprint=sp.fingerprint(), stats=sp.stats(u.y))
    fps = [cv[f"cluster_seed{s}"]["fingerprint"] for s in SEEDS]
    summ["dev_cv"] = cv
    summ["dev_cv_cluster_seeds_distinct"] = len(set(fps)) == len(SEEDS)
    out = C.SUMMARY_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: v for k, v in summ.items() if k != "dev_cv"}, indent=1, default=str))
    for s in SEEDS:
        st = cv[f"cluster_seed{s}"]["stats"]
        print(f"cluster seed {s}:", "; ".join(f"f{d['fold']} {d['test_pos']}/{d['test_neg']}" for d in st))


if __name__ == "__main__":
    main()
