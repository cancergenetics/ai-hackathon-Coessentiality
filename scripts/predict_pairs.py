"""Score an arbitrary UniProt pair file with a frozen label-free method (src/predict.py; DESIGN §3.10, §5).

Input: two tab-separated UniProt columns, no header (like datasets/HuMAP/Train_*_pairs.txt).
Output TSV (input order and orientation, with header): uniprot_a, uniprot_b, score, mapped [, z_<code> ...];
sidecar run record <out>.run.json (git commit, input SHA-256s, null-moment checks, counts, runtimes).

Methods
  single | B           rowz[socos_q0.25[spec_k20_t0.1_std]]  (D3 C_single; raw value)
  E | H | F            emb_lap512[...] | rowz[cos_heat1[...]] | spec_k20_t0.1_std  (raw value)
  gls | pearson | spearman   baselines (GLS signed t, Pearson r, Spearman rho; raw value)
  zavg:B,E,F           equal-weight mean of frozen-null z-scores of any component subset (codes B E H F G P S)
  comb_zavg_BE, comb_zavg_BEF, ...   same as zavg:B,E / zavg:B,E,F
  --model PATH         fitted combination (JSON zavg/linear spec, or pickle with .components/.predict(Z)); overrides
Unmapped proteins / same-gene pairs / undefined scores: score = min(mapped scores) - 1, mapped = 0.

Usage:
  ./.env/bin/python scripts/predict_pairs.py --pairs FILE --method comb_zavg_BE --out scores.tsv [--components]
      [--cache] [--record PATH]
--cache reads data_cache/CRISPRGeneEffect.f32.npy instead of the raw CSV (default: raw CSV, hash-checked either way).
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import predict  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="tab-separated UniProt pair file, no header")
    ap.add_argument("--method", help="single | B|E|H|F | gls|pearson|spearman | zavg:B,E,... | comb_zavg_BE...")
    ap.add_argument("--model", help="fitted model file (JSON or pickle); overrides --method")
    ap.add_argument("--out", required=True, help="output TSV")
    ap.add_argument("--components", action="store_true", help="also write per-component z-scores (z_<code>)")
    ap.add_argument("--cache", action="store_true", help="read the cached float32 matrix instead of the raw CSV")
    ap.add_argument("--record", help="sidecar JSON path (default <out>.run.json)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    method = predict.parse_method(a.method, a.model)
    rec = predict.run(a.pairs, method, a.out, components=a.components, cache=a.cache, record_path=a.record)
    print(json.dumps(dict(method=method.name, out=a.out, counts={k: rec["counts"][k] for k in
                                                                 ("n_pairs", "n_mapped", "n_unmapped")},
                          runtime_s=rec["runtime"]["total_s"]), indent=1))


if __name__ == "__main__":
    main()
