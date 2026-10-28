# mvp/ — minimal standalone implementation of zavg_BEF

A compact re-implementation of the final method (package `coessential`), written so readers can use or re-implement it without the research harness. It reproduces the research code's scores to within 3×10⁻⁵ in z on all 634,779 hu.MAP train pairs (`tests/test_equivalence.py`, run when `COESS_RESEARCH_CACHE` points to a research `data_cache/`).

- `src/coessential/method.py` — the method (gene standardisation, PC removal + ridge whitening, F; genome-wide neighbour lists, B; Laplacian eigenmap, E; fixed-null z-scores; zavg_BEF)
- `src/coessential/baselines.py` — Pearson, Spearman, GLS
- `src/coessential/data.py` — UniProt → HGNC → Entrez → DepMap mapping
- `scripts/predict_pairs.py`, `scripts/predict_genome_wide.py`, `scripts/evaluate.py` — command-line tools (run from the repository root so the default `datasets/` paths resolve)
- `tests/` — toy-data tests with known answers

```bash
pip install -r requirements.txt
cd .. && python mvp/scripts/predict_pairs.py --pairs pairs.txt --method zavg_BEF --out scores.tsv
```

Methods: `zavg_BEF` (default), `B`, `E`, `F`, `gls`, `pearson`, `spearman`. Options: `--components` adds `z_B, z_E, z_F`; `--add-gls` adds the GLS t; unmapped pairs get `mapped = 0` and the minimum score − 1 (or `--unmapped na`). Hyper-parameters are frozen at the published values.
