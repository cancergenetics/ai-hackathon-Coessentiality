# Standalone pair scorer: `scripts/predict_pairs.py`

`src/predict.py` holds the code and `tests/test_predict.py` the tests. The scorer is label-free. It reads only `datasets/DepMAP/CRISPRGeneEffect.csv` and the HGNC table. It never reads pair labels, and no metric was computed here.

```
./.env/bin/python scripts/predict_pairs.py --pairs FILE --method METHOD --out scores.tsv [--components] [--cache] [--record PATH]
./.env/bin/python scripts/predict_pairs.py --pairs FILE --model model.json|model.pkl --out scores.tsv
```

**Input.** The pairs file has two tab-separated UniProt columns and no header, like `datasets/HuMAP/Train_*_pairs.txt`. Blank lines are skipped. Any extra columns are ignored.

**Methods.** Each code below is a frozen config that is checked against the manifest config.

| code | component | source |
|---|---|---|
| `B` = `single` | `rowz[socos_q0.25[spec_k20_t0.1_std]]` (D3 C_single) | screen_v2 col 17 |
| `E` | `emb_lap512[knn_union_lin_q0.25[spec_k20_t0.1_std]]` | graph_v1 col 52 |
| `H` | `rowz[cos_heat1[knn_union_lin_q0.25[spec_k20_t0.1_std]]]` | graph_v1 col 45 |
| `F` | `spec_k20_t0.1_std` | screen_v1 col 62 |
| `G` = `gls`, `P` = `pearson`, `S` = `spearman` | GLS signed t, Pearson r and Spearman ρ (best dev variants, `src/baselines.py`) | baseline cache |

- **Single component** (e.g. `single`, `E`, `gls`): the score is the raw component value.
- **`zavg:B,E,F`** (or `comb_zavg_BEF`, `comb_zavg_BE`, …): the score is the equal-weight mean of the components' z-scores.
- **`--model PATH`**: a hook for a fitted combination later. It takes either:
  - a JSON file `{"kind": "zavg"|"linear", "components": [...], "weights": [...], "intercept": x}`, or
  - a pickle of an object with `.components` and `.predict(Z)`, where `Z` maps each code to the z-scores of the mapped pairs.

**z-scores.** The null is the fixed genome-wide set of 200,000 random BG gene pairs from `src.layers.null_pairs` (seed 0). Its SHA-256 is asserted equal to the manifest's. Null moments are recomputed on every run.
- For B, E, H and F they are asserted equal to the manifest values (tolerance 1e-4 null SD). The observed difference was exactly 0 for all four.
- G, P and S have no manifest moments, so theirs are only recorded. The GLS null is mean 0.0454, SD 1.0120.

**Output.** The TSV keeps the input row order and orientation. Columns are `uniprot_a, uniprot_b, score, mapped`, plus `z_<code>` per component when `--components` is set.
- **Unmapped pairs.** A pair is unmapped if a protein does not map, both proteins map to the same gene or it is a self pair, or the score is undefined. It gets `score = min(mapped scores in the file) − 1` and `mapped = 0` (DESIGN §3.10). This floor is the only value that depends on the rest of the file.
- **Sidecar.** `<out>.run.json`, or the path given with `--record`, holds:
  - the git commit and dirty files;
  - SHA-256 of the pair file, the raw CSV and HGNC;
  - a check that the float32 matrix's `np.save` hash equals the manifest `input.sha256`;
  - null-moment checks;
  - mapping counts;
  - stage runtimes and peak RSS.

**Reproducibility and invariance.**
- By default the matrix is recomputed from the raw CSV. `--cache` uses `data_cache/CRISPRGeneEffect.f32.npy` instead; it is hash-checked the same way.
- IDs are mapped with `src.data.build_uniprot_map`, the same rule the rest of the project uses.
- Pairs are canonicalised to (min, max) gene index and de-duplicated before scoring. The layers use genome-wide state computed in fixed gene blocks.
- GLS warps genes in fixed blocks of 1024 genes. Pearson, Spearman and GLS take per-pair row dot products. As a result, a mapped pair's score is bit-identical whatever the other pairs are and in either (a, b) order (tested).

## Runtime: union of both train pair files (IDs only)

The input is the union of both train pair files: 634,779 pairs, shuffled, IDs only. 596,597 pairs mapped and 38,182 did not, all because a protein was not mapped. The mapped set equals the cached scorable universe exactly.

Runs read the raw CSV, on 8 cores (arm64, Accelerate BLAS). "Total" is the wall clock inside the tool; `/usr/bin/time` real time is about 2 s more because of interpreter start. The records are `results/predict/union_*.run.json`.

| method | total (s) | matrix load + hash (s) | scoring by component (s), including genome-wide prep and null | peak RSS (GB) |
|---|---|---|---|---|
| `single` (B) | 28.4 | 3.4 | B 22.1 | 3.2 |
| `zavg:B,E` | 133.7 | 3.8 | B 20.3, E 106.3 | 4.0 |
| `zavg:B,E,F` | 136.9 | 3.8 | F 6.1, B 20.3, E 103.2 | 4.1 |
| `gls` | 24.2 | 4.9 | G 16.5 | 2.5 |
| `gls --components` (needs the GLS null) | 86.6 | 3.1 | G 80.2 (null 67.7) | 2.8 |

H is not part of these runs. It costs about 60 s, from the graph_v1 build log (`rowz[cos_heat1]` 59 s).

**Agreement with the caches.** This compares IDs and scores only; no label was read. Across all 596,597 mapped union pairs, the maximum absolute difference from the cached columns is:
- B 1.9e-6
- E 1.1e-16
- F 3.0e-8
- GLS 5.7e-14

The cached layers are float32. For B, E and F the difference was computed from the `z_` columns with the recorded moments.

The single and zavg records come from commit b8fdf0f, the GLS records from 3c54722. The only changes between the two touch the baseline engine's memory use: a bounded whitening cache and chunked row dot products. GLS output was byte-identical before and after.
