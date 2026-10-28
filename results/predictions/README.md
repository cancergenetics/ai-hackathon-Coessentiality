# Precomputed predictions

This file was generated with the standalone implementation in `mvp/` (verified against the research code to within 3×10⁻⁵ in z). The input was the raw DepMap **26Q3** `CRISPRGeneEffect.csv`
(1,361 models × 18,434 genes; SHA-256 `1e0c4929…7504ee`) plus the HGNC `hgnc_complete_set.txt` (SHA-256 `0f2d3f03…72c247`).
It is tab-separated and gzip-compressed, and empty fields mean "not available". All z-scores are relative to 200,000 fixed random
genome-wide gene pairs, so z ≈ 0 for a random pair and z = 3 is three null SDs above it.

**Attribution.** These predictions are derived from DepMap data, which DepMap (Broad Institute) releases under
CC BY 4.0. If you reuse them, credit DepMap: cite the DepMap 26Q3 release as described on the DepMap portal
(<https://depmap.org/portal/>) and Dempster et al. 2021, *Genome Biol* 22:343. Gene symbols and UniProt IDs come from HGNC.

Scores for hu.MAP pairs are not distributed here. Generate them from your own copy of the hu.MAP files with
`mvp/scripts/predict_pairs.py --components --add-gls` (see the main README).

## `genome_wide_top_pairs.tsv.gz`

- **Rows:** the 200,000 highest-scoring gene pairs by zavg_BEF.
- **Gene set:** all pairs of DepMap genes with < 10% missing values (17,525 genes, about 1.5×10⁸ pairs).
- **Selection:** pairs were ranked on genome-wide score blocks, and the reported values were recomputed exactly for each selected pair.
- **Order:** best first, ties broken by gene index.
- **Range:** the lowest zavg_BEF in the file is 4.38.

| column | definition |
|---|---|
| `rank` | 1 = highest zavg_BEF |
| `symbol_a`, `symbol_b` | HGNC approved symbol (matched on Entrez ID; the DepMap symbol if the Entrez ID is not in HGNC) |
| `entrez_a`, `entrez_b` | NCBI Entrez gene ID (from the DepMap column name `SYMBOL (ENTREZ)`) |
| `uniprot_a`, `uniprot_b` | HGNC `uniprot_ids` for that gene ('\|'-separated if several; empty if none, 104 pairs) |
| `zavg_BEF` | final score = (z_B + z_E + z_F) / 3; higher = more likely co-complex |
| `z_B` | z-score of B, the second-order (shared co-essential neighbours, top 0.25%) similarity, genome-wide row-z |
| `z_E` | z-score of E, the cosine in the 512-d Laplacian eigenmap of the genome-wide neighbour graph |
| `z_F` | z-score of F, the correlation of the PC-filtered (top 20 removed), ridge-whitened (τ = 0.1) profiles |

Regenerate with:

```bash
python mvp/scripts/predict_genome_wide.py --top 200000 --float-format %.5f --out genome_wide_top_pairs.tsv.gz
```

This takes about 2–3 min and 2.5 GB RAM on a 13-inch MacBook Pro (Apple M2, 24 GB).

**Caveats.**
- A high co-essentiality score is evidence of shared function, not proof of physical co-complex membership. Top pairs also
  include signalling partners, pathway members and paralog or CRISPR-artefact pairs; same-chromosome olfactory-receptor
  clusters are slightly enriched.
- The score was validated only on hu.MAP 3.0 complexes. See the README, "Limitations".

## All protein pairs (`outputs/all_pairs/`, local only — not in git)

Scores for **every** unordered pair of protein-coding DepMap genes with the frozen final method (zavg_BEF and its
components z_B, z_E, z_F; published hyper-parameters, the same fixed null moments). The files are large (≈3.4 GB) and
live only in the local, gitignored `outputs/all_pairs/`. The small, committed `all_pairs_manifest.json` (this
directory) records their checksums, counts, score distribution, runtime and check results.

```bash
python mvp/scripts/predict_all_pairs.py --out-dir outputs/all_pairs \
    --manifest-copy results/predictions/all_pairs_manifest.json      # ~2.5 min, 6.6 GB peak RAM (Apple M2)
python scripts/verify/check_all_pairs.py                               # checks (a)-(c), ~5 min; needs the research code
```

**Definitions**
- **Gene universe:** the 18,434 DepMap 26Q3 `CRISPRGeneEffect` genes.
- **Protein gene:** the Entrez ID in the DepMap column name `SYMBOL (ENTREZ)` is the `entrez_id` of an HGNC *Approved*
  record with ≥ 1 `uniprot_ids` accession. 18,429 genes qualify; 5 are excluded because their HGNC record lists no
  UniProt accession (ABHD4, C10orf90, GUCA1ANB-GUCA1A, HSD17B8, UBE2O). 18,390 protein genes have one accession, 38 have two and 1 has four.
- **Low coverage:** NaN fraction ≥ 10% (908 protein genes). They are not *background* genes of the method, so they
  never enter the covariance, the row moments, the neighbour sets or the null. The method still scores them, exactly
  as `predict_pairs` does: mean imputation, their top-K background neighbours and a Nyström embedding. They are
  included and flagged (`low_coverage` in `genes.tsv`, `low_coverage_any` in the pair table).
- **Pairs:** all i < j over the 18,429 protein genes (169,804,806 pairs; 153,483,960 among the 17,521 background protein genes).

**Files**

| file | content |
|---|---|
| `zavg_BEF.npy` | float32, 18,429 × 18,429 (1.36 GB) |
| `z_B.npy`, `z_E.npy`, `z_F.npy` | float16, 18,429 × 18,429 (0.68 GB each); rounding is ≤ 2⁻¹¹·\|z\|, i.e. up to 0.016 for \|z\| ≥ 32 |
| `genes.tsv` | `index` (matrix row/column), `symbol` (HGNC), `entrez`, `uniprot_ids` ('\|'-joined), `hgnc_id`, `low_coverage` (0/1), `nan_fraction`, `depmap_column`, `depmap_index` |
| `top_pairs.parquet` | every pair with zavg_BEF ≥ 3 (667,677 rows, 20 MB). Columns: `symbol_a`, `symbol_b`, `entrez_a`, `entrez_b`, `uniprot_a`, `uniprot_b`, `zavg_BEF`, `z_B`, `z_E`, `z_F` (float32), `rank` (1 = best; ties by index), `low_coverage_any`; a < b in the `genes.tsv` index |
| `manifest.json` | input and file SHA-256s, code commit, counts, dtypes and shapes, runtime, peak RAM, zavg_BEF quantiles over all pairs, counts ≥ 2/3/4/5, check results |

- **Matrix layout:** the upper triangle (row < column) is the computed value. The lower triangle is its exact mirror, and the diagonal is NaN.
- **How values are computed:** they come from the same fixed genome-wide blocks that `predict_genome_wide.py` ranks on, and they agree with per-pair `predict_pairs` scores to ≤ 3×10⁻⁶ before rounding.
- **Score distribution over all pairs:** zavg_BEF has median −0.09 and SD 0.70; the 99.9th percentile is 4.91. Pair counts by threshold:

| zavg_BEF | pairs |
|---|---|
| ≥ 2 | 2,433,687 |
| ≥ 3 | 667,677 |
| ≥ 4 | 278,986 |
| ≥ 5 | 163,445 |

```python
import numpy as np, pandas as pd
genes = pd.read_csv("outputs/all_pairs/genes.tsv", sep="\t")
Z = np.load("outputs/all_pairs/zavg_BEF.npy", mmap_mode="r")      # memory-mapped, nothing loaded yet
i = genes.index[genes.symbol == "TSC1"][0]; j = genes.index[genes.symbol == "TSC2"][0]
print(Z[i, j])                                                     # 36.44
row = np.asarray(Z[i], dtype=np.float64)                           # one gene against all protein genes
top = genes.symbol.values[np.argsort(-np.nan_to_num(row, nan=-np.inf))[:10]]
pairs = pd.read_parquet("outputs/all_pairs/top_pairs.parquet")     # zavg_BEF >= 3
```

**Checks** (`scripts/verify/check_all_pairs.py`; results in the manifest under `checks`):
- **(a) Agreement with per-pair scoring.** 2,000 random protein pairs (seed 0; 203 involve a low-coverage gene), 500 pairs with ≥ 1 low-coverage gene and 199 with two were scored by the research `predict_pairs` pipeline (UniProt input for 2,689 pairs, gene indices for 10) and by the mvp scorer.
  - zavg_BEF: max |Δ| ≤ 2.4×10⁻⁶ (tolerance 1×10⁻⁴).
  - z_B/E/F: max |Δ| ≤ 0.008 (tolerance 1×10⁻²).
- **(a) Agreement with the existing genome-wide file.** 199,896 of its 200,000 pairs are protein × protein pairs; the other 104 involve the excluded genes.
  - zavg_BEF: max |Δ| = 5.9×10⁻⁶.
  - z_E, z_F: max |Δ| = 0.008.
  - z_B: 1,032 pairs with |z_B| ≥ 32 differ by up to 0.0156. That is float16 spacing (2⁻⁵), so the absolute 1×10⁻² tolerance cannot hold there. All pairs are within the float16 rounding bound.
- **(b) Matrix structure.** All four matrices are exactly symmetric, have a NaN diagonal and have no non-finite off-diagonal values. zavg_BEF equals the mean of the float16 components within their rounding.
- **(c) Reproduction of the existing top pairs.** The top 199,896 pairs of the matrix among background genes are exactly the existing file's protein × protein pair set (cutoff 4.381).
