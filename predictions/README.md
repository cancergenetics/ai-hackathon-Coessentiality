# Precomputed predictions

This file was generated with this repository's code. The input was the raw DepMap **26Q3** `CRISPRGeneEffect.csv`
(1,361 models × 18,434 genes; SHA-256 `1e0c4929…7504ee`) plus the HGNC `hgnc_complete_set.txt` (SHA-256 `0f2d3f03…72c247`).
It is tab-separated and gzip-compressed, and empty fields mean "not available". All z-scores are relative to 200,000 fixed random
genome-wide gene pairs, so z ≈ 0 for a random pair and z = 3 is three null SDs above it.

**Attribution.** These predictions are derived from DepMap data, which DepMap (Broad Institute) releases under
CC BY 4.0. If you reuse them, credit DepMap: cite the DepMap 26Q3 release as described on the DepMap portal
(<https://depmap.org/portal/>) and Dempster et al. 2021, *Genome Biol* 22:343. Gene symbols and UniProt IDs come from HGNC.

Scores for hu.MAP pairs are not distributed here. Generate them from your own copy of the hu.MAP files with
`scripts/predict_pairs.py --components --add-gls` (see the main README).

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
python scripts/predict_genome_wide.py --top 200000 --float-format %.5f --out genome_wide_top_pairs.tsv.gz
```

This takes about 2–3 min and 2.5 GB RAM on a 13-inch MacBook Pro (Apple M2, 24 GB).

**Caveats.**
- A high co-essentiality score is evidence of shared function, not proof of physical co-complex membership. Top pairs also
  include signalling partners, pathway members and paralog or CRISPR-artefact pairs; same-chromosome olfactory-receptor
  clusters are slightly enriched.
- The score was validated only on hu.MAP 3.0 complexes. See the README, "Limitations".
