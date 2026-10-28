# Round 1 review: DepMap data and biology

All numbers computed from `data_cache/` on the 3,191 pos / 593,406 neg scorable pairs. Scores are label-free, pooled, descriptive (not selection results; IFT not used).

## Facts checked
- **ID mapping.** 1,114/1,151 UniProts map uniquely (Entrez-based, no shared genes). Unmapped: 33 absent from DepMap, 2 ambiguous histones (P04908, P62805), 2 not in HGNC. This costs 123 positives (3.7%): mtDNA-encoded ATP-synthase subunits (P00846, P03928, Q5VTU8: 20 each), Ig chains, haemoglobin, CDK11A/B. Only the histones are recoverable. The rule is sound.
- **CGE.** 1,361 models; 1,048 carry an Avana flag, 221 Humagne, 316 KY; 227 models are multi-library. NaN is library-driven: 648 genes are NaN in ~83.5% of models (non-Avana-only genes). Only 15 labelled genes have >50% NaN, so NaN barely affects labelled-pair scoring. It matters for genome-wide covariance.
- **SGE.** 1,611 screens for 1,363 models; 227 models have several screens (non-independent rows). Library is only in the screen-ID suffix.
- **Confounders.** Per ModelID, not per screen. CasActivity is NaN in 150/1,361 models, DoublingTime in 186. Collinear: MedianEssentialDepletion vs MADEssentials r=-0.89, Type3DO vs GrowthPatternDome 0.93.
- **OR genes.** 329 in the matrix, **0 among the 1,114 labelled genes**.
- **Chromosome.** Same-chromosome 6.2% pos vs 5.1% neg; same-arm 3.6% vs 3.0%. Mean r of same-arm negatives 0.005 vs 0.003 for different chromosomes. Excluding same-arm pairs changes AP by 0.003. Proximity artefact negligible here.
- **Complex composition.** No ribosomal, mitoribosomal or complex-I genes (0 RPL/RPS, MRP, NDUF), since >30-subunit complexes were removed. Real dominance: one merged 144-protein component (SAGA/spliceosome/chromatin) holds 25.6% of positives; the top 3 components (that, Mediator, V-ATPase/NuRD-type) hold 43%; 132 components have ≤3 proteins and hold 211 positives.
- **Essentiality.** 31% of labelled genes have mean effect < -0.5 vs 9% of the genome. Gene essentiality alone (-max mean) gives AUROC 0.68, AP 0.019 (3.6x prevalence). 40% of positives have both genes non-essential on average; Pearson AP in that stratum is 0.077 vs 0.29-0.32 in essential strata.

## Library and confounder experiments (pooled AP, Pearson / Spearman)
| Input | P | S |
|---|---|---|
| CGE all 1,361 | 0.204 | 0.227 |
| CGE Avana-flag rows (1,048) | 0.223 | 0.221 |
| CGE non-Avana rows (313) | 0.107 | 0.099 |
| 313 random Avana rows | 0.155-0.185 | |
| CGE Avana-only models (827) | 0.185 | 0.186 |
| SGE AV01 / KY / CD / all | 0.149 / 0.109 / 0.052 / 0.181 | |
| CGE remove top-10 / 20 / 40 PCs | 0.292 / 0.296 / 0.264 | 0.303 / 0.309 / 0.281 |
| regress libraries / all QC columns | 0.227 / 0.251 | |
| regress all, then PC10 | 0.299 | 0.311 |
| lineage dummies | 0.222 | |

After PC10 removal: Avana-flag subset 0.296, all rows 0.292, Avana-only models 0.253. Within-lineage correlation alone is weak (0.02-0.13 per lineage; mean over lineages 0.286; max over lineages 0.085). PC1 has R²=0.45 with library/QC and 0.31 with lineage; PC2-8 are mostly lineage.

## Interpretation
Joint-Chronos CGE beats per-library SGE by 0.07 AP on the same models, so A3 is low-yield. More rows beat library purity. PCs absorb most batch and lineage structure; the plateau over k is broad (5-40 gives 0.28-0.31). Explicit regression adds only ~0.01 on top of PCs and risks removing real signal for common-essential complexes.

## Proposed changes
1. **High: CGE + PC removal/whitening is the primary input; demote SGE to exploratory (A3).** If SGE is used, collapse duplicate screens per model and take the library from the screen ID.
2. **High: no Avana restriction by default.** Keep all 1,361 models. Only test row-weighting of non-Avana models as a secondary step.
3. **High: estimate PCs and covariance genome-wide** (genes with NaN <10%, ~17.5k; per-gene mean imputation), never on the labelled-gene set (31% essential vs 9% genome, which would shift the PCs relative to the test universe). Choose k by component-disjoint dev CV, coarse grid.
4. **Medium: confounder regression is optional, applied before PCs.** Use QC columns, library flags and CasActivity/DoublingTime with missingness indicators. Drop collinear Dome/Type columns and the TKOv3/Brunello flags (9 models); use lstsq/ridge. Check it does not hurt per-component AP for common-essential complexes.
5. **Medium: gene-reliability features.** Per-gene SD after PC removal and fraction dependent, as pair min/max, so the supervised model can discount correlations between low-variance genes (the weak 40%-of-positives stratum).
6. **Medium: A5 rank normalisation uses the genome-wide gene background**, not the labelled set; the test pairs have a different universe.
7. **Medium: fix control 3.7(d).** No labelled gene is an OR. Compute the OR x OR similarity genome-wide as a null for whitening quality and z-scoring.
8. **Low: chromosome features.** Keep same-chromosome/arm as a diagnostic or single feature; skip cytoband distance (HGNC has band resolution only).
9. **Low: lineage.** Treat as optional dummies in the confounder step; drop stratified or max-over-lineage correlation.
10. **Low: histones.** Optionally map P62805/P04908 to the mean profile of their DepMap copies (~5 positives).
11. **Reporting:** add macro AUPRC with the 144-protein component and top-3 shown separately, and AUPRC stratified by pair essentiality (both essential / one / neither). Optional ablation: weight training pairs by 1/sqrt(component size).
