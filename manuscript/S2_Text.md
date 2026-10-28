# S2 Text. Methods that did not work

## Scope and conventions

We list the main approaches that did not improve the final method, zavg_BEF. Every number is copied from a committed file named in a "Source" line; nothing was recomputed.

**Phases.** Internal phase names used below: Phase A, label-free layers; A7, graph layers; Phase B v1 and v2, supervised stacking (v1 with 12 layers chosen before Phase A results, v2 with the final frozen layers); Phase C, supervised metric learning.

**Abbreviations.** AP, average precision (area under the precision–recall curve). HT-AP, AP with Horvitz–Thompson weights, which restore the natural share of between-cluster pairs in cluster-grouped test folds. CV, cross-validation. Dev, development set (2,299 positive, 320,229 negative pairs; 190 clusters). IFT, internal final test (held-out complexes, evaluated once). OOF, out-of-fold. CI, 95% paired cluster-bootstrap interval. PC, principal component. GLS, generalised least squares. OR, olfactory receptor. kNN, k-nearest-neighbour graph. TOM, topological overlap measure. PPR, personalised PageRank. LR, logistic regression. LGBM, LightGBM. MR, mutual rank.

**Notation.** k is the number of top PCs removed; τ is the ridge added to the eigenvalues (relative to their mean) before whitening. τ = 0 is GLS-like whitening; τ = ∞ is Pearson correlation after PC removal. q is the share of genes kept as neighbours (q = 0.25% is 44 genes). F, first-order similarity of gene-standardised profiles (k = 20, τ = 0.1). B, second-order neighbour cosine on F, then row-z. E, 512-dimensional Laplacian eigenmap. H, heat-diffusion profile cosine, then row-z.

**Metric and split.** Unless stated, numbers are dev HT-AP from cluster-grouped 5-fold CV. "Seed 0" is the fold mean for split seed 0, the selection split, so the best entries are optimistic. "5-seed mean" averages seeds 0–4. "Pooled-OOF Δ" is the HT-AP difference on pooled out-of-fold predictions.

**"All-pairs, exploratory"** marks numbers from round-1 reviewer probes (disclosure D1) and the first baseline run (M1). These used all 596,597 scorable pairs, including future IFT pairs, and report pooled AP. They are not comparable with dev numbers, and no selection was made from them.

## 1. Baseline variants

Each baseline got its best of 2 (Pearson, Spearman) or 8 (GLS) variants on dev seed 0; only these were run on seeds 1–4.

**Table A. Baseline variants (dev HT-AP, cluster CV).**

| Variant | Signed / absolute | NaN handling | OR correction | Seed 0, mean ± SD over folds | Pooled HT-AP, seed 0 [CI] | 5-seed mean |
|---|---|---|---|---|---|---|
| **Pearson (selected)** | signed | pairwise-complete | – | **0.238 ± 0.095** | 0.226 [0.160, 0.292] | **0.260** |
| Pearson | absolute | pairwise-complete | – | 0.225 ± 0.091 | 0.210 [0.147, 0.275] | – |
| **Spearman (selected)** | signed | pairwise-complete | – | **0.260 ± 0.095** | 0.245 [0.179, 0.315] | **0.282** |
| Spearman | absolute | pairwise-complete | – | 0.245 ± 0.092 | 0.228 [0.163, 0.295] | – |
| **GLS (selected)** | signed | exact per-pair marginalisation | no | **0.288 ± 0.098** | 0.272 [0.215, 0.343] | **0.312** |
| GLS | absolute | exact | no | 0.275 ± 0.091 | 0.257 [0.204, 0.327] | – |
| GLS | signed | exact | yes | 0.274 ± 0.097 | 0.255 [0.198, 0.328] | – |
| GLS | absolute | exact | yes | 0.270 ± 0.093 | 0.252 [0.195, 0.323] | – |
| GLS | signed | gene-mean imputation | no | 0.287 ± 0.100 | 0.271 [0.213, 0.342] | – |
| GLS | absolute | imputation | no | 0.275 ± 0.092 | 0.256 [0.201, 0.326] | – |
| GLS | signed | imputation | yes | 0.277 ± 0.099 | 0.259 [0.203, 0.332] | – |
| GLS | absolute | imputation | yes | 0.273 ± 0.094 | 0.255 [0.199, 0.327] | – |

Source: `results/baselines_v2/README.md`, `results/baselines_v2/best_variants.json`, `results/report/tables.md` (T2); M1 numbers below from `results/baselines/README.md`.

- **Signed beat absolute** for all three, so negative correlation added no useful signal on balance.
- **The OR correction lowered GLS** (0.274 vs 0.288; all-pairs M1: 0.234 vs 0.255). *Speculation:* it also drops models lacking OR guides.
- **NaN handling did not matter** (0.287 vs 0.288). Wainberg's protocol of dropping every model with a missing value would leave 6 of 1,361 models; it was not run.
- **A pseudo-inverse** gave the same AP as dropping the 4 dependent models (all-pairs M1: 0.2547 both).

## 2. First-order filter variants and data subsets

### 2.1 Spectral filter grid (dev)

**Table B. HT-AP by k (rows) and τ (columns), genes not standardised (dev, seed 0).** LW = Ledoit–Wolf shrinkage (τ ≈ 0.055).

| k | τ = 0 | 0.01 | 0.1 | 1 | 10 | ∞ | LW |
|---|---|---|---|---|---|---|---|
| 0 | 0.279 | 0.288 | 0.300 | 0.327 | 0.307 | 0.238 | 0.295 |
| 1 | 0.279 | 0.288 | 0.301 | 0.328 | 0.314 | 0.270 | 0.296 |
| 2 | 0.279 | 0.287 | 0.300 | 0.328 | 0.316 | 0.280 | 0.295 |
| 3 | 0.278 | 0.287 | 0.300 | 0.328 | 0.319 | 0.289 | 0.295 |
| 5 | 0.277 | 0.286 | 0.299 | 0.327 | 0.320 | 0.297 | 0.293 |
| 10 | 0.274 | 0.282 | 0.296 | 0.325 | 0.321 | 0.304 | 0.290 |
| 20 | 0.265 | 0.274 | 0.287 | 0.322 | 0.332 | 0.330 | 0.282 |
| 40 | 0.241 | 0.250 | 0.261 | 0.291 | 0.307 | 0.309 | 0.256 |

**Table C. Gene standardisation and other first-order variants (dev, seed 0).**

| Variant | Standardised genes | Not standardised |
|---|---|---|
| k = 0, τ = 0.1 | 0.361 | 0.300 |
| k = 5, τ = 0.1 | 0.362 | 0.299 |
| k = 20, τ = 0.1 (F) | 0.365 | 0.287 |
| k = 20, τ = 0.03 / 0.3 / 1 | 0.364 / 0.365 / 0.358 | – / – / 0.322 |
| k = 0, Ledoit–Wolf | 0.361 | 0.295 |
| k = 20, τ = ∞ (no whitening) | 0.327 | 0.330 |
| k = 0, τ = 0, cell-line-centred (Wainberg convention) | – | 0.278 (gene-centred: 0.279) |

Source (Tables B, C): `results/phaseA/README.md` (per-layer files in `results/phaseA/screen/`).

- **Without gene standardisation, tuning k and τ gave 0.241–0.332**; the best setting gained +0.053 over GLS-like whitening (0.279), about as much as gene standardisation alone (+0.061). GLS-like τ = 0 was the worst whitened setting; k = 40 hurt at every τ.
- **Ledoit–Wolf ≈ fixed ridge** (0.295 vs 0.300 for τ = 0.1, k = 0). **Cell-line centring** made no difference (0.278 vs 0.279).
- **Gene standardisation was the one change that mattered.** After it, results were flat for k ∈ {0, 5, 20} and τ ∈ [0.01, 0.3] (0.359–0.365); over 5 seeds k = 20 and k = 0 both gave 0.395 (`results/phaseB_v2/README.md`). *Likely, not tested:* without it, high-variance essential genes dominate the covariance.

### 2.2 Data subsets and confounder removal (all-pairs, exploratory)

None of these choices was tested on dev CV.

**Table D. Data-reviewer probes on all 596,597 scorable pairs (pooled AP; prevalence 0.0053).**

| Input or processing | Pearson | Spearman |
|---|---|---|
| CRISPRGeneEffect, all 1,361 models | 0.204 | 0.227 |
| Rows with an Avana library flag (1,048) | 0.223 | 0.221 |
| Non-Avana rows (313) | 0.107 | 0.099 |
| 313 random Avana rows | 0.155–0.185 | – |
| Avana-only models (827) | 0.185 | 0.186 |
| ScreenGeneEffect per library: AV01 / KY / CD / all | 0.149 / 0.109 / 0.052 / 0.181 | – |
| Remove top 10 / 20 / 40 PCs | 0.292 / 0.296 / 0.264 | 0.303 / 0.309 / 0.281 |
| Regress out library flags / all QC columns | 0.227 / 0.251 | – |
| Regress out all, then remove 10 PCs | 0.299 | 0.311 |
| Regress out lineage dummies | 0.222 | – |
| After removing 10 PCs: Avana-flag rows / all rows / Avana-only models | 0.296 / 0.292 / 0.253 | – |

Source: `reviews/round1/data_biology.md`.

**Table E. Algorithms-reviewer probes and M1 run on all scorable pairs (pooled AP).**

| Score | AP |
|---|---|
| GLS signed, exact / mean-imputed | 0.255 / 0.253 |
| GLS with shrunk covariance, α = 0.2 / 0.35 / 0.9 | 0.299 / 0.301 / 0.245 |
| Eigenvalue floor at the 90th percentile · PCA-whitening of the top 400 | 0.297 · 0.259 |
| Spearman of whitened profiles | 0.245 |
| GLS on ScreenGeneEffect Avana (1,051 models) vs CRISPRGeneEffect on the same pairs | 0.2437 vs 0.2573 |

Source: `reviews/round1/algorithms_compute.md` (§0); last row `results/baselines/README.md`.

- **Per-library inputs and Avana-only models were worse.** The Avana-flag subset was higher with raw Pearson (0.223 vs 0.204) but not after PC removal (0.296 vs 0.292). More rows beat library purity.
- **Confounder or lineage regression added at most about 0.01 over PC removal** (0.299 vs 0.292). PC1 had R² = 0.45 with library and QC covariates, and PCs 2–8 were mostly lineage, so PCs already absorb this structure.
- **Lineage-specific scoring was weak** (0.02–0.13 per lineage; maximum over lineages 0.085). The mean over lineages gave 0.286; not followed up on dev.

## 3. Balancing per gene

**Table F. Per-gene balancing of a base similarity (dev, seed 0).**

| Base | Raw | Row-z | MR | Balanced MR |
|---|---|---|---|---|
| Pearson (k = 0, τ = ∞) | 0.238 | 0.266 | 0.280 | 0.284 |
| Pearson after removing 20 PCs | 0.330 | 0.336 | 0.346 | 0.344 |
| k = 0, τ = 0.1, not standardised | 0.300 | 0.315 | 0.314 | 0.317 |
| k = 20, τ = 0.1, not standardised | 0.287 | 0.303 | 0.303 | 0.306 |
| **F** (standardised, k = 20, τ = 0.1) | 0.365 | 0.366 | – | 0.363 |
| Second-order cosine on F, q = 0.25% | 0.401 | **0.435 (B)** | – | – |
| Heat diffusion (t = 1) profile cosine | 0.402 | 0.442 (H) | – | – |
| Laplacian eigenmap, d = 512 (E) | 0.460 | 0.458 | – | – |

Source: `results/phaseA/README.md`, `results/phaseA_graph/README.md`.

- **Kept:** row-z of the second-order (B) and diffusion (H) scores.
- **Not kept:** row-z or MR of first-order scores, and row-z of E. Balancing helped weak bases but not F. MR ≈ row-z (±0.01); balanced MR ≈ plain MR.
- *Speculation:* whitening already evens out each gene's similarity distribution, whereas second-order scores depend on neighbourhood density, which row-z corrects.

## 4. Second-order variants

**Table G. Second-order neighbourhood scores (dev, seed 0).**

| Variant | Base | HT-AP |
|---|---|---|
| Weighted cosine, q = 2 / 1 / 0.5 / 0.25% | k = 20, τ = 0.1, not standardised | 0.339 / 0.352 / 0.362 / 0.369 |
| Jaccard (binary neighbour sets), q = 2 / 1 / 0.5% | same | 0.228 / 0.267 / 0.298 |
| Weighted cosine, q = 0.5 / 1 / 2% | raw Pearson (base alone 0.238) | 0.204 / 0.175 / 0.145 |
| Weighted cosine, q = 0.1 / 0.25 / 0.5%, no row-z | F | 0.412 / 0.401 / 0.387 |
| Weighted cosine, q = 0.1 / 0.25 / 0.5%, row-z | F | 0.424 / **0.435** / 0.429 |
| Binary kNN weights, q = 0.25%, row-z | F | 0.380 |
| Soft-power weights p = 4 / 8, row-z | F | 0.399 / 0.352 |
| Larger neighbourhoods: union q = 0.5%, mutual q = 1%, row-z | F | 0.429 / 0.422 |

Source: `results/phaseA/README.md`, `results/phaseA_graph/README.md`. All-pairs, exploratory probes: `reviews/round1/algorithms_compute.md`.

- **Binary overlap lost to weighted cosine** (Jaccard 0.04–0.11 lower; binary kNN weights 0.380 vs 0.435). *Likely:* binary sets discard link strength.
- **Larger neighbourhoods were worse after row-z.** *Likely:* weak partners dilute the shared signal.
- **Second order needs a whitened base:** on raw Pearson it was worse than the base (0.204 vs 0.238). *Speculation:* shared global factors dominate unwhitened neighbour lists.
- **All-pairs, exploratory:** dense second order 0.205 vs top-1% sparsified 0.310; z-average of first and second order 0.314 vs rank average 0.263. Rank averaging was never run on dev.

## 5. Graph family

Layers were built genome-wide on F with q = 0.25% unless stated. B scored 0.435 at seed 0.

**Table H. Graph layers (dev, seed 0).**

| Family | Variant | HT-AP |
|---|---|---|
| kNN symmetrisation (second-order cosine, row-z) | directed / union / mutual / average | 0.435 / 0.435 / 0.435 / 0.437 |
| TOM on kNN graph | linear weights, raw / row-z | 0.365 / 0.428 |
| TOM on kNN graph | binary weights, union q = 0.25% / mutual q = 0.5%, row-z | 0.383 / 0.369 |
| Dense soft threshold (β = 6) | cosine, raw / row-z | 0.419 / 0.420 |
| Dense soft threshold (β = 6) | TOM, raw / row-z | 0.395 / 0.422 |
| Diffusion profile cosine, row-z | heat t = 1 / heat t = 3 / PPR α = 0.5 | 0.442 / 0.441 / 0.441 |
| Random-walk powers, profile cosine, row-z | N¹ / N² / N³ | 0.432 / 0.427 / 0.347 |
| Raw diffusion kernel values, row-z | PPR / N² | 0.411 / 0.423 |
| Laplacian eigenmap, dimension d | 32 / 128 / 512 / 1024 | 0.321 / 0.429 / **0.460** / 0.451 |
| Eigenmap, d = 512, variants | row-z / q = 0.5% / k = 0 base / diffusion-map weighting λ⁴ | 0.458 / 0.455 / 0.451 / 0.373 |

Source: `results/phaseA_graph/README.md`, `results/phaseA_graph/screen.json`, `results/phaseA_graph/nested.json`, `results/phaseA_graph/compare.json`.

- **Nested check:** choosing the best of 67 graph layers on 4 folds and scoring the 5th gave 0.452, against 0.460 in sample (optimism 0.009). Every outer fold chose the d = 512 eigenmap or its row-z.
- **Over 5 seeds**, E scored 0.463 and H 0.460, against 0.448 for B. Pooled-OOF Δ vs B was −0.004 to +0.030 for E (every CI included 0) and +0.007 to +0.017 for H (CI above 0 in 3/5 seeds).
- **Symmetrisation did not matter; TOM and soft thresholds never beat cosine; long walks over-smoothed** (N³: 0.347).
- **No graph layer replaced B** under the pre-specified pooled-OOF rule. E had higher macro AP per cluster (seed 0: 0.394 vs 0.341; five-seed means 0.406 vs 0.351), so it entered the ensemble instead.
- **Cost:** d > 1024 needs a dense eigendecomposition (about 26 min). E took 94 s and H 59 s, against 23 s for B.

## 6. Ensembles

**Table I. Label-free z-average ensembles (dev HT-AP).**

| Ensemble | Members | Seed 0 | 5-seed mean |
|---|---|---|---|
| Greedy forward selection (Phase A) | B + (k = 40, τ = 0.01, not standardised) | 0.441 | 0.456 |
| Rule-based "base3" | F + row-z F + second-order q = 0.5% | 0.403 | 0.438 |
| Rule-based "fam3" | F + row-z F + B | 0.436 | 0.459 |
| Rule-based "top3" | B at q = 0.1 / 0.25 / 0.5% | 0.439 | 0.454 |
| zavg_BE | B, E | 0.456 | 0.466 |
| zavg_BF | B, F | 0.439 | 0.458 |
| **zavg_BEF (final)** | B, E, F | 0.456 | **0.470** |
| zavg_BEHF | B, E, H, F | 0.454 | 0.469 |
| *Reference: B alone* | – | 0.435 | 0.448 |

Source: `results/phaseA/ENSEMBLE.md`, `results/phaseA/ensemble.json`, `results/phaseA/COMPARE.md`, `results/phaseB_v2/README.md`, `results/phaseB_v2/nested.json`, `results/verify/README.md`, `results/report/tables.md` (T6).

- **Greedy selection was selection noise.** It added a layer weak alone (0.250). In nested CV it scored 0.428 held out, equal to the best single layer (0.428); members changed between outer folds; pooled-OOF Δ vs the best layer was −0.003 to +0.005 (all CIs included 0).
- **Pre-specified z-averages were robust:** nested selection among the four gave 0.468 vs 0.470 in sample.
- **Adding H (zavg_BEHF) tied** (0.469 vs 0.470) at extra cost; the pre-specified argmax chose zavg_BEF. *Speculation:* H and E both smooth the same kNN graph.
- **Why equal weights:** nothing to overfit, and learned weights did not beat them (Section 7).
- **The margin over B depends on E's settings:** its CI was above 0 in 5/5 seeds at d = 512, 4/5 at d = 1024 or q = 0.5%, and 2/5 at d = 256 (independent check); the point estimate stayed at 0.466–0.470.

## 7. Supervised stacking

### 7.1 Phase B v1 (seed 0; 12 layers chosen by name before any Phase A results)

**Table J. Phase B v1 stackers (dev, seed 0).**

| Model | HT-AP (fold mean ± SD) | Pooled-OOF HT-AP |
|---|---|---|
| GLS (reference) | 0.288 ± 0.098 | 0.256 |
| LR on Pearson + Spearman + GLS | 0.322 ± 0.106 | 0.296 |
| LGBM on Pearson + Spearman + GLS | 0.314 ± 0.114 | 0.264 |
| Label-free z-mean of the 12 layers | 0.357 ± 0.116 | 0.317 |
| LR, 12 layers | 0.355 ± 0.108 | 0.308 |
| LGBM, 12 layers | 0.383 ± 0.120 | 0.322 |
| LGBM, 12 layers + binned gene features | 0.385 ± 0.119 | 0.355 |
| LGBM, 12 layers + raw gene features | 0.401 ± 0.121 | 0.359 |
| LGBM, 12 layers + chromosome features | 0.384 ± 0.118 | 0.345 |
| LR, 12 layers + binned gene features | 0.351 ± 0.115 | 0.308 |

| Pooled-OOF comparison | Δ [CI] |
|---|---|
| LGBM (12 layers) − z-mean of the same 12 layers | +0.006 [−0.034, +0.032] |
| Adding binned gene features to LGBM | +0.033 [−0.017, +0.084] |
| Adding raw gene features to LGBM | +0.037 [−0.004, +0.076] |
| Adding chromosome features to LGBM | +0.023 [−0.002, +0.047] |
| LR on the three baselines − GLS | +0.040 [+0.012, +0.075] |

Source: `results/phaseB/README.md`, `results/phaseB/bootstrap_cluster_seed0.json`.

Over the three baselines, supervision gained a little over GLS (last row), far less than new layers did. Over 12 layers, LGBM matched the label-free z-mean; gene and chromosome features added nothing significant.

### 7.2 Phase B v2 (5 seeds; 7 frozen inputs: B, E, H, F, F with k = 0, GLS, second order without row-z)

**Table K. Phase B v2 stackers vs the z-average (dev, 5-seed means).**

| Model | HT-AP (fold mean) | Pooled-OOF HT-AP | Macro AP per cluster | AP without top-10 clusters |
|---|---|---|---|---|
| LR | 0.462 | 0.440 | 0.418 | 0.356 |
| LR + binned gene features | 0.449 | 0.427 | 0.415 | 0.350 |
| LGBM | 0.466 | 0.444 | 0.415 | 0.360 |
| LGBM + binned gene features | 0.464 | 0.445 | 0.417 | 0.355 |
| **zavg_BEF** | **0.470** | **0.464** | 0.395 | 0.360 |

Source: `results/phaseB_v2/README.md`, `results/phaseB_v2/compare.json`.

- **No stacker beat the z-averages:** no pooled-OOF CI against any of the four was above 0 in any seed; LR with gene features was below 0 in 11/20 seed × average cells. Fold models put scores on different scales, which lowers pooled-OOF AP.
- **Stackers had higher macro AP per cluster** (0.415–0.418 vs 0.395), the only hint that supervision might help small complexes.
- **Weights were not interpretable:** LR coefficients had mixed signs on collinear inputs; LGBM put 0.50–0.55 of its gain on E.
- A further stacker was not run, as the pre-specified rule required beating the best z-average first.

### 7.3 Leakage gap: random-pair vs cluster split (seed 0)

**Table L. Leakage gap by split (dev HT-AP, seed 0).**

| Model | Cluster split | Protein-disjoint split | Random pair split (leaky) | Gap (random − cluster) | Gap minus that of the label-free z-mean |
|---|---|---|---|---|---|
| GLS | 0.288 | 0.288 | 0.297 | +0.009 | −0.018 |
| z-mean of 12 layers (label-free) | 0.357 | 0.370 | 0.384 | +0.026 | 0.000 |
| LGBM, 12 layers | 0.383 | 0.405 | 0.413 | +0.030 | +0.004 |
| LGBM, 12 layers + binned gene features | 0.385 | 0.417 | 0.430 | +0.046 | +0.019 |
| LGBM, 12 layers + raw gene features | 0.401 | – | 0.445 | +0.044 | +0.018 |
| LR, 12 layers + binned gene features | 0.351 | 0.379 | 0.392 | +0.041 | +0.014 |

Source: `results/phaseB/README.md`.

Label-free scores gain 0.01–0.03 on a random split only because the evaluated pairs change, so supervised leakage is the excess in the last column. Gene features raised it from +0.004 to +0.014–0.019. The binned gene signature was unique for 24% of dev genes; leaf-size limits and cluster-disjoint folds, not binning, prevented memorisation.

### 7.4 Controls

**Table M. Controls for supervised models (dev, seed 0 unless stated).**

| Control | HT-AP | Reference |
|---|---|---|
| Gene statistics only: LGBM binned / LGBM raw / LR binned | 0.024 / 0.022 / 0.019 | HT prevalence 0.0091 |
| Chromosome features only (LGBM) | 0.009 | 0.0091 |
| Held-fold label shuffle, Phase B v1 LGBM | 0.042 | 0.041 |
| Held-fold label shuffle, Phase B v2 (4 models) | 0.042–0.043 | 0.041 |
| All labels shuffled, monotone LGBM (5 permutations × 5 folds) | mean 0.146 | 0.0101 |
| All labels shuffled, LGBM without monotone constraints | mean 0.018, median 0.009 | 0.0101 |
| All labels shuffled, LR | mean 0.061 (0.121 when the fit correlated positively with GLS, 0.014 otherwise) | 0.0101 |

Source: `results/phaseB/README.md`, `results/phaseB_v2/README.md`, `results/phaseB/control_shuffle_rep_train_*.json`.

**Monotone constraints defeat the all-label shuffle control.** With shuffled training labels, monotone LGBM still scored 0.146, and its scores tracked GLS (mean |Spearman ρ| = 0.78). The constraint itself encodes "more similar means more likely". This is not leakage: without it, median AP was at prevalence. Shuffle controls need unconstrained models.

**Why stacking failed** (*partly speculation*): the label-free layers carry the gain; gene features carry little transferable signal (about 2.5 × prevalence) and add leakage; inputs are collinear; and 190 dev clusters may be too few to learn weights.

## 8. Supervised metric learning (Phase C)

A capacity ladder, each rung adopted only if it beat the one below: L1, learned cell-line weights or spectral filter; L2, low-rank projections (d = 32, 64, 128; binary cross-entropy, or InfoNCE at d = 64); L3, the B pipeline on the learned space. Maps were fitted per training fold, with regularisation and step count chosen on an inner cluster-grouped split.

**Table N. Metric-learning ladder (dev, seed 0).**

| Rung | First-order cosine | Second order + row-z (L3) |
|---|---|---|
| Identity (no learning) | 0.365 | 0.435 |
| L1 cell-line weights | 0.366 | 0.434 |
| L1 spectral filter | 0.363 | 0.432 |
| L2 d = 32 | 0.365 | 0.434 |
| L2 d = 64 | 0.365 | 0.434 |
| L2 d = 128 | 0.364 | 0.432 |
| L2 d = 64, InfoNCE | 0.365 | 0.435 |
| Training labels shuffled (L1 weights / L2 d = 64) | 0.365 / 0.360 | – / 0.423 |

Source: `results/phaseC/README.md`, `results/phaseC/compare.json`.

- **No rung beat its starting point.** Pooled-OOF Δ was within ±0.003 for every first-order rung against the identity and every L3 rung against B (CIs including 0). On seeds 1–4, L1 weights equalled the identity to 3 decimals (5-seed L3 mean 0.448, same as B).
- **Why:** training loss fell from 0.57 to 0.09 (the ~650 training genes were memorised) while inner-validation AP on unseen clusters stayed at its identity value (±0.003), or collapsed with a larger learning rate (0.47 to 0.27). Early stopping kept the maps near the identity. Our interpretation: about 640 genes and 1,900 positives per fold are too few complexes to learn a transferable re-weighting of about 1,340 coordinates.
- **Positive control:** with planted structure in a toy, L2 raised held-out HT-AP from about 0.05 to 0.27.
- **Side finding:** outside the top-10 clusters, F already equalled B (+0.004 [−0.031, +0.036]; macro AP 0.371 vs 0.341). This motivated adding F to the final average.

## 9. Post-hoc reference: Pearson after removing principal components

At the manuscript reviewer's request, after the internal test had been evaluated, we scored Pearson correlation after removing the top k principal components (τ = ∞), with and without gene standardisation. This is the normalisation used by Boyle et al. and one layer of Hassan et al. It was not a pre-specified baseline and was not a candidate.

**Table O. Pearson after removing principal components (dev, 5-seed means; IFT single post-hoc look).**

| Variant | Dev HT-AP | Dev macro AP | Dev HT-AP without top-10 clusters | IFT HT-AP |
|---|---|---|---|---|
| k = 5, genes not standardised | 0.334 | 0.325 | 0.285 | – |
| k = 20, genes not standardised | 0.358 | 0.328 | 0.302 | – |
| k = 5, genes standardised | 0.317 | 0.309 | 0.263 | – |
| k = 20, genes standardised | 0.363 | 0.342 | 0.310 | 0.267 |
| zavg_BEF (final) | 0.470 | 0.395 | 0.360 | 0.345 |

Every variant was below zavg_BEF in every seed (pooled out-of-fold Δ −0.08 to −0.18, CI excluding zero in 5/5 seeds). On the IFT, the best variant was 0.078 below zavg_BEF (95% CI −0.151 to −0.028). Its pooled AP was above GLS on the development set (0.312) and on the IFT (0.242), but its CI against GLS excluded zero in only 2 of 5 seeds, and its macro AP was below GLS (dev 0.342 vs 0.349; IFT 0.320 vs 0.369).

Source: `results/revision/README.md`; IFT look "2-posthoc" in `results/ift_looks.jsonl`.

## 10. Process failures relevant to methods

- **Exposure before the split (D1).** Two design reviewers computed AP of label-free scores on all scorable pairs before the IFT existed (shrunk-covariance whitening about 0.30 vs GLS 0.255). No selection was made, but the whitening prior was formed on data including IFT pairs (logged as IFT look 0). Sources: `RESEARCH_TREE.md` (D1), `results/ift_looks.jsonl`.
- **Hash-seed-dependent split.** Louvain tie-breaking depended on Python's hash seed, so the IFT/dev split could change between processes. A test caught this; the canonical assignment (used by all results) is committed with a checksum. Sources: `RESEARCH_TREE.md`, `LESSONS.md`, `results/splits/protein_assignment.sha256`.
- **Unpassable IFT gate.** The original gate required the IFT CI to exclude 0. A dev-only power simulation gave a pass probability of 0.215 at a true +20% gain (between-method correlation 0.9) and 0.020 at the observed correlation 0.483; the IFT's Kish effective size is 7.7 clusters. Before any look, the CI requirement moved to dev pooled-OOF predictions; the IFT kept a ≥ +10% point-estimate gate plus non-inferiority (D2). Sources: `results/splits/power_ift.json`, `results/report/tables.md` (T1), `RESEARCH_TREE.md` (D2).
- **Smaller, disclosed issues:** second screening rounds were chosen after seeing the first; some runs overrode the clean-tree check because of another agent's untracked file (recorded); a superseded Phase C run was overwritten (both trials remain in `results/trials.jsonl`). Sources: the READMEs of `results/phaseA/`, `results/phaseA_graph/`, `results/phaseB_v2/`, `results/phaseC/`.
