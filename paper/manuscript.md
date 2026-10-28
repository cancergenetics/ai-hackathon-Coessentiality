---
title: "Shared co-dependency neighbourhoods improve prediction of protein co-complex membership from CRISPR screens: a benchmark run by AI agents"
---

**Short title:** Neighbourhood co-essentiality predicts co-complex pairs

**Authors:** [Author names and affiliations to be added by the corresponding author]

**Corresponding author:** [to be added]

# Abstract

This study tested whether AI agents can carry out a bioinformatics project: an AI agent acting as principal investigator, with AI sub-agents, did the design, analysis, verification and first draft, while humans set the task and supplied the data.

The task was to predict whether two proteins belong to the same complex using only gene co-essentiality across CRISPR knockout screens of cancer cell lines from the Cancer Dependency Map (DepMap). The conventional scores are Pearson or Spearman correlation of two genes' fitness profiles, and generalised least squares (GLS), which accounts for correlation between cell lines. Labels were the hu.MAP 3.0 training pairs (3,314 positive, 631,465 negative). Because the negatives cover nearly all other pairs among the same proteins, we evaluated on groups of complexes held out from training, so that no protein appeared on both sides of a split.

The final score uses no labels. It averages three similarities computed from standardised, filtered and whitened profiles: direct correlation, the overlap of the two genes' genome-wide co-dependency neighbourhoods, and their proximity in a spectral embedding of the neighbour graph. In five repeats of cross-validation grouped by complex, it reached a weighted area under the precision–recall curve of 0.470, compared with 0.312 for GLS, 0.282 for Spearman and 0.260 for Pearson. These estimates are optimistic because the method was selected on these data.

In one pre-specified evaluation on held-out complexes, it scored 0.345 against 0.242, 0.146 and 0.134. The difference from GLS was +0.103 (95% confidence interval +0.035 to +0.209). This test set is small and was partly exposed early on. The gain was largest for large complexes; on small complexes the evidence was mixed. Supervised models did not improve on the label-free score. The withheld hu.MAP test pairs have not yet been evaluated.

# Author summary

Genes whose loss harms the same cancer cell lines tend to work together, often as parts of one protein complex. Large CRISPR screens across cell lines let us measure this "co-essentiality". We asked how well it predicts which proteins form complexes, and whether we could improve on the standard correlation-based scores.

This project was itself an experiment. An AI agent acting as principal investigator, assisted by other AI agents, designed and ran the analysis, while humans defined the task and provided the data.

The best score did not come from training a classifier. It came from asking whether two genes share the same co-dependent partners across the genome, rather than only whether their profiles correlate. We combined this neighbourhood view with direct correlation and a graph embedding.

Compared with the best conventional method, the score raised the area under the precision–recall curve by about half in cross-validation (0.470 vs 0.312). On held-out complexes the gain was 42% (95% confidence interval 13% to 90%). Gains were concentrated in larger complexes in development data.

We describe how the agents built a benchmark that avoids information leaking between training and test data, what failed, and which errors the agents' own checks caught.

# Introduction

This work was run as a test of whether AI agents can carry out a bioinformatics project; humans set the task in a prompt (S1 Text), provided the data and later requested this manuscript (Methods: Agent setup).

Genes that encode subunits of the same protein complex often have correlated fitness effects across cancer cell lines: losing either subunit disrupts the same complex, so the same lines suffer {pan2018, boyle2018, kim2019}. The Cancer Dependency Map (DepMap) {tsherniak2017} provides gene effects estimated with Chronos {dempster2021chronos} from genome-wide CRISPR screens; the release used here covers 1,361 cancer models {depmap2026}. Co-essentiality networks built from such screens recover known complexes and have been used to assign functions to uncharacterised genes {pan2018, wainberg2021}.

Most studies score gene pairs with Pearson or Spearman correlation of their gene-effect profiles. Correlation is confounded, because cell lines are not independent samples: they share lineage, screen batch, library and global technical factors. Boyle et al. removed shared technical components estimated from olfactory receptor genes {boyle2018}. Wainberg et al. introduced generalised least squares (GLS), which whitens the profiles with the covariance between cell lines estimated from all genes {wainberg2021}. Gheorghe and Hart found that this whitening gave the largest single improvement among normalisation choices {gheorghe2022}. Hassan et al. combined similarity "layers" computed after removing different numbers of principal components {hassan2023}. Rahman et al. showed that a few large complexes can dominate pooled precision–recall benchmarks {rahman2021flex}.

We asked whether a co-essentiality score can clearly beat Pearson, Spearman and GLS at predicting co-complex membership. The objective was the area under the precision–recall curve (AUPRC; estimated as average precision, AP), which is more informative than the receiver operating characteristic (ROC) when positives are rare {saito2015}. Labels came from the hu.MAP 3.0 training pairs {fischer2025humap}, which derive from Complex Portal complexes {complexportal2022}. Benchmarks of pair predictors are easily inflated when test pairs share proteins with training pairs {park2012}. Here the negatives cover nearly every non-positive pair among the same proteins, which makes this risk especially acute. We therefore built a benchmark that holds out groups of complexes, and we confirmed the final method once on an internal test set chosen before the method screen (Fig 1).

# Results

## A benchmark that holds out complexes

The training labels contain 3,314 positive and 631,465 negative pairs over 1,151 UniProt accessions (prevalence 0.52%). Of these, 1,114 map to genes in DepMap's CRISPRGeneEffect matrix, which has 1,361 models and 18,434 genes. Two features of the labels shaped the design.

First, a model can learn from a protein's identity how many positive partners it has, and carry this over between random pair splits. hu.MAP's own test split is made at the level of complexes {fischer2025humap}. We therefore never let a protein appear on both sides of a split.

Second, positive pairs are clustered. The positive graph has 241 connected components; the largest (144 proteins) contains 27% of the mapped positive pairs. Because hu.MAP complexes have at most 30 subunits, a component this large must merge overlapping complexes. We split components larger than 40 proteins into Louvain communities {blondel2008}, which divided the giant component into seven and gave 247 clusters; 121 positive edges crossed communities.

Before the method screen we set aside 26.5% of mapped positive pairs, chosen by cluster, as an internal final test (IFT): 847 positive and 40,786 negative pairs in 57 clusters (Fig 1B). The remaining development set (2,299 positive and 320,229 negative pairs in 190 clusters) was used for five-fold cross-validation grouped by cluster, repeated over five fixed seeds (Table 1). Pairs spanning development and test proteins (45 positive, 232,391 negative) were not used.

**Table 1. Data and splits.** Clusters are connected components of the positive-pair graph, with the largest split into Louvain communities.

| | Proteins | Clusters | Positive pairs | Negative pairs |
|---|---|---|---|---|
| hu.MAP 3.0 training labels | 1,151 | – | 3,314 | 631,465 |
| Mapped to DepMap (scorable) | 1,114 | 247 | 3,191 | 593,406 |
| Development set (5-fold cross-validation) | 819 | 190 | 2,299 | 320,229 |
| Internal final test (IFT) | 295 | 57 | 847 | 40,786 |
| Development–test pairs (unused) | – | – | 45 | 232,391 |

Grouping by cluster changes the mix of pairs in each test fold. A between-cluster pair is kept only when both clusters fall in the same fold, so test folds over-represent within-cluster pairs: the positives, and the "hard" negatives from the same complex neighbourhood. We corrected this with Horvitz–Thompson weights {horvitz1952}: between-cluster pairs receive weight K (the number of folds) and within-cluster pairs weight 1. The weighted average precision (HT-AP) then estimates performance at the natural prevalence of 0.53%. All AUPRC values below are HT-AP unless stated otherwise. Because pooled AP can be dominated by a few large complexes {rahman2021flex, mcdermott2024}, we also report the mean AP over clusters (macro AP), AP excluding the largest clusters, and AP by essentiality of the pair.

## Conventional baselines

Each baseline was given its best variant, selected on development data by the same rule as our methods but from only 2–8 variants each (S2 Text, Table A). For all three, signed scores beat absolute values, so negative correlations added no useful signal on balance. GLS beat both correlations, and the olfactory-receptor correction {boyle2018, wainberg2021} lowered GLS performance on this release. Mean development HT-AP over five seeds was 0.260 for Pearson, 0.282 for Spearman and 0.312 for GLS (Fig 2A).

A further reference is Pearson correlation after removing the top principal components, as used by Boyle et al. and as one of Hassan et al.'s layers {boyle2018, hassan2023}. We added it after the internal test had been evaluated, at a reviewer's request, so it was not a pre-specified baseline. Its best variant (20 components removed, genes standardised) scored 0.363 over five seeds, below zavg_BEF in every seed (CI excluding zero; S2 Text). It was above GLS in pooled AP, but its CI against GLS excluded zero in only 2 of 5 seeds and its macro AP was lower (0.342 vs 0.349).

## Standardising genes before whitening

Our method keeps the logic of GLS but changes how the covariance is estimated, and then adds two neighbourhood views (Fig 1A). GLS estimates the cell-line covariance from raw gene effects, so strongly essential genes with large variance dominate it.

We first standardised each gene to unit variance, then removed the top 20 eigencomponents and whitened the rest with a ridge term. The resulting first-order similarity (F) reached 0.395 over five seeds (Fig 2A). The steps in Fig 3 (seed 0) show where this gain came from: gene standardisation added +0.061 and the ridge term +0.021. Removing principal components added little to F (+0.004 on seed 0; 0.395 with or without over five seeds). It did help the neighbourhood scores below (second-order score 0.401 with 20 components removed vs 0.389 without, seed 0). On seed 0, F changed by less than 0.01 across 0–20 removed components and ridge values 0.01–0.3.

## Shared neighbourhoods carry most of the gain

The largest single gain came from a second-order score (B). For each gene we listed its top 0.25% genome-wide partners by first-order similarity, about 44 genes. Two genes then scored highly if their partner lists overlapped, measured as the cosine of the weighted neighbour vectors. The score was then standardised against each gene's genome-wide distribution (row-z). B reached 0.448.

The idea follows topological overlap and shared-nearest-neighbour similarity from network analysis {ravasz2002, zhang2005wgcna, jarvis1973}. Subunits of a complex are expected to share co-dependent partners: other subunits, assembly factors and pathway members.

The gain was not uniform (Fig 4). On seed 0, B beat GLS in 73 of 177 development clusters and lost in 80, with a mean difference of +0.001. Its pooled advantage came from the ten largest clusters. Excluding those clusters, B scored 0.332 against 0.296 for GLS (five-seed means), and its macro AP equalled that of GLS (0.351 vs 0.349). Small complexes probably have too few partners to share for neighbourhood overlap to help.

B also had a lower area under the ROC curve (AUROC) than the baselines (development 0.755 vs 0.765 for GLS; IFT 0.674 vs 0.726). It ranks the highest-scoring pairs better but the full list worse.

## Combining three views helps smaller complexes in development data

A 512-dimensional Laplacian eigenmap {belkin2003} of the genome-wide neighbour graph gave a smoother graph-based similarity (E), which reached 0.463 with a macro AP of 0.406. The equal-weight average of the z-scores of B, E and F (zavg_BEF) scored 0.470, with a macro AP of 0.395 and an AP of 0.360 without the ten largest clusters (Fig 2A, 2C).

On seed 0, zavg_BEF beat GLS in 93 of 177 development clusters and lost in 57, with a mean difference of +0.042 (Fig 4A). Against GLS, the gain in pooled out-of-fold HT-AP was +0.175 to +0.193 across seeds, with every 95% cluster-bootstrap confidence interval (CI) above zero (Fig 2B). Against B alone, zavg_BEF gained +0.014 to +0.021, with the CI above zero in all five seeds; in seed 2 the lower bound was only +0.0005. On pairs outside the largest clusters the gain over B was +0.024 to +0.031.

We used equal weights because nothing needs to be fitted and learned weights did not do better (next section). The z-scores are calibrated on random genome-wide gene pairs, so no label information enters the combination (Methods).

These development estimates are optimistic. About 300 label-free variants were screened on seed 0, and the members of the final average were chosen after seeing development results. Selecting among the candidate averages within nested cross-validation gave 0.468 against 0.470. This check covers only the final choice, not the whole search.

## Supervision did not help

We tested whether labels could improve on the label-free scores, with 1,808–1,981 training positives per fold (S2 Text). Logistic regression and gradient-boosted trees {ke2017lightgbm} over seven similarity views scored 0.462 and 0.466, below zavg_BEF; their macro AP was higher (0.415–0.418 vs 0.395). Adding each gene's own essentiality statistics did not help. Learned cell-line weights, low-rank projections trained with contrastive or binary losses, and second-order scores on the learned space all stayed within ±0.003 of their starting point. These models fitted the training genes but did not improve on unseen complexes.

A model using only gene-level statistics reached 0.024, about 2.6 times the prevalence. This probably reflects a transferable tendency of essential genes to form complexes. The leakage check is the gap between split schemes: supervised models gained at most +0.02 more than the label-free scores when moving from cluster-grouped to random pair splits (S2 Text, Table L).

## Confirmation on held-out complexes

We evaluated zavg_BEF (primary) and B (secondary) once on the IFT, in an order fixed beforehand (Fig 5, Table 2). zavg_BEF scored 0.345 (95% CI 0.197 to 0.590), against 0.242 for GLS, 0.146 for Spearman and 0.134 for Pearson. The differences were +0.103 against GLS (95% CI +0.035 to +0.209; +42% relative, CI +13% to +90%), +0.199 against Spearman (+0.099 to +0.321) and +0.211 against Pearson (+0.103 to +0.338). zavg_BEF therefore met the pre-specified criterion against all three baselines.

**Table 2. Performance (HT-weighted AP unless stated).** Development (Dev): mean of five cross-validation repeats. IFT: single pre-specified evaluation; AUROC is HT-weighted. Macro AP: mean AP over clusters. Dev AP without top-10 clusters is computed on pooled out-of-fold predictions. Runtime: wall-clock time from the raw gene-effect file to scores for 634,779 pairs on a 13-inch MacBook Pro (Apple M2, 8 cores, 24 GB memory), measured in one session; *B was timed in an earlier session. Run-to-run variation was about ±20%, mostly from the eigenmap, which is why E alone (125 s) took longer than zavg_BEF (118 s); an earlier zavg_BEF run took 137 s.

| Method | Dev AP | Dev macro AP | Dev AP without top-10 clusters | IFT AP | IFT macro AP | IFT AUROC | Runtime (s) |
|---|---|---|---|---|---|---|---|
| Pearson | 0.260 | 0.258 | 0.211 | 0.134 | 0.234 | 0.734 | 11 |
| Spearman | 0.282 | 0.258 | 0.216 | 0.146 | 0.235 | 0.739 | 11 |
| GLS | 0.312 | 0.349 | 0.296 | 0.242 | 0.369 | 0.726 | 19 |
| F (first order) | 0.395 | 0.388 | 0.339 | – | – | – | 15 |
| B (second order) | 0.448 | 0.351 | 0.332 | 0.317 | 0.296 | 0.674 | 28* |
| E (eigenmap) | 0.463 | 0.406 | 0.352 | – | – | – | 125 |
| **zavg_BEF (final)** | **0.470** | 0.395 | **0.360** | **0.345** | 0.365 | **0.770** | 118 |

At 25% recall, zavg_BEF had a precision of 0.77, against 0.35 for GLS. It scored higher in each essentiality stratum: 0.276 vs 0.214 for pairs of two non-essential genes, 0.388 vs 0.349 for mixed pairs, and 0.394 vs 0.263 for pairs of two essential genes. Its AP without the largest connected component was 0.470, against 0.350 for GLS, but its macro AP (0.365) equalled that of GLS (0.369). So on the IFT, the evidence for smaller complexes is mixed. When the 9 positive and 2,352 negative pairs that involve an unmapped protein were added at the minimum score, AP was 0.332 against 0.233 for GLS.

B alone gained +0.075 over GLS but failed the non-inferiority criterion on macro AP (0.296 vs 0.369). This is consistent with the development pattern, but it depends on the macro-AP definition (Methods): with within-cluster pairs only, B would pass. The post-hoc reference, Pearson after principal-component removal, scored 0.267 on the IFT (difference from zavg_BEF −0.078, 95% CI −0.151 to −0.028), with a macro AP of 0.320.

The IFT values are lower than development values for every method. This probably reflects the 57 held-out clusters, which include 4 of the 10 largest clusters and 2 of the 7 giant-component communities. The IFT is small (Kish effective number of clusters 7.7 {kish1965}). It was also not fully untouched: before it was split off, two design reviewers computed AP of label-free scores on all pairs, including future IFT pairs. No choice was made from those numbers, but they informed the general expectation that whitening helps.

An agent that had not produced the result recomputed all IFT metrics with its own code and obtained values identical to within 10⁻¹⁶. Its independent re-implementation of the scores gave the same AP to four decimals.

## Controls and artefacts

Negative controls behaved as expected. Shuffling gene profiles reduced AP to prevalence (0.009–0.011), as did shuffling labels in held-out folds of supervised models and using chromosome information alone. For zavg_BEF, removing same-chromosome pairs raised pooled development AP slightly (0.445 to 0.455, seed 0), as it did for B and GLS. One control behaved unexpectedly: a gradient-boosted model constrained to increase with similarity still scored 0.146 (mean of repeats) when trained on fully shuffled labels, because the constraint itself encodes "more similar is more likely".

Olfactory receptor genes are not essential, so pairs of them should score as background. Under zavg_BEF, 8.4% of these pairs exceeded the null 99th percentile (expected 1%; B 7.2%, GLS 6.6%). The excess was mostly in pairs on the same chromosome (22.6%, against 4.0% on different chromosomes; GLS 16.1% and 3.6%). This points to a known proximity artefact of CRISPR screens {kim2019, wainberg2021, lazar2024}, which the neighbourhood scores slightly amplify.

# Discussion

Scoring shared co-essentiality neighbourhoods instead of direct co-essentiality improved co-complex prediction, without labels, more than any supervised model we tried. The final method combines published ideas:

- whitening of the cell-line covariance {wainberg2021, gheorghe2022};
- removal of dominant principal components {boyle2018, hassan2023}, which here mattered for the neighbourhood views but not for F, and which on its own (as Pearson after component removal) scored well below zavg_BEF;
- per-gene balancing of similarity distributions, following mutual rank and balanced correlation {obayashi2009, bacon2025};
- topological overlap and shared-nearest-neighbour similarity {ravasz2002, zhang2005wgcna, jarvis1973};
- spectral embedding {belkin2003};
- combining several normalised similarity layers {hassan2023}.

Our contribution is to combine these, tune them on a benchmark that holds out complexes, and test them once on held-out complexes. The largest steps were standardising genes before estimating the covariance (+0.061 on seed 0) and the neighbourhood score with its row standardisation (+0.070); averaging three views calibrated on a fixed genome-wide null added +0.021 and helped smaller complexes.

The improvement has limits. It is largest for big, densely connected complexes. For smaller complexes the gain was clear in development data, but on the IFT macro AP was equal to GLS. B ranks fewer pairs correctly overall (lower AUROC), which matters for users who apply low thresholds.

Labels did not help with about 1,800–2,000 training positives per fold and the model classes we tried; more labels or other data types might change this. All hyper-parameters (20 components, ridge 0.1, 0.25% neighbours, 512 dimensions) were selected on development data. The size of the margin of zavg_BEF over B was stable across eigenmap settings, but whether it was significant depended on them (S2 Text).

Two groups of proteins cannot be scored and receive the minimum score; together they account for 3.7% of positive training pairs:

- proteins without a DepMap profile, such as immunoglobulin segments, mitochondrially encoded genes and haemoglobins;
- proteins that map to several DepMap genes (two histones).

Same-chromosome artefacts are slightly amplified. The method was developed on one DepMap release and on hu.MAP 3.0 complexes of at most 30 subunits.

On the process: the agents built a careful benchmark, but made errors; their own checks caught several of them.

- **Early exposure of the test data.** Two design reviewers computed AP on all pairs before the internal test set existed.
- **An irreproducible split.** The cluster assignment depended on Python's hash seed, through tie-breaking in Louvain. A test written by another agent caught this. The canonical assignment, which all results used, was fixed and committed.
- **A significance gate that could rarely be passed.** A power simulation showed that the gate first specified for the internal test would pass with probability at most 0.22, even at a true 20% gain. The gate was amended before the test was opened.
- **Wasted effort on supervised models.** Full-scale supervised experiments continued after early signs that they gave no gain.
- **Lapses in record-keeping.** The research record went stale at times, and some confidence intervals that were needed for the report were never computed.

Re-implementation of every headline result by a separate agent was the most useful safeguard.

# Materials and methods

## Data

We used DepMap CRISPRGeneEffect (Chronos {dempster2021chronos}), release 26Q3 as stated by the human investigators (the files carry no version stamp), together with Model, ModelCondition, CRISPRConfounders and ScreenGeneEffect {depmap2026}. Gene annotations came from the HUGO Gene Nomenclature Committee (HGNC) complete set {hgnc2023}; its download date was not recorded. Labels were the train positive and negative pairs provided by the human investigators as hu.MAP 3.0 {fischer2025humap, humapweb2026}; we did not independently confirm the version or the licence of these files. The hu.MAP test pairs were withheld by design and never accessed. File checksums are in the code repository.

UniProt accessions were mapped to approved HGNC genes, then to Entrez IDs, then to DepMap columns, matching on Entrez ID rather than symbol. Of 1,151 accessions, 1,114 mapped. Two (histones H2B and H4) mapped to several DepMap genes and were dropped as ambiguous, and 35 had no DepMap gene. The 27,046 pairs among labelled proteins that appear in neither label file were never used.

## Evaluation design

**Clusters.** We took the connected components of the graph of positive pairs among mapped proteins. Components with more than 40 proteins were split by Louvain community detection {blondel2008}, a fast and widely used modularity method (networkx {hagberg2008networkx}, resolution 1, seed 0, integer node labels, PYTHONHASHSEED=0); only the 144-protein component qualified. Each cluster is the unit for splitting and resampling.

**Internal final test.** Clusters were assigned to the IFT by a seeded random draw, stratified by cluster size, until about 25% of positives were included. A 25% fraction keeps enough development data for five-fold cross-validation. A draw was rejected if its effective number of clusters was below 20% of that of all clusters (25.2). The effective number of clusters is $(\sum_c w_c)^2/\sum_c w_c^2$, where $w_c$ is the number of positive pairs in cluster $c$ {kish1965}. IFT pairs have both proteins in IFT clusters, and development pairs have both in development clusters; pairs spanning the two were excluded.

**Cross-validation.** Development clusters were randomly assigned to five folds, balancing positives; five folds leave about 38 clusters per test fold. Training pairs have both proteins in training folds, and test pairs both in the held-out fold. This was repeated with seeds 0–4. The repeats re-split the same 190 clusters, so they measure split variance rather than independent replication. Selection used seed 0.

**Metric.** We used average precision with Horvitz–Thompson weights. In a K-fold split, a within-cluster pair always enters a test fold together with its cluster, but a between-cluster pair enters only when both clusters share a fold (probability about 1/K). Weighting between-cluster pairs by K (on the IFT, by 1/f, where f = 0.265 is the IFT fraction of proteins) restores the population mix. In a toy simulation, fold-averaged HT-AP was within 3% of the AP over the full set of pairs.

Macro AP is the unweighted mean, over clusters with at least one positive, of the HT-AP of all pairs touching the cluster. This definition was fixed in code before the IFT was opened but was not written in the design document.

Essentiality strata: a gene was called common-essential if its gene effect was below −0.5 in at least half of the cell lines.

**Uncertainty.** We used a paired cluster bootstrap {field2007cluster} with 1,000 resamples. Clusters were drawn with replacement; a within-cluster pair was weighted by its cluster's multiplicity, and a between-cluster pair by the product of the two multiplicities. In 400 toy datasets with known truth and AP values similar to ours, nominal 95% intervals for the difference covered the truth in 94.8% of cases (92.8% in a low-AP scenario); misses were mostly on one side.

**Pre-specified criteria.** The criteria and candidates were fixed in a version-controlled design document before the IFT was opened; they were not registered publicly. A candidate "clearly beats" a baseline if all of the following hold:

1. It wins in at least 4 of 5 development folds, and has the higher mean in each seed repeat.
2. The 95% cluster-bootstrap CI of the AP difference, computed on the pooled out-of-fold predictions for the development set, excludes zero in each seed repeat.
3. Its IFT point estimate is at least 10% higher in relative terms.
4. On the IFT, its macro AP and its AP without the largest component are each at least 95% of the baseline's.

These must hold against all three baselines. At most two candidates were tested, in a fixed sequence.

The design document originally required the IFT confidence interval itself to exclude zero. A power simulation on development data gave a pass probability of at most 0.22 at a true 20% gain, because the IFT has few effective clusters. We therefore moved the interval requirement to the development data before opening the IFT.

**Candidate selection.** Before the final combination round, we fixed the candidates as B and the best of eight pre-listed combinations, judged by mean development HT-AP over five seeds. B was the single-score reference because E, although higher on average (0.463 vs 0.448), had not beaten B with a CI excluding zero, which our rule required for replacing it (S2 Text). The combinations were four z-averages and four supervised stackers. The combination would become primary if its gain over B had a CI above zero in at least 3 of 5 seeds and its macro AP was not lower. zavg_BEF met both conditions. The members B, E, H (a diffusion score) and F had been chosen earlier, after seeing development results.

## Baselines

Pearson and Spearman correlations were computed on the cell lines observed for both genes. GLS followed Wainberg et al. {wainberg2021}: whitening by the Cholesky factor of the inverse cell-line covariance, followed by a regression t-statistic for each pair. We implemented it in matrix form and tested it against a line-by-line port of the original code.

We documented three deviations from the original code:

- **Rank deficiency.** In this release the covariance has rank 1,357 of 1,361, so four linearly dependent models were dropped by pivoted QR.
- **Missing values.** The original code drops any cell line with a missing value, which here would remove 1,355 of 1,361 lines. We instead estimated the covariance from the 17,082 genes without missing values and handled missing values per pair, on the samples both genes share.
- **Variant selection.** Variants (olfactory-receptor correction, imputation, signed or absolute scores) were selected on development data.

## Final method (zavg_BEF)

**Preprocessing.** Let $x_g \in \mathbb{R}^n$ be the gene-effect profile of gene $g$ over $n = 1{,}361$ cell lines. Each gene is centred on its observed mean, its missing values are set to that mean, and it is divided by its standard deviation.

**Filter.** Let BG be the background set of 17,525 genes with less than 10% missing values, and $G = |\mathrm{BG}|$. Let $Q \in \mathbb{R}^{n\times(n-1)}$ be an orthonormal basis of the complement of the all-ones vector, so that $Q^\top x_g$ removes the intercept. We estimate the second-moment matrix
$$\Sigma = \frac{1}{G}\sum_{g\in \mathrm{BG}} Q^\top x_g x_g^\top Q = \sum_i \lambda_i u_i u_i^\top,$$
with eigenvalues in descending order. The filter is
$$W=\sum_{i>k} (\lambda_i+\tau\bar\lambda)^{-1/2}\, u_i u_i^\top,\qquad k=20,\ \tau=0.1,$$
where $\bar\lambda$ is the mean eigenvalue. With $k=0$ and $\tau=0$ this is a GLS-like whitening. It differs from GLS in the gene standardisation, the imputation and the use of cosine instead of a t-statistic, and it scored 0.279 against GLS's 0.288 (Fig 3). With $\tau\to\infty$ it is Pearson correlation after removing principal components.

**First-order similarity (F).** $F_{ab}$ is the cosine of the filtered profiles $f_a = W Q^\top x_a$ and $f_b$. This equals their Pearson correlation, because the intercept has been removed.

**Second-order similarity (B).**

1. Standardise F within each gene's row over background genes, $z_{aj}=(F_{aj}-\mu_a)/\sigma_a$, and symmetrise it: $S_{aj}=(z_{aj}+z_{ja})/2$.
2. Each gene keeps its top $K=\mathrm{round}(0.0025\,G)=44$ background neighbours by $S$, with weights $\max(S_{aj},0)$.
3. The second-order score of $a$ and $b$ is the cosine between their neighbour vectors, after removing $b$ from $a$'s vector and $a$ from $b$'s. It therefore measures shared third-party partners only.
4. This score is row-standardised over the genome in the same way, which gives B.

**Graph embedding (E).**

1. Form the union of the neighbour graphs from step 2, with weights $\max(S,0)$.
2. Compute the symmetric normalised adjacency $N=D^{-1/2}AD^{-1/2}$.
3. Take its 512 leading non-trivial eigenvectors (ARPACK {lehoucq1998arpack} via SciPy {virtanen2020scipy}, with a seeded start vector).
4. E is the cosine between rows of this embedding. Genes outside the background set are embedded by Nyström extension {williams2001nystrom}.

**Combination.** Each of B, E and F is converted to a z-score using the mean and standard deviation of its values on 200,000 random background gene pairs (seed 0). zavg_BEF is the mean of the three z-scores.

No step uses labels, and a pair's score does not depend on which other pairs are scored. Proteins that cannot be mapped receive the minimum score.

## Search and logging

All variants were screened on development folds with seed 0. The screen covered 148 first-order, balancing and second-order layers, 69 graph layers, and supervised stacking and metric-learning models (S2 Text). Every evaluation wrote a start and an end record (845 started, 844 completed, no errors; one run was aborted).

## Agent setup

The project ran in the Claude Code desktop application. A principal-investigator agent (Claude Opus 5.5) received the task prompt (S1 Text) and worked autonomously.

Human input consisted of:

- the prompt and the data files;
- one "continue" message during the research phase;
- after the final report, a second prompt asking for this manuscript, the code repository and the supplementary files.

The principal investigator wrote the design document, kept a research tree of decisions and status and a list of process lessons, and made and recorded all decisions. It delegated work to sub-agents, each given a written brief that named its goal, its files and its rules.

The research phase used 25 sub-agents (18 Claude Opus 5.5, 7 Claude Sonnet 5.5). Their roles:

| Role | Sub-agents |
|---|---|
| Literature review | 1 |
| Data and evaluation infrastructure | 2 |
| Design review, round 1 (external principal investigator; statistics; algorithms and compute; data and biology; prior work; failure modes of AI agents) | 6 |
| Design review, round 2 | 2 |
| Method development | 6 |
| Engineering | 3 |
| Independent verification | 3 |
| Retrospectives at two milestones | 2 |

The manuscript phase used eight further sub-agents (33 in total): code repository, figures, two for references, the first draft of S2 Text, two independent reviews of this manuscript, and additional analyses requested by the first review. Fresh-agent retrospectives were run at two milestones, although the prompt asked for one at each milestone. S3 Text lists every sub-agent with its model, brief and commits. These counts come from the session record; they are not stored in the repository.

The principal investigator relied on several safeguards:

- **IFT isolation.** Only one script could import the IFT loader, which a test enforced by parsing the code. Every access to the IFT was logged: an initial note disclosing the reviewers' early exposure; the single evaluation; an independent recomputation; a re-plot for Fig 5; and the post-hoc reference added at review.
- **Fixed seeds.** Seeds were fixed in advance (0–4).
- **Clean code state.** Results were refused if the source code had uncommitted changes, unless this was overridden and recorded. Some development runs and the IFT run carry such a flag, for unrelated files only.
- **Independent verification.** Each headline result of the research phase was re-implemented from raw files by an agent that had not produced it, importing nothing from the producer's code except the split. The additional analyses made during manuscript revision (post-hoc reference, controls on zavg_BEF, runtimes, bootstrap coverage) were not independently re-implemented.

The research phase took 5.6 hours from the first commit to the final report, plus environment setup. Preparing the manuscript took several further hours. This manuscript was drafted by the principal-investigator agent and must be checked by the human authors.

## Software

We used Python 3.12 with NumPy {harris2020numpy}, SciPy {virtanen2020scipy}, pandas {mckinney2010pandas}, scikit-learn {pedregosa2011sklearn}, networkx {hagberg2008networkx}, LightGBM {ke2017lightgbm} and PyTorch {paszke2019pytorch}. Versions are pinned in the repository. About 200 unit tests check the code on toy data with known answers.

# Data availability

DepMap data are available from the DepMap portal (https://depmap.org/portal/data_page; CC BY 4.0), HGNC data from https://www.genenames.org, and hu.MAP 3.0 from https://humap3.proteincomplexes.org. The code repository holds:

- the top 200,000 gene pairs genome-wide;
- the assignment of proteins to clusters and splits.

The repository is https://github.com/cancergenetics/ai-hackathon-Coessentiality. It contains no DepMap, hu.MAP or HGNC files; scores for any pair list can be regenerated from the downloaded files with the provided scripts.

# Code availability

The code is available at https://github.com/cancergenetics/ai-hackathon-Coessentiality.

# Funding

[To be added by the authors.]

# Competing interests

[To be added by the authors.]

# Acknowledgments

[To be added.] The analysis and first draft were produced by AI agents (Claude, Anthropic); the AI agents are not authors.

# References

{REFERENCES}

# Supporting information

**S1 Text. Original task prompt.** The verbatim prompt given to the principal-investigator agent.

**S2 Text. Methods that did not work.** Baseline variants, filter variants, balancing, second-order and graph variants, ensembles, supervised stacking, gene-level features and supervised metric learning, with development numbers.

**S3 Text. Agent log.** Every sub-agent with its model, brief and commits.
