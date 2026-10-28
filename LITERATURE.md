# LITERATURE.md: co-essentiality for predicting co-complex membership

Compiled 2026-10-27 by the literature agent, from web searches and full-text reads of open-access versions (bioRxiv, Europe PMC, publisher PDFs).

Conventions:
- **[verified]** means I read the statement in the paper or code text.
- **[abstract-only]** means only the abstract or a secondary summary was available.
- **unverified** marks anything I could not check.
- Numbers are as reported by the cited source.
- Per project rules, I did not look for hu.MAP test-pair files. Only the hu.MAP 3.0 paper's Methods text was read.

---

## 1. Building co-essentiality networks

### 1.1 Correlation-based networks on DepMap

**Pan J, Meyers RM, …, Tsherniak A, Kadoch C (2018).** "Interrogation of mammalian protein complex structure, function, and membership using genome-scale fitness screens." *Cell Systems* 6:555–568. doi:10.1016/j.cels.2018.04.011 [abstract-only]
- **What they did:** built co-essentiality similarity networks from DepMap RNAi and CRISPR screens. The Broad press release says these were 342 CRISPR and 501 RNAi lines. They then combined the networks with 7 PPI datasets.
- **Findings:** the networks recapitulate the structure, modules and subunit membership of known complexes. They also found new SWI/SNF (BAF) modules and new members.
- **Implication:** this is the founding evidence that co-essentiality encodes co-complex membership, including sub-complex modularity. Quantitative numbers were not retrieved (unverified).

**Boyle EA, Pritchard JK, Greenleaf WJ (2018).** "High-resolution mapping of cancer cell networks using co-functional interactions." *Mol Syst Biol* 14:e8594. doi:10.15252/msb.20188594 [verified]
- **Data:** 436 Avana screens, Achilles 18Q2 (CERES), 17,634 genes.
- **Olfactory-receptor (OR) correction:**
  - Take 250 OR genes, removing those that share sgRNAs.
  - Run PCA on the 250 × 436 OR matrix.
  - Keep the top 5 PCs, chosen because they beat 100 within-line permutations. PC1 explains more than half the variance.
  - Project every gene profile onto those PCs and subtract the reconstruction.
  - Score with Pearson. The null is a normal distribution fitted to OR–OR correlations, with a 10% FDR cut.
- **Results:**
  - 1,528,726 co-functional pairs after correction, versus 30,761 uncorrected.
  - 53.9% of CORUM within-complex pairs were called.
  - The median correlation between cell-line profiles dropped from about 0.85 to about 0.
- **Implication:** most raw DepMap covariance between cell lines is technical or global. Removing a few global components helps Pearson a lot. OR genes give a label-free nuisance subspace.

**Kim E, Dede M, Lenoir WF, Wang G, Srinivasan S, Colic M, Hart T (2019).** "A network of human functional gene interactions from knockout fitness screens in cancer cells." *Life Sci Alliance* 2:e201800278. doi:10.26508/lsa.201800278 [verified]
- **Data:** BAGEL2 Bayes factors on Avana screens. They kept 276 of 338 lines by a quality filter (F-measure > 0.85 at BF = 5, fewer than 2,000 essentials), then quantile-normalized within each line.
- **Network:** Pearson correlation, positive edges only at Bonferroni p < 0.05. Genes essential in fewer than 3 lines were excluded.
- **Exclusions:** they removed same-chromosome pairs within 20 Mb, because of CRISPR copy-number and proximity artefacts. They flagged 270 genes in 30 clusters that tracked their own copy number.
- **Bias:** they note that ETC and mitochondrial-ribosome genes form a dominant cluster.
- **Implication:**
  - Screen-quality filtering and quantile normalization are cheap options.
  - Pairs that are close on the genome are confounded.
  - Restricting to "variable" genes is a common step, but it would drop never-essential complex members.

**Rauscher B, …, Boutros M (2018).** "Toward an integrated map of genetic interactions in cancer cells." *Mol Syst Biol* 14:e7656. doi:10.15252/msb.20177656 [verified]
- **What they did:** MINGLE integrates 85 screens from different labs and libraries.
- **Batch effects:** they found "considerable batch effects driven primarily by the sgRNA library". The effects were gene-specific; for example, CDK7 looked non-essential in one library. They corrected with quantile normalization plus a robust linear model of per-gene library offsets.
- **Network:** Spearman correlation of interaction profiles. Recovery of CORUM complexes was poor until low-signal interactions were filtered out.
- **Implication:** library effects are gene-specific. Mixing Avana, KY and Humagne screens without correction can create false correlations.

**Wainberg M, …, Kundaje A, Bassik MC (2021).** "A genome-wide atlas of co-essential modules assigns function to uncharacterized genes." *Nat Genet* 53:638–649. doi:10.1038/s41588-021-00840-z. Preprint doi:10.1101/827071. Code: github.com/kundajelab/coessentiality (MIT licence). [verified]
- **Data:** Achilles 18Q3 Avana CERES, 485 lines, 17,634 genes.
- **Algorithm, as given in the paper and `load_screens.py` / `gene_pairs.py`:**
  1. Bias correction: fit `sklearn PCA(n_components=4)` on the OR-gene matrix. OR genes are the samples and cell lines the features. Then compute `screens -= inverse_transform(transform(screens))`. Because of how sklearn centres, this gives (X − μ_OR)(I − VᵀV), where μ_OR is the per-line mean over OR genes. The data end up centred on OR means and 4 directions in cell-line space are removed.
  2. Drop the last 4 cell lines (`screens.iloc[:, :-4]`), leaving 481. Removing 4 PCs makes the line × line covariance rank-deficient by 4, so dropping 4 lines keeps it invertible.
  3. Drop genes with any NaN (`screens.dropna(axis=1)`). The code comments say this is needed for newer releases. **With 26Q3, 1,352 genes have at least one NaN, so a faithful port loses them; we need a documented imputation or pairwise-complete variant.**
  4. Estimate the covariance between lines from all genes, then whiten: `cholsigmainv = np.linalg.cholesky(np.linalg.inv(np.cov(screens.T)))`, `warped = screens.values @ cholsigmainv`, `warped_intercept = cholsigmainv.sum(axis=0)`.
  5. For each gene pair, run OLS of gene A's warped profile on [warped intercept, gene B's warped profile]. The standard error is `sqrt(pinv(XᵀX)[1,1]·RSS/df)` with df = n − 2. The p-value is `2·stdtr(df, −|β/SE|)`, which is two-sided and symmetric in A and B. The code also saves `sign(β)`.
- **Results:**
  - GLS p-values are calibrated (median p = 0.48), while Pearson is inflated (median p = 0.21).
  - 93,575 pairs at 10% FDR; 99.4% are positively signed.
  - Top-partner enrichment, GLS vs bias-corrected Pearson: CORUM ~160× vs ~120×; hu.MAP ~130× vs ~90×; STRING 7.5× vs 5.5×.
  - Removing the PCA correction hurts Pearson but not GLS.
  - Syntenic (same-chromosome) modules are flagged as possible copy-number artefacts. Syntenic pairs are less enriched in CORUM and hu.MAP than non-syntenic pairs.
- **Implication:** GLS is equivalent to a Pearson t-test after whitening the line × line covariance. Its ranking is a monotone function of |r| between whitened profiles with an intercept. The baseline must specify how signs are handled. A two-sided p ranks strong anti-correlations high; a signed score is the fairer co-complex score. Report both, and document which one is "the GLS baseline".

### 1.2 Normalization and debiasing

**Gheorghe V, Hart T (2022).** "Optimal construction of a functional interaction network from pooled library CRISPR fitness screens." *BMC Bioinformatics* 23:510. doi:10.1186/s12859-022-05078-y [verified]
- **Design:** a factorial comparison on Avana (17.8k genes × 730 lines).
  - Scores: BF, Z-score, CERES 20Q4, Chronos 22Q2.
  - Variance normalization: quantile, or Boyle OR-PCA with 4 PCs.
  - Covariance normalization: Cholesky whitening, Σ⁻¹ = LLᵀ, W = Lᵀ; or PCA whitening, W = D^−½Eᵀ.
  - Similarity: Pearson or OLS.
  - Evaluation: log-likelihood score against a "clean" Reactome with mitochondrial translation and OXPHOS removed, using hu.MAP 2.0 as the reference precision.
- **Findings:**
  - Covariance whitening was the largest single improvement for every score type.
  - PCA whitening is about equal to Cholesky whitening.
  - After whitening, Pearson and OLS are mathematically identical.
  - Quantile or OR normalization helps BF and Z-scores but barely helps CERES or Chronos.
  - CERES beat Chronos: at the hu.MAP2 threshold, CERES kept 17k edges, BF 15k and Chronos 12k. Chronos was weaker on transcription and translation.
  - Partial correlation is useful for "moonlighting" trios such as KRAS–NRAS–RAF1, but was not used for global ranking.
- **Implication:** our score is Chronos, so whitening is mandatory. The whitening step is the main free design choice: shrinkage, rank, and which genes estimate Σ. We cannot use CERES or BF (not provided). Whether ScreenGeneEffect behaves differently is an open empirical question.

**Rahman M, Billmann M, Costanzo M, …, Myers CL (2021).** "A method for benchmarking genetic screens reveals a predominant mitochondrial bias." *Mol Syst Biol* 17:e10013. doi:10.15252/msb.202010013. Code: FLEX_R. [verified]
- **Method:** FLEX ranks pairs by Pearson on DepMap 19Q2 CERES (563 lines) against CORUM 3.0. It reports global PR curves, contribution-diversity plots, and a module-level PR (mPR) that counts distinct complexes.
- **Findings:**
  - 2 of 1,697 CORUM complexes (ETC complex I and the 55S mito-ribosome) supply about 76% of the 3,348 true positives at precision 0.5.
  - The signal is linked to protein stability and screen sampling time: ETC strength predicts sampling time (r = 0.61).
  - Pearson and Spearman perform about the same; both beat cosine and dot product.
  - GLS (Wainberg) gives much greater functional diversity.
  - Performance saturates at about 300 screens and drops sharply at 15 or fewer.
- **Implication:**
  - Pooled AUPRC can be dominated by one or two large complexes, so report per-complex (macro) metrics.
  - Assay-length and doubling-time confounders exist; CRISPRConfounders contains `ScreenDoublingTime`.
  - hu.MAP 3.0 removed complexes with more than 30 subunits (§2.4), so the mito-ribosome and complex I are probably absent or fragmented in our labels. Still, the train set has a 144-protein connected component, so the risk of dominance is real.

**Hassan AZ, Ward HN, Rahman M, Billmann M, Lee Y, Myers CL (2023).** "Dimensionality reduction methods for extracting functional networks from large-scale CRISPR screens." *Mol Syst Biol* 19:e11657. doi:10.15252/msb.202311657. Preprint doi:10.1101/2023.02.22.529573. [verified methods; mostly qualitative results]
- **Data:** DepMap 20Q2 CERES, with NAs imputed by gene means.
- **Normalizations compared:**
  - PCA: subtract the reconstruction from the first n PCs, n ∈ {1, 3, …, 19}.
  - Robust PCA: keep the sparse component S, with λ = f/√max(r,c) for f = 0.7…1.3.
  - A 1-D-latent convolutional autoencoder: subtract its reconstruction.
- **"Onion" normalization:** compute Pearson on each normalized layer, convert to an affinity (SNF `affinityMatrix`, k = 5, σ = 0.3–0.5), and fuse the layers with Similarity Network Fusion (10 iterations). Simple averaging and max were also tried (supplementary).
- **Findings:**
  - Robust PCA + onion (RPCO) performs best. It beats GLS and OLF normalization, and GLS ≈ autoencoder-onion (qualitative; exact AUPRCs are only in figures, so unverified).
  - OLF normalization does not reduce the mitochondrial signal much.
  - At 15k top edges, RPCO covers 12,017 genes versus 1,524 for raw DepMap.
- **Implication:** ensembling across the number of removed components ("layers") is a proven, label-free way to recover smaller complexes. A supervised model can learn the layer weighting directly instead of using SNF.

**Rohde T, Demirtas TY, Süsser S, Shaw AH, Kaulich M, Billmann M (2025).** "BaCoN (Balanced Correlation Network) improves prediction of gene buffering." *Mol Syst Biol* 21:807–824. doi:10.1038/s44320-025-00103-7. Preprint doi:10.1101/2024.07.01.601598. [verified for the preprint; exact formula not retrieved]
- **Method:** a post-hoc correction of a correlation matrix. Each ρ_AB is penalized by the density of equal-or-more-extreme values in gene A's and gene B's correlation distributions, so a pair scores high only if it stands out for both genes.
- **Results:** with Chronos 23Q2 and 1,019 lines, top-100 paralog hits rose from 16 to 71. Cholesky whitening + BaCoN was best. No gain at 100 lines; gains from about 200 lines.
- **Implication:** per-gene rank or density balancing (a cousin of mutual rank) is cheap and stacks with whitening. It suppresses hub genes, such as broadly essential genes that correlate with everything.

**Obayashi T, Kinoshita K (2009).** "Rank of correlation coefficient as a comparable measure for biological significance of gene coexpression." *DNA Res* (2009). [abstract-only]
- **Method:** mutual rank (MR) is the geometric mean of the rank of B among A's partners and the rank of A among B's partners.
- **Finding:** MR is a better functional indicator than raw PCC and more robust to sample redundancy.
- **Implication:** apply MR to whitened correlations as a label-free feature. I found no DepMap-specific MR benchmark (unverified for DepMap).

**Langfelder P, Horvath S (2012).** "Fast R functions for robust correlations and hierarchical clustering." *J Stat Softw* 46(11). [abstract-only]
- **What it is:** biweight midcorrelation (bicor), a median/MAD-based robust correlation.
- **Implication:** DepMap profiles have heavy tails, such as lineage-specific dependencies driven by a few lines. Bicor sits between Pearson (outlier-driven) and Spearman. I found no published DepMap comparison (unverified).

**Lazar NH, …, (Recursion) (2024).** "High-resolution genome-wide mapping of chromosome-arm-scale truncations induced by CRISPR–Cas9 editing." *Nat Genet* 56:1482–1493. [abstract and summaries]
- **Finding:** a CRISPR "proximity bias": knockouts resemble unrelated genes on the same chromosome arm, caused by telomeric arm truncations. The bias is visible in DepMap 19Q3 and 22Q2 even after processing. They propose an arm-based correction.
- **Pipeline note:** the DepMap pipeline now aligns per-arm medians across screens (DepMap forum, "CRISPR Pipeline and Analysis").
- **Implication:**
  - Same-arm and same-chromosome co-essentiality is partly artefactual.
  - HGNC chromosome and arm features should be used to *discount* that bias, and to model it explicitly as a nuisance interaction with correlation, not as a co-complex prior.
  - Check whether hu.MAP positives are enriched for same-chromosome pairs (e.g. histone and PCDH clusters), because that would make the feature look predictive for the wrong reason.

**Partial correlation and graphical lasso.** I found no DepMap paper that uses a global sparse-precision (graphical-lasso) network to predict complexes; that claim is unverified. Gheorghe & Hart used partial correlation only locally, for trios. General statistical literature: Epskamp & Fried 2018 tutorial (arXiv:1607.01367); scale limits of graphical lasso at genome scale (arXiv:2403.18602).
- **Implication:**
  - Genes × lines is 18k × 1.36k, so p ≫ n. A full-genome precision matrix needs shrinkage, a low-rank-plus-diagonal factor model, or restriction to the 1,116 labelled genes plus neighbours.
  - It is a plausible new direction, because co-complex partners should stay conditionally dependent after conditioning on module-level factors.
  - Treat it as exploratory.

**Cell-line weighting and covariates.**
- GLS is the principled way to weight cell lines: it down-weights redundant, related lines.
- Kim 2019 drops low-quality screens.
- BaCoN tried lineage as a regression covariate and ComBat; ComBat + BaCoN was among the best.
- Chen X, …, Lu C (2023), *Nat Commun* 14:4259 (doi:10.1038/s41467-023-39990-5) built 145 co-dependency modules and put them in context by lineage, mutation and expression [abstract-only]. This supports lineage-conditional co-dependency as extra information.

### 1.3 Other relevant network and representation work
- **Pan J, …, Hahn WC, Tsherniak A, Zitnik M (2022).** "Sparse dictionary learning recovers pleiotropy from human cell fitness screens" (Webster). *Cell Systems*. arXiv:2111.06247. [abstract-only]
  - Graph-regularized dictionary learning on 2,921 high-variance genes × 675 lines gives 220 latent "functions", with about 4 functions per gene.
  - Gene-loading similarity is an alternative pair score that separates pleiotropic genes.
  - Its restriction to high-variance genes is a warning: low-variance complex members carry little signal.
- **Billmann M, Costanzo M, …, Myers CL, Moffat J, Boone C (2025 preprint; published in *Cell* 2026 per journal page, unverified).** "A global genetic interaction map of a human cell reveals conserved principles of genetic networks." doi:10.1101/2025.06.30.662193 [verified for the preprint]
  - DepMap co-essentiality sub-networks of 298 lines overlap each other at Jaccard about 0.5 (PCC > 0.4) but overlap the HAP1 GI network at less than 0.1.
  - Genes with variable dependency carry the network signal; core-essential genes with uniform effects have few interactions.
  - **Implication:** the gene's own dependency variance is a key moderator of how informative its correlations are. Use it as an interaction feature.
- **Schmid EW, Walter JC (2025).** "Predictomes, a classifier-curated database of AlphaFold-modeled protein-protein interactions." *Mol Cell* 85:1216–1232. doi:10.1016/j.molcel.2025.01.034 [verified]
  - The SPOC random forest uses DepMap Chronos vectors (1,095 lines) as one "biological" feature, alongside co-expression, BioGRID, ORCS and localization.
  - "All features were derived by combining the metrics of both proteins so that the classifier could not identify or learn from protein identities." Biological features carried 45% of the Gini importance.
  - **Implication:** use symmetric pair-combined features (min, max, product, |diff|) and never raw per-protein identity features.
- **Dury J (2026).** "Beyond expression similarity: contrastive learning recovers functional gene associations…" arXiv:2603.20955 [abstract-only]
  - A contrastive MLP trained on STRING reports cross-boundary AUC 0.947 on a DepMap gene set and 0.826 on a node-disjoint split.
  - It notes that negative sampling is the main difficulty.
  - Single-author preprint, metric not comparable to ours, low evidential weight.
- **De Kegel B, Quinn N, Thompson NA, Adams DJ, Ryan CJ (2021).** "Comprehensive prediction of robust synthetic lethality between paralog pairs in cancer cell lines." *Cell Systems*. [abstract-only]
  - A supervised pair classifier built on DepMap plus pair features.
  - It is a template for pair-feature engineering; its target is synthetic lethality, not co-complex membership.
- **No paper was found that trains a supervised or contrastive model on DepMap profiles specifically to predict hu.MAP or CORUM co-complex pairs and evaluates it complex-disjointly.** Learned DepMap gene embeddings (autoencoder or VAE) for co-complex prediction also seem absent; the onion paper's 1-D autoencoder is used for *normalization*. I searched for "DEPOI" and similar names and found nothing. **This gap is our opportunity, and also a warning: there are no published leakage-free numbers to compare against.**

---

## 2. hu.MAP 3.0 and how its split was built

**Fischer SN, Claussen ER, Kourtis S, Sdelci S, Orchard S, Hermjakob H, Kustatscher G, Drew K (2025).** "hu.MAP3.0: atlas of human protein complexes by integration of >25,000 proteomic experiments." *Mol Syst Biol* 21:911–943. doi:10.1038/s44320-025-00121-5. PMC12222714. Preprint doi:10.1101/2024.10.11.617930. [Methods read]

**Gold standard:**
- Complex Portal human complexes, version 2022_07_11, with isoforms removed.
- Complexes with more than 30 subunits were removed.
- Redundancy was removed iteratively: for any pair with Jaccard > 0.6, the larger complex was dropped.

**Split:**
- "The list of complexes is randomly shuffled and split into test and training sets, 50:50." The split is **complex-level**.
- Complexes sharing protein *pairs* across sets were handled by removing the largest one, and intersecting PPIs were removed.
- **Proteins are not stated to be disjoint.** A protein in two complexes can probably appear in both splits (unverified). The test set is therefore mostly C3/C2 pairs relative to our training proteins, with possibly a few C1.

**Labels:**
- "Positive pairs are proteins within the same complex; negative pairs are … two proteins present in separate complexes and never in the same complex."
- Negatives therefore come only from complex members. "Being in any complex" carries no signal, but a gene-level propensity to have many partners does.
- The hu.MAP classifier subsampled 10,000 negatives for training; AutoGluon picked an XGBoost ensemble by test AUPRC. hu.MAP 3.0's own AUPRC was 0.68, against 0.61 for hu.MAP 2.0 and 0.38 for BioPlex3. These numbers come from proteomics features and are not comparable to ours.

**Implications for us:**
1. The withheld test set consists of *other complexes*. Our primary validation should split by **connected component of the train-positive graph**, which approximates a complex-level split. There are 232 components and the largest has 144 proteins. That split should be combined with a **protein-disjoint (C3)** check.
2. Test negatives will be pairs between different test complexes. The prevalence will be similarly low (train: 0.52%), but the exact prevalence is unknown.
3. No complex has more than 30 subunits, so very large-complex dominance (FLEX) is reduced. Overlapping complexes still merge into large components, so per-component metrics remain necessary.

---

## 3. Evaluation pitfalls for pair prediction

**Park Y, Marcotte EM (2012).** "Flaws in evaluation schemes for pair-input computational predictions." *Nat Methods* 9:1134–1136. doi:10.1038/nmeth.2259 [verified]
- **Classes:** C1 means both proteins are in training, C2 one, C3 neither.
- **Findings:** all 7 PPI predictors performed significantly and often much worse on C2 and C3. In random cross-validation more than 99% of test pairs were C1. The cause is the different representation of individual proteins among positives and negatives.
- **Implication:** report C1, C2 and C3 separately. In our data, negatives are almost all non-positive pairs among 1,151 proteins, so per-protein positive degree (median 3, max 41) is learnable. Random pair splits will inflate any model that has node-level features or learned embeddings.

**Bernett J, Blumenthal DB, List M (2024).** "Cracking the black box of deep sequence-based protein–protein interaction prediction." *Brief Bioinform* 25:bbae076. doi:10.1093/bib/bbae076 [abstract-only]
- **Finding:** with random splits, models learn node degree and similarity shortcuts. After leakage is removed, performance is about random. Simple degree and similarity baselines match deep models.
- **Implication:** include a "node-degree-only" negative-control model; it should score near prevalence on C3 splits. Report every gain relative to the Pearson, Spearman and GLS baselines on the same split.

**Joeres R, Blumenthal DB, Kalinina OV (2025).** "Data splitting to avoid information leakage with DataSAIL." *Nat Commun* 16. doi:10.1038/s41467-025-58606-8 [abstract-only]
- **What it is:** leakage-reduced splitting posed as an optimization problem (NP-hard; solved with a clustering + ILP heuristic), including two-dimensional (pair) splits.
- **Implication:** a principled tool for building protein- and component-disjoint folds. Grouped K-fold on components is a simpler equivalent for our case.

**Saito T, Rehmsmeier M (2015).** *PLoS ONE* 10:e0118432 [abstract-only]
- **Finding:** PR is more informative than ROC on imbalanced data.

**McDermott M, et al. (2024).** "A closer look at AUROC and AUPRC under class imbalance." *NeurIPS 2024*. arXiv:2401.06091 [abstract-only]
- **Finding:** AUPRC preferentially rewards improvements in sub-populations with more positives.
- **Implication:** pooled AUPRC will favour large or dense components. Pair it with macro-averaged per-component AUPRC, and report AUROC as secondary.

**Boyd K, Eng KH, Page CD (2013).** "Area under the precision-recall curve: point estimates and confidence intervals." *ECML PKDD*, LNCS 8190:451–466. doi:10.1007/978-3-642-40994-3_29 [abstract-only]
- **Finding:** average precision is a robust point estimate. Some common AUCPR intervals are invalid; logit or binomial intervals are recommended.
- **Implication:** use average precision. Pairs are not independent (they share proteins and complexes), so CIs should come from a **cluster bootstrap over components or proteins**, with paired resampling to compare methods. This is my recommendation and is not taken from a paper on pair data (unverified as a published standard).

**Further pitfalls specific to this project, inferred from the papers above:**
- Transductive label use: features built from training-label neighbourhoods do not transfer to complexes that are absent from training.
- Selecting hyperparameters on the same folds used for reporting.
- Per-gene propensity, such as mean essentiality, acting as a proxy for protein identity.

---

## 4. Confounders in DepMap

- **Chronos.** Dempster JM, et al. (2021), *Genome Biol* 22:343, doi:10.1186/s13059-021-02540-7 [abstract-only].
  - A population-dynamics model with the "lowest copy number and screen quality bias of evaluated methods".
  - Vinceti, Iorio, Dempster et al. (2024), *Genome Biol*, doi:10.1186/s13059-024-03336-1: AC-Chronos best corrects copy-number and proximity bias in joint processing [abstract-only].
  - A DepMap forum reply (unofficial, unverified) says Chronos2 has been used since 23Q4.
- **Library and batch effects.**
  - Dempster et al. (2019), *Nat Commun* 10:5817, doi:10.1038/s41467-019-13805-y: Broad and Sanger screens are concordant; batch effects are driven mainly by **library and assay length**.
  - Pacini C, et al. (2021), *Nat Commun* 12:1661: integrating Broad and Sanger needs per-gene batch correction, using 168 overlapping lines.
  - Rauscher 2018: library effects are gene-specific.
- **How the DepMap pipeline produces the two matrices** (forum post "CRISPR Pipeline and Analysis"; 26Q1 and 26Q3 release notes) [verified]:
  - `ScreenGeneEffect` = Chronos run **per library/screen type**, concatenated across libraries, with one row per screen. A model can have several screens.
  - `CRISPRGeneEffect` = Chronos run **jointly across libraries** with built-in library batch correction, one row per model.
  - Both are copy-number corrected after Chronos. They are scaled so that the median of common essentials is −1 and of non-essentials 0, with screen-quality correction and chromosome-arm median alignment (the proximity-bias step).
  - Screen QC filters include NNMD < −1.25, replicate Pearson > 0.19, and others.
- **Release-specific notes.**
  - 26Q1 extended library correction to genes present in more than one batch.
  - 26Q3 fixed a regularization bug that had under-corrected library effects through 26Q1. It also fixed two bugs for genes present in only some libraries, mean-centred chrY genes before copy-number correction, and removed 96 TKOv3/Brunello-only genes.
  - **Implication:**
    - 26Q3 `CRISPRGeneEffect` should have fewer library artefacts than earlier releases.
    - `ScreenGeneEffect` is *not* jointly corrected, so it is the natural input for per-library co-essentiality: Avana-only (1,024 AV01 screens), KY-only (316) and Humagne-only (202).
    - Concatenating its libraries without correction will inject library structure.
- **Does restricting to Avana help?**
  - The classic co-essentiality papers (Boyle, Kim, Wainberg, Gheorghe & Hart) all used Avana-only data, but none compared Avana-only against integrated data for complex recovery. I found no direct published answer (unverified).
  - FLEX shows saturation at about 300 screens, so Avana-only (about 1,000 screens) loses little power.
  - This is a cheap empirical question for us: compare Avana-only ScreenGeneEffect with all of CRISPRGeneEffect.
- **Lineage.** GLS whitening absorbs lineage relatedness. Lineage-specific co-dependencies exist (Chen 2023).
- **Mitochondrial, doubling time and sampling time.** See Rahman 2021. Candidate covariates include `ScreenDoublingTime` and the quality columns in CRISPRConfounders. The column definitions were not found in official documentation; per the forum they are in the release README (unverified).
- **Copy number and proximity.** See Kim 2019 (20 Mb exclusion), Wainberg (syntenic flagging) and Lazar 2024 (same-arm bias).

---

## 5. Licenses (exact statements found)

| Resource | Statement | Source |
|---|---|---|
| DepMap data | "Generally, data generated by DepMap are made available under the permissive CC BY 4.0 license. Data from other projects hosted on the DepMap portal may have different licenses." (DepMap staff, forum, 30 Jul 2020). Per forum, the exact per-file licence is shown in the portal download panel. | https://forum.depmap.org/t/license-for-data-found-in-the-depmap-portal/130 |
| DepMap portal page | "You're always welcome to grab everything from the downloads section — please don't scrape the portal." No other terms found on that page. | https://depmap.org/portal/terms |
| hu.MAP 3.0 paper | Open access, CC BY (Europe PMC licence field). | https://doi.org/10.1038/s44320-025-00121-5 |
| hu.MAP 3.0 data | Hugging Face dataset card (DrewLab/hu.MAP_3.0) metadata: `license: cc-by-4.0`. The hu.MAP website /about page states **no** licence. The licence of the files the user obtained is therefore **unverified**; CC BY 4.0 is likely if they came from the HF release. Cite Fischer et al. 2025. | https://huggingface.co/datasets/DrewLab/hu.MAP_3.0 ; https://humap3.proteincomplexes.org/about |
| Complex Portal (upstream of hu.MAP labels) | Not found. EMBL-EBI states a general preference for CC0, but the Complex Portal licence itself is **unverified**. | https://www.ebi.ac.uk/licencing |
| HGNC | "all data is released under the Creative Commons Public Domain (CC0) License"; "any form of reuse of the content is permitted". **DATA.md should change "freely available" to CC0.** | https://www.genenames.org/about/license/ |
| kundajelab/coessentiality (GLS reference code) | MIT licence. | https://github.com/kundajelab/coessentiality |

DATA.md lists DepMap as CC BY 4.0, which matches the forum statement. Attribution means citing DepMap and the release.

---

## 6. Implications for design

### 6.1 Ranked method ideas

Ranking reflects expected AUPRC gain on complex-disjoint splits relative to risk.

1. **Whitened correlation with tuned covariance estimation** (GLS done well).
   - Approach: estimate the line × line Σ with Ledoit–Wolf or eigenvalue-floor shrinkage, or with PCA whitening that keeps the top-m eigen-directions. Try estimating Σ from all genes or only from non-essential/OR genes. Handle NaNs by imputation instead of dropping genes.
   - Benefit: high. Gheorghe & Hart found whitening the largest single gain.
   - Risk: low. It is close to the GLS baseline, so the margin may be small.
2. **Multi-layer ("onion") similarity stack fed to a supervised pair model.**
   - Approach: compute Pearson, Spearman and bicor on (a) raw data, (b) data with k PCs removed (k = 1…20), (c) robust-PCA sparse data, and (d) whitened data. Use a gradient-boosted classifier over these features instead of SNF.
   - Benefit: high (Hassan 2023).
   - Risk: moderate overfitting with about 3.2k positives in 232 components. Keep the model shallow and tune on grouped CV.
3. **Per-gene rank balancing on every layer.**
   - Approach: mutual rank, BaCoN-style density penalty, or row/column z-scoring of the correlation matrix.
   - Benefit: medium–high. It suppresses hub and essential-gene correlations and reportedly stacks with whitening (BaCoN).
   - Risk: low, label-free and cheap.
4. **Second-order (neighbourhood) similarity.**
   - Approach: correlate whole rows of the whitened correlation matrix, or use topological overlap or shared top-k neighbours. Complex members share partners.
   - Benefit: medium–high.
   - Risk: low, label-free. Do not use training labels in neighbourhoods.
5. **Per-library and per-context correlations as separate features.**
   - Approach: compute similarity separately on Avana, KY and Humagne ScreenGeneEffect and on lineage groups; feed max, mean and consistency as features. Also compare Avana-only input against CRISPRGeneEffect.
   - Benefit: medium. Replication across libraries is evidence against artefacts, and context-specific complexes are captured.
   - Risk: smaller groups give noisier estimates.
6. **Confounder regression before correlation.**
   - Approach: regress each gene on CRISPRConfounders covariates (doubling time, screen quality, library) and possibly OR-PCs.
   - Benefit: medium; targets the sampling-time and mitochondrial artefact (Rahman).
   - Risk: low–medium; it can remove real growth-related biology.
7. **Symmetric gene-level context features.**
   - Approach: own mean, SD, skew and fraction dependent, combined as min, max, product and |diff| (Predictomes style), plus interactions with similarity. Billmann found variable-dependency genes carry the signal.
   - Benefit: medium.
   - Risk: **high leakage risk**. Must be validated on C3/component-disjoint splits against a degree-only negative control.
8. **Chromosome nuisance features.**
   - Approach: same chromosome, same arm, and genomic distance (from the HGNC `location` field) as interactions that let the model discount proximity-driven correlation (Lazar, Kim, Wainberg).
   - Benefit: low–medium.
   - Risk: medium. Real co-complex gene clusters exist, so audit how positives are enriched for synteny.
9. **Partial correlation or regularized precision.**
   - Approach: shrinkage precision on the 1,116 labelled genes plus their top neighbours, or a low-rank-plus-diagonal factor model, to separate direct from indirect co-dependency.
   - Benefit: uncertain, possibly medium.
   - Risk: p ≫ n instability and compute cost.
10. **Self-supervised gene embeddings, then a pair scorer.**
    - Approach: PCA of whitened data, or a denoising autoencoder on profiles; then cosine similarity or a small Siamese head trained on train positives.
    - Benefit: uncertain.
    - Risk: high overfitting and memorization of protein identity. Only adopt if it beats ideas 1–4 on C3/component-disjoint CV.
11. **Sparse dictionary or NMF factor-loading similarity** (Webster-like).
    - Benefit: low–medium; it handles pleiotropy and moonlighting subunits.
    - Risk: medium compute and tuning effort.
12. **Robust estimators.**
    - Approach: bicor or winsorized Pearson inside whitening; Spearman of whitened profiles.
    - Benefit: low–medium.
    - Risk: low.
13. **Rank-average ensemble of the best label-free scores.**
    - Benefit: a strong, nearly overfit-proof fallback if the supervised stacker does not generalize across components.
    - Risk: low.

### 6.2 Evaluation recommendations
- **Primary split.** Use grouped K-fold over *connected components* of the train-positive graph, which mirrors hu.MAP's complex-level 50:50 split. Keep negatives whose two proteins fall in the same fold's components, and drop cross-fold negatives from evaluation.
- **Secondary split.** Use protein-disjoint (C3) splits, report C1, C2 and C3 separately, and include a random pair split only for contrast.
- **Final internal test.** Hold out a fixed set of components, untouched until the end, as an internal final test. Fix the seed list.
- **Metrics.**
  - Primary: average precision at natural prevalence.
  - Also report macro AUPRC per component (FLEX mPR analogue), AUPRC excluding the largest component, AUROC, and precision@k.
- **Uncertainty.** Use a cluster bootstrap over components for CIs, and paired bootstrap differences against Pearson, Spearman and GLS.
- **Controls.**
  - Degree-only and gene-propensity-only models should score near prevalence on component-disjoint folds.
  - Label-permuted features as a negative control.
  - OR-gene pairs should score as background.
  - A toy simulation with planted complexes and planted lineage covariance, where GLS should beat Pearson.
- **Baseline fidelity.**
  - Implement GLS exactly as in the kundajelab code: OR 4-PC removal, drop 4 lines, Cholesky of inv(cov), two-sided t-test.
  - Document NaN handling and sign handling (signed vs two-sided).
  - Also run "GLS without OR removal" as a sensitivity check.
  - All baselines use the same gene universe and pairs.
- **Report runtime** for every method alongside AUPRC.

## Errata (from round-1 prior-work review, reviews/round1/prior_work.md)
1. §1.1 GLS step 3: in kundajelab `gene_pairs.py`, `screens.dropna(axis=1)` drops **cell lines** with any missing gene (screens is genes × lines), not genes; the `dropna` in `load_screens.py` applies only to the OR-gene subset. A literal port on 26Q3 (1,352 NaN genes) could drop most lines, so a documented NaN protocol is required.
2. Hassan 2023: AE layers use latent sizes {1,2,3,4,5,10} plus the unnormalised layer (not one 1-D AE); RPCA λ should be the standard 1/√max(r,c); AUPRC claims are figure-only, CORUM gold standard only.
3. BaCoN gains were shown on paralog pairs, not complexes. Rahman 2021 numeric claims unverified.
