# Key insights

**The main point:** the biology lesson is that genes in the same complex are recognised more reliably by *who else they depend on together with* than by how closely their own profiles match, at least for large complexes. The method lesson is that, with a few thousand labelled pairs, careful label-free geometry beat every supervised model the agents tried, and that pooled benchmark numbers hide where a gain comes from.

Insights are sorted into three groups and ranked within each by how general and how surprising they are. Each links to where it arose. "Shown" means measured in this project; "believed" means a likely explanation that was not tested.

## 1. Scientific and domain insights

These would interest a researcher working with DepMap or protein complexes, whatever method they use.

<a id="ins-s1"></a>
**S1. Shared co-dependency partners carry more co-complex signal than direct profile similarity, but mainly for large complexes.** *Shown.* Scoring a pair by the overlap of the two genes' top-44 partner lists raised development AP from 0.395 (direct similarity of the same cleaned profiles) to 0.448. The gain came almost entirely from the ten largest complex clusters: averaged complex by complex, the overlap score tied GLS (0.351 vs 0.349). *Believed:* small complexes have too few partners to share. → [Turning points 4](JOURNEY.md#tp4) and [5](JOURNEY.md#tp5); [Step 3](METHOD.md#step3).

<a id="ins-s2"></a>
**S2. How the cell-line covariance is estimated matters more than how many components are removed.** *Shown.* Standardising each gene before estimating the covariance added +0.061 AP, the largest single change to the first-order score. After it, removing 0, 5 or 20 components and ridge strengths from 0.01 to 0.3 all gave 0.359–0.365. Unregularised whitening, as in GLS, was the worst whitened setting. *Believed, not tested:* without standardisation, a few hundred strongly essential genes with large variance dominate the covariance. → [Turning point 4](JOURNEY.md#tp4); [Step 1](METHOD.md#step1).

<a id="ins-s3"></a>
**S3. A few thousand labelled pairs did not improve on label-free scores for unseen complexes.** *Shown.* With about 1,800–2,000 training positives per fold in about 150 clusters, logistic regression and gradient-boosted trees over seven similarity views (0.462, 0.466), adding each gene's essentiality statistics (no gain), and learned cell-line weights or projections (within ±0.003 of their starting point) never beat the plain average (0.470). The learned models fitted the training genes but not new complexes. → [Turning point 7](JOURNEY.md#tp7).

<a id="ins-s4"></a>
**S4. Pooled precision–recall on co-complex benchmarks is dominated by a handful of large complexes.** *Shown.* The ten largest of 190 development clusters held 64% of out-of-fold positives. The neighbour-overlap score was +0.155 AP over GLS in pooled terms but +0.039 (not significant) without those ten clusters. This confirms, on DepMap and hu.MAP, the warning of Rahman et al. (2021) for CORUM. → [Turning point 5](JOURNEY.md#tp5).

<a id="ins-s5"></a>
**S5. Anti-correlation carries no useful co-complex signal on balance.** *Shown.* For Pearson, Spearman and GLS alike, signed scores beat absolute values (for example GLS signed 0.255 vs two-sided 0.241 on all pairs). The olfactory-receptor correction of Boyle et al. also lowered GLS on this release. → [Turning point 3](JOURNEY.md#tp3); [appendix](JOURNEY.md#appendix).

<a id="ins-s6"></a>
**S6. Neighbourhood scores slightly amplify the CRISPR chromosome-position artefact.** *Shown.* Among unrelated olfactory-receptor gene pairs, 8.4% scored above the 99th percentile of random pairs under the final score (22.6% for same-chromosome pairs), against 6.6% under GLS (expected: 1%). Removing same-chromosome pairs from the benchmark did not lower AP, so the artefact does not drive the benchmark result. → [Limitations](METHOD.md#limitations).

<a id="ins-s7"></a>
**S7. Essential genes are more often in complexes, which any benchmark must account for.** *Shown.* A model that sees only each gene's own essentiality statistics, and no pair similarity, reached 2.6 times the positive rate on unseen complexes. Labelled genes are 31% common-essential against 9% genome-wide. Results should therefore be reported by essentiality group, as here: the final score led GLS in all three groups. → [Turning point 7](JOURNEY.md#tp7).

## 2. Methodological insights

These generalise to other pair-prediction or similarity-scoring methods.

<a id="ins-m1"></a>
**M1. When negatives are "all other pairs among the same proteins", hold out whole complexes, then reweight.** *Shown.* Random pair splits let a model learn which proteins have many partners. Holding out clusters of complexes fixes that but changes the mix of pairs in each test fold: too many within-cluster pairs, so too many positives and too many hard negatives (uniform reweighting over-weighted hard negatives 4–5×). Weighting each pair by the inverse of its chance of being in the fold (Horvitz–Thompson) restores the natural 0.53% positive rate. → [Turning point 3](JOURNEY.md#tp3).

<a id="ins-m2"></a>
**M2. Simulate the power of a held-out test before fixing its pass rule.** *Shown.* The held-out test had 57 clusters but an effective size of about 8 independent clusters. A confidence-interval rule on it would have passed with probability at most 0.22 even if the true gain were 20%. The rule was changed before the test was opened. → [Turning point 3](JOURNEY.md#tp3).

<a id="ins-m3"></a>
**M3. Combine label-free views by z-scoring against random pairs and averaging with equal weights; distrust greedy ensembles.** *Shown.* A greedy forward-selected ensemble gained in-sample and nothing under nested cross-validation (0.428 vs 0.428 for the best single view). A small set of pre-specified equal-weight averages lost almost nothing under nesting (0.468 vs 0.470). → [Turning points 5](JOURNEY.md#tp5) and [8](JOURNEY.md#tp8); [Step 5](METHOD.md#step5).

<a id="ins-m4"></a>
**M4. Report every headline gain with its per-cluster average, the gain without the largest clusters, several split seeds and a nested check.** *Shown.* Each of these changed a conclusion here: the per-cluster average exposed B's weakness, seeds showed which differences were stable, and nesting removed the greedy-ensemble gain. → [Turning point 5](JOURNEY.md#tp5).

<a id="ins-m5"></a>
**M5. Have a separate agent re-implement each headline result from raw data, and vary the chosen settings slightly.** *Shown.* All three verifications reproduced the numbers exactly. Varying the embedding settings showed that the final score's small margin over B was stable in size but not in significance (confidence interval above zero in 5 of 5 seeds at 512 dimensions, 2 of 5 at 256). → [Turning points 5](JOURNEY.md#tp5) and [8](JOURNEY.md#tp8).

<a id="ins-m6"></a>
**M6. Pilot a new supervised approach on one fold against its starting point before running it in full.** *Shown.* Three supervised phases each gave the same null answer only after full five-seed runs. A one-fold pilot with a stop rule (|gain| < 0.005) would have saved most of that compute. → [Turning point 7](JOURNEY.md#tp7).

<a id="ins-m7"></a>
**M7. Agents that do not evaluate must not see labels.** *Shown.* Two design reviewers, asked only to comment, computed AP on all labelled pairs, including pairs later held out for the final test. Nothing was chosen from those numbers, but the test was no longer fully untouched. Briefs now say "compute no AP or AUROC". → [Turning point 2](JOURNEY.md#tp2); [the research process](JOURNEY.md#process).

## 3. Engineering details for reproduction

Needed only to rerun or port the code.

- **Community detection must be seeded and hash-stable.** Louvain tie-breaking depended on Python's string hashing, so the cluster assignment changed between runs. All results use `PYTHONHASHSEED=0`; the assignment is committed as `results/splits/protein_assignment.csv`. ([The research process](JOURNEY.md#process))
- **The covariance is rank-deficient in this DepMap release** (rank 1,357 of 1,361); GLS drops 4 dependent cell lines found by pivoted QR. The method's ridge term makes this harmless.
- **Remove the intercept exactly** by projecting onto an orthonormal basis of the complement of the all-ones vector (Householder), rather than re-centring cell lines.
- **The eigenmap is seeded** (ARPACK start vector from seed 0) and built from a float32 normalised adjacency, as in the original run; results agree to about 1e-5 across reimplementations. The spectrum is nearly degenerate around 512 dimensions, so small numerical changes can rotate coordinates without changing cosines much.
- **Genes with ≥10% missing values** are embedded by a Nyström extension and flagged; they never enter the covariance, the null pairs or other genes' partner lists.
- **Genome-wide row statistics are computed in fixed gene blocks** so that each pair's score is identical whatever pairs are requested and in whatever order (tested).
- **torch and LightGBM in one pytest process can deadlock on OpenMP**; run those test files separately.
- **Apple Silicon:** the default Anaconda was x86 under Rosetta; a native arm64 Python 3.12 environment was needed.
- **Storage:** all-pairs z-scores stored as float16 lose precision above |z| ≈ 32 (errors up to 0.016); the ranked pair table stores float32.
- **Runtime:** about 2 minutes and 4–7 GB of memory for the full method on an 8-core laptop; the eigenmap takes most of it.
