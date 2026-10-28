# Round 1 review: algorithms and compute (Phases A–D)

## 0. Quick empirical probes (informational only)
I ran label-free probes from a scratchpad script that is not in the repo.
- Data: CRISPRGeneEffect 26Q3, NaNs mean-imputed, all 596,597 scorable pairs.
- Metric: pooled AP; prevalence is 0.0053.
- **Caveat:** these pairs include the future IFT pairs. Nothing here is a selection: re-tune everything on dev CV and log this look.

| Score | AP |
|---|---|
| Pearson / Spearman / GLS signed (cached, exact) | 0.202 / 0.222 / 0.255 |
| GLS, mean-imputed | 0.253 |
| GLS, Σ shrunk, α = 0.2 / 0.35 / 0.9 | 0.299 / 0.301 / 0.245 |
| Eigenvalue floor (90th pct) · PCA-whitening top-400 | 0.297 · 0.259 |
| Remove top-k PCs then Pearson, k = 10 / 20 / 50 | 0.292 / 0.298 / 0.250 |
| α = 0.2, then genome-wide row z-score · mutual rank | 0.309 · 0.307 |
| Spearman of whitened profiles | 0.245 |
| Second-order: dense · top-1%-sparsified | 0.205 · 0.310 |
| First + second order: z-average · **rank** average | 0.314 · 0.263 |

Shrinkage here means λ_i ← (1 − α)λ_i + α·mean(λ). The top eigenvalue is 82% of the trace.

Two take-aways:
1. Cheap label-free changes beat GLS by about +0.05 AP.
2. Combinations must preserve the top of the list: z-scores work, ranks do not.

## 1. Phase A (label-free): well chosen, but re-prioritise

**A1 is the main lever. Treat it as a spectral filter f(λ) on the eigenvectors of the line × line Σ.**
- GLS is f = λ^{-1/2}. PC removal zeroes the top k. Ridge, Ledoit–Wolf and eigenvalue floors are f = (λ + τ)^{-1/2}.
- Write one `whiten(X, f)` and sweep ridge τ, optionally with hard removal of the top k ≤ 5. One eigendecomposition (0.5 s) serves every variant.
- Tune τ on dev CV, by AP. Ledoit–Wolf's analytic α targets covariance error, which is a different objective.
- Keep "Σ estimated from non-essential/OR genes only" as one variant.

**Missing values.** Imputation vs exact marginalisation changed GLS by only 0.002.
- Explore on an imputed dense matrix (mean or low-rank).
- Keep the exact pattern-block code for the faithful baseline and the final model.
- **Ranking caveat:** t = r·√(df/(1−r²)) depends on df, and df varies across missingness patterns. Compare r versus Fisher-z·√(df−3) as the cross-pattern ranking score on dev.

**A5 (rank balancing): keep.** Use row z-scores, with each labelled gene's row computed against all 18k genes; this gave about +0.01. Prefer row z-scores to mutual rank.

**A6 (second-order): keep, sparsified only.** Use top-q% neighbours with q ∈ {0.5, 1, 2, 5}. The dense form is useless. The sparsified form was the strongest single layer.

**A4 (robust estimators after whitening): deprioritise.** Whitening mixes cell lines, so ranks lose meaning.

**A2 (confounder regression) is a special case of A1:** it removes the covariate directions. It is cheap, but expect little gain.

**A3 (per-library):** run the same A1/A5/A6 pipeline on ScreenGeneEffect AV, KY and CD, each with its own Σ. Use these as stacker inputs. Avana-only is also a standalone candidate.

## 2. Phase B (supervised stacker)

**Model.** LightGBM over 10–30 symmetric layer scores, plus symmetric gene statistics (min/max/product) and chromosome nuisance features.

**To transfer to unseen complexes:**
- Monotone-increasing constraints on the similarity features.
- num_leaves ≤ 15 and min_child_samples ≥ 200.
- Early stopping on inner complex-grouped AP.

**Class imbalance.**
- Use no class weights and no focal loss. AP only depends on ranking, and up-weighting distorts the top of the list.
- Bag 5–10 models, each trained on all positives plus a fresh negative subsample at about 1:50.
- Also try `lambdarank` with one query per protein. It optimises within-protein ranking and resists per-protein propensity.

**Component dominance.** Weight each positive by 1/√(component n_pos) so the 144-protein component does not set the trees. Report weighted and unweighted results.

**Controls.** The 3.7(b) propensity control must use the same model class.

**Logistic regression.** Fitted in-sample on 3 features, LR scored *below* plain z-averaging (0.292 vs 0.314). Log-loss is not aligned with AP, so select hyperparameters by AP.

## 3. Phase C (metric learning): narrow it

A free MLP encoder trained on about 2.5k positives among about 900 proteins will memorise proteins and components. Use a capacity ladder instead, adopting each step only if it beats the previous step on complex-CV:
1. **Learned spectral filter:** f(λ) on 8–16 log-spaced bins, fitted with a smooth-AP or pairwise logistic loss. It transfers to unseen genes. Full-batch L-BFGS on CPU takes seconds.
2. **Diagonal cell-line weights** after whitening, with L2 shrinkage toward 1.
3. **Low-rank projection** (r = 32–128) trained with InfoNCE, using all labelled train genes as negatives and line dropout. This takes about 1 minute per fold; MPS is optional.

## 4. Phase D (ensemble)

- Replace "rank average" with z- or logit-scale averaging, or with the stacker. In the probe, rank averaging cost 0.05 AP.
- Keep a label-free fallback: a z-average of shrunk GLS, row-z and sparse second-order scores, with weights fixed on dev. It is hard to overfit and is likely the safest "clearly beats GLS" candidate.

## 5. Compute and implementation

- **All-pairs GLS** is Wnᵀ Wn after whitening, removing the warped intercept and normalising. This is the FWL form already in `baselines.py`.
- **Labelled × all genes** is 1114 × 18434, about 160 MB. Each variant takes 2–5 s.
- **Second-order** needs the full 18k × 18k matrix (1.36 GB float32). Build it in 2k-row blocks and keep only sparse top-q% CSR rows.
- **Caching:** cache pair-score vectors (about 2.4 MB each) under a config hash.
- **LightGBM** on 600k × 30 takes under 10 s per fit, so 25 folds × bagging is affordable.
- **Bootstrap mismatch:** `metrics.bootstrap_*` is a stratified *pair* bootstrap, but DESIGN §3.5 specifies a component cluster bootstrap. The weighted-AP code supports per-component multiplicity weights. Fix this before headline comparisons.

## 6. Likely wastes

- Genome-wide graphical lasso (p ≫ n; subsumed by spectral filtering).
- SNF.
- Bicor or Spearman on whitened data.
- Dense second-order similarity.
- Deep Siamese MLPs.
- Dictionary learning.
- PC removal with k ≥ 50.
- Truncated PCA-whitening.

## 7. Proposed changes

| # | Change | Expected effect | Priority |
|---|---|---|---|
| P1 | A1 as a spectral-filter family (ridge τ + top-k removal), tuned on dev | ≈ +0.04–0.05 AP over GLS | High |
| P2 | Imputed matrix for exploration; choose a cross-pattern ranking (r vs Fisher-z·√(df−3)) | Speed; avoids a df artefact | High |
| P3 | Genome-wide row-z and sparsified second-order layers | ≈ +0.01–0.02 | High |
| P4 | Z/logit averaging or stacker instead of rank averaging; label-free fallback ensemble | Avoids ≈ −0.05 | High |
| P5 | Shallow, monotone, bagged-negative LightGBM; no class weights; component weights; lambdarank by protein | Transfer to unseen complexes | Medium |
| P6 | Phase C as a capacity ladder | Less memorisation | Medium |
| P7 | Component cluster bootstrap in `metrics.py` | Correctness of §3.5 | High |
| P8 | Deprioritise the §6 wastes | Saves effort | Low |
