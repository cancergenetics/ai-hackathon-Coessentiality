# Round 1 review: prior work (assessor)

Scope: LITERATURE.md claims spot-checked against primary sources; DESIGN.md v1 method ranking. hu.MAP test files were not sought.

## 1. Spot-check of LITERATURE.md

| Claim | Result |
|---|---|
| GLS recipe: OR-gene PCA(4) with inverse_transform subtraction; `iloc[:, :-4]`; `cholesky(inv(np.cov(screens.T)))`; warped intercept = column sums; `2*stdtr(df, -abs(coef/se))`, df = n-2; sign saved separately | **Confirmed** against load_screens.py and gene_pairs.py. Regression is `np.linalg.lstsq`, with SE from the pinv(XᵀX) formula. |
| NaN handling ("drop genes with any NaN"; 26Q3 faithful port loses 1,352 genes) | **ERROR.** `screens` is genes × lines. `screens.dropna(axis=1, inplace=True)` drops **cell lines** that have any missing gene (code comment: "Remove cell lines with any missing genes ... You can use other strategies ... like imputing, removing genes with missing cell lines"). The `dropna()` in load_screens.py applies only to the OR-gene subset used to fit the PCA. With 1,352 NaN-containing genes, a literal port could discard most or all lines. The "faithful" baseline therefore needs a documented choice (drop NaN genes, or impute), and that choice affects the GLS baseline's strength. |
| Onion (Hassan 2023): layers, Pearson per layer, SNF (k=5, σ 0.3–0.5, 10 iterations), RPCO best, GLS ≈ AEO, OLF poor at removing the mito signal | **Confirmed** (PMC10632734). Evidence is mostly figures, with CORUM as the only gold standard (FLEX). No AUPRC numbers are given in the text. Two small mismatches: the RPCA λ is written as f/√max(r,c) in LITERATURE but f ÷ max(r,c) in the text I retrieved, which is probably an extraction artefact, so use the standard 1/√max(r,c). The AE layers are latent sizes {1,2,3,4,5,10} plus the unnormalized layer, not a single "1-D-latent" AE. GLS reduces the mito signal "less distinctly" than AEO/RPCO. |
| Gheorghe & Hart: whitening largest gain; Cholesky ≈ PCA whitening; Pearson ≡ OLS after whitening; CERES > Chronos | **Confirmed** (abstract and PMC summary). The abstract says BF and CERES are best, and Chronos is worse. |
| BaCoN: 23Q2 Chronos, 1,019 lines, paralogs 16→71, no gain at 100 lines, gains from ~200 | **Confirmed**. Caveat missing in LITERATURE: the target is paralog/buffering pairs, not complexes (see 2.2). Cholesky + BaCoN was best on the relaxed set, but the authors call it higher-risk. |
| Rahman 2021 (2 complexes ≈ 76% of TPs; Pearson ≈ Spearman; ~300-screen saturation) | **Not verified** by me (paywalled/PDF unreadable). A different source, Hassan, reports ~80% of TPs at precision 0.8 from mito complexes, which is consistent in spirit. |

New work found: a 2026 Nature paper (s41586-026-11003-7) uses 26Q1 co-essentiality with a graph attention network for proximity antigens; not a complex benchmark, paywalled, low relevance. I found no 2024–2026 paper that trains a supervised model on DepMap profiles for co-complex prediction with a complex-disjoint evaluation, which agrees with LITERATURE §1.3.

## 2. Does the design reflect the evidence?

### 2.1 Ordering is broadly sound
Label-free whitened layers feed a supervised stacker, then embeddings last. This matches the evidence: whitening gives the largest label-free gain (Gheorghe), the onion-style layer ensemble helps (Hassan), and there is no evidence that learned embeddings beat these.

### 2.2 Concerns

1. **GLS baseline is under-specified.** GLS code is the recipe, but the NaN rule above is ambiguous. The design should fix the rule in 3.6 and report GLS under at least two NaN strategies, picking the best on dev only.
2. **Rank balancing (A5) can hurt complexes.** BaCoN and mutual rank penalize genes with many high-correlation neighbours. Complex members are exactly a clique of mutually high correlations, and large complexes (up to 30 subunits, plus merged components of up to 144) will be penalised. BaCoN was validated on paralogs, not complexes. Treat A5 as a feature (raw and balanced both supplied), not as a replacement.
3. **No explicit "onion" stack in Phase A.** Hassan's strongest finding (multi-layer fusion over removed-PC counts and RPCA) is only partly present. A3/A4 cover library and estimator, but the k-sweep is only mentioned inside A1.
4. **Chronos evidence is against the input.** Gheorghe found Chronos weaker than CERES/BF, which the project cannot use; treat gene-effect scaling as a design variable (change 5).

## 3. Proposed changes (in scope)

| # | Change | Why | Priority |
|---|---|---|---|
| 1 | Correct LITERATURE §1.1 step 3 (drop cell lines, not genes) and define the GLS NaN protocol in DESIGN 3.6: (a) drop genes with any NaN, (b) mean/KNN-impute, (c) drop-lines literal port when it leaves ≥ 300 lines. Report all, with the best on dev as "GLS baseline". | Baseline fidelity; avoids a strawman or accidental weakness. | High |
| 2 | Make an explicit onion stack a Phase A item: layers = raw, PC-removed k ∈ {1,3,…,19}, RPCA (f 0.7–1.3, λ = f/√max(r,c)), whitened; fuse by (i) SNF (Hassan settings), (ii) rank-average, (iii) feed raw layers to Phase B. Compare supervised stacking with SNF. | Hassan is the best label-free published method for complex recovery and is a likely competing "strong baseline" for the user. | High |
| 3 | Add RPCO as an extra reported baseline (not required by PROMPT.md but the strongest published competitor). | "Clearly beats" is more credible if the stronger published method is included. | Medium |
| 4 | Provide balanced and unbalanced versions of A5 as separate features, and add a per-complex-size breakdown to detect hub penalization. | Concern 2.2.2. | Medium |
| 5 | Add a gene-effect rescaling variant (per-gene z-score or quantile) as an A-layer, since the Chronos score is weaker than BF in Gheorghe. | Cheap. | Low |
| 6 | Add partial-correlation features restricted to labelled genes plus top neighbours as an exploratory Phase A item. | Gheorghe found utility for moonlighting subunits. Not decisive. | Low |
| 7 | In LITERATURE, mark Hassan numeric claims as figure-only and add the BaCoN paralog-target caveat. | Accuracy. | Low |

No change needed to the evaluation design from the prior-work perspective; component-grouped CV, C1/C2/C3 reporting and degree-only negative controls match Park & Marcotte and Bernett et al.
