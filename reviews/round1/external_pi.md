# Round 1 review: external PI

Reviewer: external agentic PI. Date: 2026-10-27. Read: PROMPT.md, DESIGN.md v1, LITERATURE.md (§2, §6 in full; §1.2–1.3 skimmed), DATA.md, RESEARCH_TREE.md, data_profile.json, and src/splits.py and src/metrics.py (read only).

## Verdict
The design is sound and careful about leakage. Component-grouped CV correctly mirrors hu.MAP's complex-level 50:50 split. The method plan starts with the right lever (whitening done well), and the controls are appropriate. The changes below cover:
- evaluation details that could make the dev ranking disagree with the withheld test;
- one missing integrity rule;
- an effect-size floor and power check for "clearly beats";
- reprioritisation.

## Proposed changes

**C1 (high). Prevalence mismatch.**
- Problem: within-fold evaluation keeps only pairs inside the held-out proteins. With 5 folds on the 80% dev set, that is roughly 180 proteins, 16k pairs and 510 positives per fold, so prevalence is about 3%. The IFT is about 2.6%. The withheld test is probably about 0.5%, like train.
- Why it matters: AP rewards the high-precision head much more at 0.5%, so the winner at 3% can lose at 0.5%.
- Change:
  - Evaluate label-free layers on all dev pairs at once; with no training, folds only add noise.
  - For supervised models and for the decision rule, also report prevalence-reweighted AP (negatives weighted to 0.52%). Weighted AP already exists in `metrics.py`.
  - Log per-fold prevalence.
- Effect: model selection matches the test regime.

**C2 (high). Never use unlabelled pairs.**
- Problem: 1,151 proteins give 661,825 pairs. Of these, 3,314 are positive and 631,465 negative, so **27,046 pairs are in neither file**. Under hu.MAP's negative definition ("never in the same complex"), many are probably co-complex in test complexes or in removed complexes with more than 30 subunits.
- Using them in any way (weak positives, PU learning, hard-negative mining over all pairs, diagnostics) would infer withheld test information, which PROMPT.md forbids.
- Change:
  - Add an explicit rule to §6.
  - Add a unit test asserting that the pair universe equals the union of the two files.
- Effect: closes an integrity hole an agent could fall into innocently.

**C3 (high). IFT discipline and the final refit.**
- Problem: "Once per candidate" still allows selection on the IFT.
- Change:
  - Pre-declare at most two candidates (best label-free, best supervised/ensemble) and a selection rule based on dev CV only. The IFT is confirmatory only. If the chosen method fails there, report it; do not swap in the other candidate.
  - State that the frozen pipeline is then refit on all train pairs before `predict_pairs.py`.
- Effect: an unbiased IFT estimate, and about 25% more training data for the final model.

**C4 (high). Make "clearly beats" honest and achievable.**
- Add a pre-registered **minimum effect**: relative AUPRC gain of at least 10% over the best baseline variant, on prevalence-reweighted dev-CV AP, in addition to the CI excluding 0.
- **Power check now.** Draw IFT-sized subsets of dev components (about 46 components) and measure the ΔAP CI width for GLS vs Pearson. If an IFT-sized test cannot resolve a 10% gain, pre-register a weaker IFT criterion before looking: same sign, and point estimate ≥ 50% of the dev effect.
- **Bootstrap.**
  - §3.5 specifies a component cluster bootstrap. `metrics.py` currently resamples pos/neg i.i.d., which understates the variance.
  - For supervised models, average per-fold AP within each replicate instead of pooling scores from different fold models, because those scores are not on a common scale.
- Add a guard: macro per-component AP is not worse than the best baseline. This catches a win carried by the 144-protein component.

**C5 (high). Collapse Phase A into one cheap screening harness.**
- Sweep in one pass over labelled × genome:
  - covariance estimator: GLS-OR, Ledoit–Wolf, eigenvalue floor, PCA-whiten rank m;
  - post-processing: none, mutual rank, BaCoN-like z-scoring;
  - first- vs second-order similarity.
- Make a **rank-average of the top label-free layers** a mandatory candidate. With 232 components it is the likeliest robust winner and is nearly overfit-proof.
- Phase B should use a compact set of about 10–20 layers, not the full grid.
- Effect: most of the attainable gain quickly, plus a strong fallback.

**C6 (medium). Recast Phase C as low-parameter metric learning.**
- Problem: an MLP encoder over 1,361 lines, trained on about 3k positives in about 185 components, will memorise.
- Change:
  - Learn per-line (or per-lineage/library/confounder-parametrised) weights, or a low-rank correction to Σ⁻¹ shrunk toward GLS, with score xₐᵀWx_b (GLS is W = Σ⁻¹).
  - Defer the deep encoder and graphical lasso unless C5 stalls.
- Effect: a supervised gain that can transfer to unseen complexes.

**C7 (medium). Clarify how gene-level features are judged.**
- Don't expect control 3.7(b) to sit at prevalence. Essential genes sit in larger complexes, which may transfer legitimately; treat it as a reference floor.
- Adopt gene features only if similarity + gene features beats similarity-only on component-disjoint, prevalence-reweighted AP.
- Use symmetric combinations only. Forbid label-derived node features such as training degree.

**C8 (medium). Hard-negative stratification.**
- Report AP separately for negatives inside the same positive component (sibling sub-complexes, 2-hop pairs) and negatives across components. This shows where errors come from and whether second-order features help.

**C9 (medium). Audit chromosome features.**
- Measure same-arm enrichment among positives vs negatives.
- Use chromosome features only as interactions with similarity, and adopt them only if they win on component-disjoint CV.

**C10 (low). Lineage context layers.**
- Add per-lineage-group similarity (max, consistency) from Model.csv inside the C5 harness.

**C11 (low). Unmapped proteins.**
- Check ScreenGeneEffect coverage before falling back to the minimum score.
- Break ties consistently, and report how much the fallback costs in "all-pairs" AP.

## Wasteful or cuttable
- 5×5 repeated CV and the full C1/C2/C3 breakdowns should be kept for the final 3–5 candidates only.
- Use a single seed during exploration, as already written.

## Not needed
No change to data sources, baselines, the primary metric or the split philosophy.
