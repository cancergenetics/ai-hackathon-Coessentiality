# Round 2 review: AI failure modes + algorithms (DESIGN v2)

Read-only. No label metrics computed. Current `src/` has no `run_record`, trials log, IFT split or cluster bootstrap yet (`metrics._boot_weights` is still a pair bootstrap), so these are promises, not facts.

## 1. Round-1 status
- **Adopted:** cluster bootstrap (design only), IFT plus look log, `fit(train_idx)`, invariance tests, gene-ID test, unlabelled-pair rule, equal baseline budget, z-averaging, spectral-filter family, sparsified second-order, capacity ladder, independent rerun, handback format.
- **Dropped silently:** dirty-tree refusal; "no tree entry without commit and tests"; "new feature family needs a dated design revision"; "a failed negative control is a finding"; held-fold-label-shuffle test; LGBM over Pearson+Spearman+GLS ablation (separates supervision from new layers); weighted and unweighted component results.
- **D1 contamination:** the shrinkage, row-z and top-1% priors came from a look at all pairs, including future IFT pairs. Record it as look #0 in `ift_looks.jsonl`.

## 2. New risks
- **R2-1 Protein identity via gene stats (B).** Each protein is in about 1,150 pairs, and min/max of mean effect and SD are near-unique per gene. A leaf of ≥200 pairs can isolate one protein's propensity. The "no gene-ID features" test cannot see it.
- **R2-2 Label-free layers** (row-z, mutual rank, top-1% neighbours) leak only if the background or neighbour universe is restricted to labelled or train genes, or fed by a label graph. Enforce this structurally.
- **R2-3 Seed repeats are not fresh evidence for Phase A.** τ, k and q are tuned on all dev pairs, and label-free scores ignore the fold seed. Only IFT is unbiased, so §3.7(i) overstates the evidence.
- **R2-4 z-averaging.** Per-fold or per-file standardisation breaks batch invariance. Scores from different fold models cannot be pooled. Tuned weights over many candidates carry winner's curse.
- **R2-5 Phase C.** Louvain communities of one real complex sit in both train and test, and low-rank InfoNCE exploits that. The 144-protein component dominates the gradients.
- **R2-6 IFT is convention only.** The agent can read the labels. A random draw can swallow the giant component.
- **R2-7 `trials.jsonl` is voluntary.** One-off probes, crashed runs and unlogged grid points escape it.

## 3. Likely waste
- A2 (a special case of A1).
- Mutual rank (keep row-z).
- A5 beyond one run.
- Phase C rungs 2–3.
- Phase D if B already ingests A6.
- A fine τ×k grid, which only buys winner's curse.

## 4. Proposals

| # | Change | Fixes | Pri |
|---|---|---|---|
| 1 | Phase A layers live in a label-free module with signature `(X, ia, ib)`, cached by config hash before any split. B reads only the cache. Test: layer outputs are bit-identical for any labels or split. | R2-2 | P0 |
| 2 | B gene features coarsened (≤5 quantile bins). `num_leaves ≤ 7`, `min_child_samples ≥ 2000`. Add the gene-stat-only AP under random vs cluster split as an identifiability check. | R2-1 | P0 |
| 3 | Freeze A configs (coarse grid, 1-SE rule) before B. State that seeds measure split variance and IFT is the only unbiased check. Add the Pearson+Spearman+GLS+gene-stats LGBM ablation. | R2-3 | P0 |
| 4 | z or logit constants come from a frozen genome-wide random-pair null, never from the evaluated file. Weights are equal over ≤4 layers, or L2 logistic regression inside train folds. | R2-4 | P0 |
| 5 | IFT stratified by community size (none above 30% of IFT positives), pre-declared in code. Only `final_eval.py` may import the IFT loader (AST test). ≤2 candidates and config hashes committed before the first look. | R2-6 | P0 |
| 6 | One `log_trial()` inside the single dev-evaluation entry point, writing `start` and `end` records, so crashes show. Log every grid point (config hash, split fingerprint, per-fold AP, commit). REPORT trial count = line count. Reviewers get no out-of-repo label probes. | R2-7 | P0 |
| 7 | C adoption gate: beats the previous rung in ≥4/5 folds **and** under strict-component grouping (§3.4a). 1/√n_pos weights. Init at shrunk-Σ with a learning curve starting at epoch 0. Inner cluster early stopping. Report the train-vs-held-out gap. Time-box 2–3 hours. | R2-5 | P1 |
| 8 | Restore the dropped round-1 rules (dirty-tree refusal, tree preconditions, failed control = finding, held-fold shuffle test). | §1 | P1 |
| 9 | Merge D into B, with A6 as the label-free fallback candidate. Drop A2 and mutual rank. | §3 | P2 |
