## Findings (written after the runs)

1. **Equal-weight z-averages that include the Laplacian embedding E are the best combinations.** `zavg_BEF` 0.470 and
   `zavg_BEHF` 0.469 (5-seed mean HT-AP) vs C_single B 0.448 and the best single layer E 0.463. Pooled-OOF Δ vs B:
   `zavg_BEF` +0.014..+0.021, CI above 0 in 5/5 seeds (seed 2 only barely: lower bound +0.0005); `zavg_BEHF`
   +0.016..+0.019 with the tightest CIs (lower bound ≥ +0.009 in every seed). BEF vs BEHF is a tie (0.001 difference),
   and D3's argmax picks BEF. `zavg_BF` (no E) helps mainly on small complexes (all-pairs CI above 0 in 2/5 seeds, no-top10 in 5/5).
2. **The gain now includes the small/medium complexes.** Without the top-10 clusters, `zavg_BEF` − B is +0.024..+0.031
   (CI excludes 0 in 5/5 seeds), and the macro per-cluster AP rises from 0.351 to 0.395 (fold mean; pooled 0.338 to
   0.384). This removes the main caveat of C_single, whose gain was concentrated in large dense complexes. Every
   essentiality stratum improves (both non-essential 0.307 to 0.327).
3. **Supervision adds nothing over equal weights.** LR and LGBM on the 7 inputs give 0.449–0.466 fold-mean HT-AP but
   only 0.427–0.445 pooled OOF (fold models put scores on different scales). Pooled-OOF Δ vs B has CIs that include 0
   in every seed. Δ vs the 4 z-averages is ≤ 0 in 20/20 seed × z-average cells for both LR variants and 16/20 for each
   LGBM variant, and no CI lies above 0 in any cell (LR+genebin is below 0 in 11/20). They do
   have the highest macro per-cluster AP (0.415–0.418) and AUROC (0.814–0.818), and on the no-top10 pairs they are
   ≈ `zavg_BEF`. Binned gene features hurt LR (−0.013) and do not help LGBM (−0.002). The LR coefficients have mixed
   signs on the collinear second-order inputs (B, SO, F negative; H, F0, E positive), so they cannot be read as layer
   importance. LGBM puts most of its gain on E (0.50–0.55). Under the promise rule fixed in advance (a supervised
   entry must beat the best z-average in 5-seed mean and in pooled-OOF Δ in ≥3/5 seeds), **the optional stacker (4)
   was not run.**
4. **Selection optimism is small.** Nested-by-fold selection among the 4 z-averages gives 0.468 (5-seed mean) vs an
   in-sample best of 0.470. Among all 8 combinations it gives 0.466 (it sometimes picks LR/LGBM in-sample: seed 1 LR,
   seed 4 LGBM). BEF or BEHF is selected in 22/25 outer folds of the z-average selection.
5. **vs baselines (best dev variants):** `zavg_BEF` − GLS +0.175..+0.193; − Spearman +0.166..+0.232; − Pearson
   +0.190..+0.249. All CIs exclude 0 in 5/5 seeds. Fold-wise, `zavg_BEF` beats GLS/Spearman/Pearson in 5/5 folds in
   every seed, and B in 5/5 folds (4/5 in seed 2).
6. **Controls.** Held-fold label shuffle for all 4 supervised entries gives 0.042–0.043 vs an HT prevalence of 0.041
   (pass).
7. **Runtime** (fresh process, all 596,597 scorable pairs, 8-core laptop shared with another agent's jobs): B 23 s;
   `zavg_BEF` 125 s (E's Laplacian eigenmap ≈ 100 s dominates); `zavg_BEHF` 233 s (this run was slowed by
   concurrent load; the members' standalone times sum to ≈ 180 s); supervised inputs 153 s + fit/score 0.6–0.9 s per
   fold (LR) or 3–4 s per fold (LGBM, 5 bags with early stopping). Peak RSS ≤ 6.1 GB.

Process notes: rows with src/ clean were run at commit e884f44 (all of seed 0 and part of seed 1). Partway through seed 1
another agent's untracked `src/predict.py` made src/ dirty (it is not imported by any code used here). The remaining
runs used `--allow-dirty`, which is recorded in each run_record and trial, and the agent later committed that file
(b8fdf0f). The first runtime stage attempt failed on a config-dispatch bug in this script (rowzso configs, fixed) and
was rerun. OOF arrays are in data_cache/phaseB_v2_oof/ (gitignored).
