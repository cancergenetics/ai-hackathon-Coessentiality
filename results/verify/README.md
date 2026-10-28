# Independent verification: Phase-A best layer `rowz[socos_q0.25[spec_k20_t0.1_std]]`

Script: `scripts/verify/verify_phaseA_best.py` (runtime ~41 s). Output: `results/verify/phaseA_best.json`.
The verifier did not produce the result it checks.

## Method (independent re-implementation)

Everything is rebuilt from `datasets/DepMAP/CRISPRGeneEffect.csv` (1361 × 18434). The script does **not** import
`src/layers*.py`, `src/metrics.py`, `src/baselines.py` or `src/evaluate.py`. The only project import is
`src/splits.make_split`, used for the fold assignment.

Dev pairs come from the committed `protein_assignment.csv` (its SHA-256 matches `results/splits/protein_assignment.sha256`).
Scorable rows of `data_cache/pairs.pkl` that have any non-dev protein are dropped as soon as the file is read. No
metric is computed on IFT pairs, and the IFT loader is never touched.

Layer steps, written from the docstrings:
1. Gene-centre each gene on its observed mean and set NaN to 0. Divide by the observed SD (ddof = 0). This choice
   reproduces the manifest's `mean_eig = 0.998630` exactly.
2. Background (BG) genes are those with NaN < 10%: 17,525 genes.
3. Build a Householder basis of 1⊥. Compute Σ = Y'Y / G_BG over BG genes, then eigendecompose it.
4. Weights are w = 0 for the top 20 eigenvalues and (λ + 0.1·mean λ)^-1/2 for the rest. Filtered profiles are scaled to unit norm, and the similarity S is their cosine.
5. Compute row moments of S over BG \ {self}. Balanced z is B = (z_ab + z_ba)/2. Take the top K = round(0.25% · 17525) = 44 BG neighbours by B, with weights max(B, 0).
6. Take the weighted cosine of the neighbour vectors with partner exclusion. The dot product is unchanged; each norm drops the partner's entry.
7. Compute exact row moments of that second-order score over BG \ {self}, then the row-z: (z_ab + z_ba)/2.

AP is a step-wise weighted AP with tied scores grouped. HT weights are 1 within a cluster and K = 5 between clusters
in a fold, or 1/f for the pooled metric over all dev pairs. The paired cluster bootstrap uses 1000 resamples, gives a
within-cluster pair weight m_c and a between-cluster pair weight m_c1·m_c2, and uses seed 0.

## Results

**(1) My scores vs the cache** (322,528 dev pairs; the cache pair order matches the scorable table)

| layer | r | max abs diff | mean abs diff |
|---|---|---|---|
| spec_k20_t0.1_std (v1 col 62) | 1.0000000 | 2.8e-8 | 2.0e-9 |
| socos_q0.25[…] (v2 col 10) | 1.0000000 | 2.6e-7 | 1.8e-9 |
| rowz[socos_q0.25[…]] (v2 col 17) | 1.0000000 | 1.8e-3 | 1.6e-6 |
| pairwise-complete Pearson (baseline cache) | 1.0 | 3.1e-8 | 1.6e-9 |

The largest rowz difference (1.8e-3) affects a handful of pairs. These are most likely float32 near-ties when the
44th neighbour is chosen. They have no effect on AP.

**(2) HT-AP per fold, SEEDS[0].** The split fingerprint `912b53116c50…` is identical to the one in the reported results.

| scorer | f0 | f1 | f2 | f3 | f4 | mean (mine) | reported |
|---|---|---|---|---|---|---|---|
| best layer (my scores) | 0.4418 | 0.3712 | 0.3613 | 0.6765 | 0.3230 | **0.435** | 0.435 |
| socos_q0.25 (my scores) | 0.4328 | 0.3608 | 0.2797 | 0.6193 | 0.3116 | 0.401 | 0.401 |
| spec_k20_t0.1_std (my scores) | 0.4061 | 0.3130 | 0.3034 | 0.5402 | 0.2614 | 0.365 | 0.365 |
| Pearson (my scores) | 0.3625 | 0.2356 | 0.2072 | 0.2792 | 0.1050 | **0.238** | 0.238 |
| GLS-like k0 τ0, mean-imputed (my scores) | 0.3567 | 0.1785 | 0.2495 | 0.4082 | 0.2028 | 0.279 | 0.279 (spec_k0_t0) |
| GLS signed (cached scores, my AP) | 0.3719 | 0.1919 | 0.2679 | 0.4070 | 0.2010 | **0.288** | 0.288 |

Pooled HT-AP over all dev pairs: best 0.426 (reported 0.426), GLS 0.272 (reported 0.272).

Pooled OOF dev (65,943 pairs, 2,264 positives): best 0.411 vs GLS 0.256. **ΔAP = +0.155 [+0.065, +0.246]**
(+60%). The reported value is +0.155 [+0.065, +0.246]. My bootstrap code is independent, but it uses the same seed
and the same resampling scheme, so the identical CI shows the two implementations agree. It is not a second random draw.

**(3) Leakage audit: PASS (no label or IFT input).**
- `src/layers.py` imports only numpy, scipy, sklearn's `ledoit_wolf_shrinkage` and `src.config`. `src/layers_ext.py` imports only `src.layers`.
- `get_engine` loads only the DepMap matrix.
- `scripts/12_build_layers_v2.py` reads the scorable pair table and keeps only `a, b, ia, ib`; the labels are dropped there.
- Neighbour selection (`LayerEngine.neighbours`) runs over all genes in fixed blocks and uses BG columns only.
- My re-implementation uses only genome-wide BG genes and has no access to labels, and it reproduces the cached
  scores to about 1e-8. This shows directly that the cached scores depend only on X.
- Remaining caveat: the hyper-parameters (k = 20, τ = 0.1, standardisation, q = 0.25, row-z on top) were selected on
  SEEDS[0] dev folds over about 150 layers. That is optimism from selection, not label leakage into the scores. The
  IFT gate remains the only unbiased check.

**(4) Negative control.** I shuffled the gene columns of X (seed 0) and reran the full pipeline. HT-AP fold mean is
0.0100 against a mean HT prevalence of 0.0091, a ratio of 1.10. Pooled OOF AP is 0.0082. **PASS**: the score sits
at prevalence once the gene identities are broken.

**(5) Is the gain concentrated?** For each top-10 dev cluster (ranked by within-cluster OOF positives), per-cluster AP
is computed over pairs in its test fold that touch the cluster, with HT weights.

| cluster | within pos | fold | AP best | AP GLS | Δ | pooled-OOF Δ without this cluster |
|---|---|---|---|---|---|---|
| 0 | 325 | 3 | 0.795 | 0.415 | +0.380 | +0.113 |
| 43 | 262 | 1 | 0.446 | 0.237 | +0.209 | +0.142 |
| 84 | 153 | 0 | 0.528 | 0.468 | +0.060 | +0.156 |
| 17 | 145 | 4 | 0.280 | 0.162 | +0.118 | +0.162 |
| 8 | 121 | 1 | 0.279 | 0.119 | +0.160 | +0.158 |
| 15 | 105 | 3 | 0.500 | 0.453 | +0.047 | +0.168 |
| 211 | 94 | 2 | 0.534 | 0.297 | +0.237 | +0.151 |
| 18 | 94 | 4 | 0.356 | 0.219 | +0.137 | +0.158 |
| 12 | 80 | 2 | 0.302 | 0.324 | −0.022 | +0.163 |
| 42 | 36 | 2 | 0.723 | 0.582 | +0.141 | +0.150 |

- The best layer beats GLS on 9 of the top 10 clusters. No single cluster drives the result: dropping cluster 0, the
  largest, still leaves Δ = +0.113.
- The top 10 clusters hold 64% of OOF positives. **After removing all of them, Δ = +0.039 [−0.013, +0.087]**, with
  AP 0.339 vs 0.299 on the remaining 808 positives across 167 clusters. The CI includes 0.
- Across all 177 clusters with positives, the macro per-cluster AP is 0.326 for the best layer vs 0.325 for GLS. The
  best layer wins 41% of clusters, ties 14%, and the median Δ is 0.
- **So the large gain is real and spread over several clusters, but it is carried by the large, dense complexes.** On
  small complexes the layer is roughly level with GLS (+13% relative, not significant). The README's "Profile of the
  gain" note says the same.

## Verdict
The headline is reproduced: HT-AP 0.435 vs GLS 0.288 and Pearson 0.238, and pooled-OOF Δ +0.155 with a CI that
excludes 0. No label or IFT leakage was found, and the negative control sits at prevalence. Two caveats remain:
1. The configuration was selected on dev, so the dev numbers are optimistic.
2. The gain is concentrated in the top 10 clusters. Outside them the advantage over GLS is small and not significant,
   and per-cluster macro AP is level.

---

# Independent verification: Phase-B v2 primary candidate `zavg_BEF`

Script: `scripts/verify/verify_zavg_BEF.py` (runtime about 17 min; most of it is the 1025-eigenpair ARPACK run in the robustness check). Output: `results/verify/zavg_BEF.json`.
The verifier did not produce the result it checks. Claim checked (`results/phaseB_v2/README.md`): `zavg_BEF` = mean of the null-z-scored B, E and F reaches dev HT-AP 0.470 (5-seed mean) vs B 0.448, and the pooled-OOF Δ vs B has a CI above 0 in 5/5 seeds.

## Method
- **Imports.** The script does not import `src/layers*.py`, `src/predict.py`, `src/metrics.py`, `src/evaluate.py` or `src/supervised.py`. Project code it uses: `src.splits.make_split` for folds, and the previous verifier's independent helpers in `verify_phaseA_best.py` (F, B, AP, HT weights, bootstrap).
- **Dev pairs.** Same as the previous verifier: the assignment SHA-256 is checked and IFT rows are dropped on load.
- **E, rebuilt from the `src/layers_graph.py` docstrings.**
  1. Balanced-z top-K BG neighbours on F, with K = round(0.25% · 17525) = 44 and weight max(B, 0).
  2. Union graph max(W, W′), with W′ restricted to BG columns. Non-BG rows keep their directed edges.
  3. N = D^-1/2 A D^-1/2, then `eigsh` for the top 513 eigenpairs (largest algebraic) of N_BG,BG. The leading eigenvector is dropped, and non-BG genes are placed by Nyström extension.
  4. The score is the cosine of the 512-dim embedding rows.
- **Deliberate deviations from the producer.** These test whether the construction is stable:
  - N is kept in float64.
  - The ARPACK start vector uses seed 12345, not SEEDS[0].

  The graph is connected: exactly one eigenvalue is 1, and no node has zero degree.
- **Null moments.** My own 200k BG×BG pair draw (seed 0) is identical to the stored null pairs, with SHA-256 `780b34d9…` matching the manifest. I also drew a second sample with seed 1.

## Results
**Null moments (mine, seed 0, vs manifest).**

| layer | mean shift (in null SD) | relative SD shift |
|---|---|---|
| B | 1e-7 | 2e-7 |
| E | −1.3e-5 | −1e-6 |
| F | 3e-10 | 1e-10 |

A different null sample (seed 1) moves the moments by at most 2% of the SD (B: SD 1.018 vs 0.996). With my own moments in place of the manifest's, HT-AP is identical to 4 decimals.

**Score agreement on the 322,528 dev pairs.**

| scores | r | max abs diff |
|---|---|---|
| my F vs cache | 1.0000000 | 2.8e-8 |
| my B vs cache | 1.0000000 | 1.8e-3 |
| my E vs cache | 0.999997 | 2.7e-3 (mean 9e-5) |
| my zavg_BEF vs cache formula | 0.9999996 | 0.019 (mean 6.6e-4) |
| `predict_pairs.py --method zavg:B,E,F` vs mine | 0.9999996 | 0.019 |
| `predict_pairs.py` vs cache formula | | 0 (identical to mine-vs-cache) |

- The E difference comes from the float32 operator and the start vector. The spectrum is dense at the cut: λ513 ≈ 0.2579, and neighbouring eigenvalues are ≈ 1e-4 apart.
- The predictor's z_B, z_E and z_F agree with mine to 1.8e-3, 0.057 (E's z units) and 8e-7.
- The predictor's seed-0 HT-AP is 0.45584, the same as reported.

**HT-AP (fold mean) per seed.** Mine uses my own AP code on my own scores. Reported values come from `results/phaseB_v2/runs`.

| entry | s0 | s1 | s2 | s3 | s4 | mean (mine) | reported mean |
|---|---|---|---|---|---|---|---|
| zavg_BEF | 0.4558 | 0.4563 | 0.4812 | 0.4687 | 0.4862 | **0.4696** | 0.470 (s0 0.4558, s1 0.4563) |
| B | 0.4348 | 0.4298 | 0.4615 | 0.4490 | 0.4666 | 0.4484 | 0.448 |
| E | 0.4605 | 0.4572 | 0.4708 | 0.4548 | 0.4706 | 0.4628 | 0.463 |
| F | 0.3648 | 0.3767 | 0.4058 | 0.4055 | 0.4237 | 0.3953 | 0.395 |

The largest per-fold difference from the reported values, across all 25 folds, is 2.6e-5 for zavg_BEF and 1.1e-4 for E. The split fingerprints match.

**Pooled-OOF Δ(zavg_BEF − B), paired cluster bootstrap (1000 resamples, bootstrap seed = split seed).**

| seed | Δ | 95% CI | matches reported |
|---|---|---|---|
| 0 | +0.0202 | [+0.0082, +0.0345] | exact |
| 1 | +0.0207 | [+0.0076, +0.0329] | exact |
| 2 | +0.0143 | [+0.0005, +0.0319] | yes |
| 3 | +0.0203 | [+0.0069, +0.0327] | exact |
| 4 | +0.0148 | [+0.0041, +0.0272] | exact |

The CI is above 0 in 5/5 seeds, but in seed 2 only barely.

**Leakage audit: PASS.**
- **E is label-free.** `src/layers_graph.py` imports only numpy, scipy, `src.layers` and `src.config.SEEDS`. `scripts/15_build_graph_layers.py` keeps only `a, b, ia, ib` from the pair table. My label-free rebuild from the CSV reproduces the cache to 2.7e-3 with identical AP.
- **The z-averaging has no fitted or fold-dependent part.**
  - `scripts/21_phaseB_v2.py:zlayer` uses fixed manifest constants, which are reproduced from a genome-wide null sample that uses no labels.
  - The producer's OOF arrays (`data_cache/phaseB_v2_oof/zavg_BEF_*`) equal, in every seed and fold, the single fixed vector mean((cache − null mean)/null SD), with max diff 0. So no per-fold transform is applied.
- **Remaining caveat: selection.** E's configuration (union/lin, q = 0.25, d = 512) was chosen in the Phase-A graph screen on dev, and zavg_BEF was then picked as the argmax of 8 combinations on dev. Nested selection gives 0.468 vs 0.470 in-sample. That is optimism from selection, not leakage.

**Negative control: PASS.** I permuted gene columns (seed 0), rebuilt B, E and F, and recomputed null moments.

| seed | HT-AP | HT prevalence | ratio |
|---|---|---|---|
| 0 | 0.0092 | 0.0091 | 1.01 |
| 1 | 0.0105 | 0.0102 | 1.04 |

**Robustness to E's hyper-parameters.** All rows use my own null moments; mean HT-AP is over 5 seeds.

| E variant | E alone | zavg_BE′F | − B (0.448) | folds won vs B | pooled-OOF CI > 0 |
|---|---|---|---|---|---|
| d = 512, q = 0.25 (claimed) | 0.463 | **0.470** | +0.021 | 24/25 | 5/5 |
| d = 256 | 0.445 | 0.466 | +0.018 | 24/25 | **2/5** (lower bounds −0.004..+0.005) |
| d = 1024 | 0.459 | 0.469 | +0.021 | 24/25 | 4/5 (seed 2 [−0.003, +0.034]) |
| q = 0.5 | 0.460 | 0.469 | +0.020 | 24/25 | 4/5 (seed 2 [−0.001, +0.030]) |

The cached d = 1024 and q = 0.5 layers (`graph_v2`) agree with my rebuilds (r ≥ 0.99999).

## Verdict
**Reproduced.** zavg_BEF gives 0.4696 vs B 0.4484 (5-seed means). Per-fold values agree to ≤ 3e-5, and the pooled-OOF Δ and CIs match the reported ones. The producer's predictor gives the same scores up to E's numerical noise (max 0.019 on a z-average scale of about 1–20), and its rank correlation with mine is 0.999997.

No leakage was found, and the negative control sits at prevalence.

**How robust the gain is:**
- **The point estimate is not fragile.** Every E variant gives 0.466–0.470 and beats B in 24/25 folds.
- **The significance margin is thin.** The seed-2 lower bound is +0.0005, and it falls below 0 in 1–3 of 5 seeds for d = 1024, q = 0.5 and d = 256.
- **The "CI > 0 in 5/5" result is specific to the dev-selected E configuration.** Reading it as a robust significance result would overstate it. The size of the improvement (≈ +0.02 AP) is robust; its significance is marginal.

---

# Independent verification: IFT look #1 (`results/ift/look01_2026-10-27.json`)

Script: `scripts/verify/verify_ift_look1.py` (~4 min with predictor runs). Output: `results/verify/ift_look1.json`.
This was logged in `results/ift_looks.jsonl` as look `"1-verify"` before the first run. The candidates are the same
and no selection was made. The verifier did not produce the result it checks.

## Method
- **No imports from the producer's evaluation code.** The script does not import `src/ift.py`, `src/metrics.py`,
  `src/evaluate.py`, `src/scorers.py`, `src/universe.py` or `scripts/final_eval.py`. From `src` it uses only
  `src/data.py` (`depmap_columns`, `build_uniprot_map`) for UniProt → DepMap mapping.
- **IFT pair set.** It is rebuilt from the raw `datasets/HuMAP/Train_*_pairs.txt` and the committed
  `results/splits/protein_assignment.csv` (SHA-256 checked). An IFT pair has both proteins on side `ift`. A pair is
  scorable when both proteins map to different genes.
- **HT weights.** Within-cluster pairs get weight 1 and between-cluster pairs get 1/f. f = mapped IFT proteins /
  all mapped proteins = 295/1114.
- **Weighted AP.** I wrote my own weighted AP, with tied scores grouped. It matches sklearn exactly on a toy example
  with ties, both weighted and unweighted.
- **Scores, route (a).** The producer's `scripts/predict_pairs.py` (raw CSV) is run as a black box. Its input is a
  shuffled IFT pair-ID file with no labels. Methods: `zavg:B,E,F`, `B`, `gls`, `pearson`, `spearman`.
- **Scores, route (b).** The verifier re-implementations of B, E and F (`verify_phaseA_best.py`,
  `verify_zavg_BEF.py`) plus my own pairwise-complete Pearson. These are built from the raw CSV on the IFT pairs.
- **Bootstrap.** Own paired cluster bootstrap over the 57 IFT clusters, 1,000 resamples, **seed 20261027** (the
  producer used seed 0).

## Results
**Counts reproduced exactly:**
- 847 positive / 40,786 negative IFT pairs in 57 clusters, with 0 positives between clusters.
- f = 0.264811 and HT prevalence 0.00552.
- 9 positive / 2,352 negative IFT pairs are unmapped.
- Dev has 2,299 positive / 320,229 negative pairs.
- The committed `mapped` column agrees with `src.data` for all 1,151 proteins.

**Scores.**
- The predictor's mapped flag equals my scorable flag.
- My zavg_BEF against the predictor: r = 0.9999997, Spearman 0.999995, max abs diff 0.057. The cause is near-ties in
  neighbour and eigensolver choices. AP is identical to 4 d.p.
- My B against the predictor: r = 0.999999998.
- My Pearson against the predictor: max diff 4e-8.
- Own-null versus manifest-null z moments give identical AP.

**IFT metrics.** Mine are computed from the predictor scores; the look01 values are identical to ≤1e-16 under the
definitions below.

| method | HT-AP | raw AP | macro-cluster AP | no-largest-comp | AP incl. unmapped | both non-ess | mixed | both ess |
|---|---|---|---|---|---|---|---|---|
| zavg_BEF | 0.3451 | 0.4381 | 0.3646 | 0.4695 | 0.3320 | 0.2759 | 0.3878 | 0.3942 |
| B | 0.3174 | 0.3987 | 0.2965 | 0.4411 | 0.3054 | 0.2524 | 0.3679 | 0.3600 |
| GLS | 0.2423 | 0.3055 | 0.3686 | 0.3496 | 0.2332 | 0.2137 | 0.3490 | 0.2628 |
| Spearman | 0.1463 | 0.2462 | 0.2349 | 0.2206 | 0.1408 | 0.1021 | 0.2036 | 0.1807 |
| Pearson | 0.1341 | 0.2336 | 0.2343 | 0.1976 | 0.1291 | 0.0884 | 0.1987 | 0.1533 |

The own re-implementation of zavg_BEF gives the same HT-AP, 0.3451, and its other metrics agree to 4 d.p.

**Paired cluster bootstrap of ΔHT-AP, 95% CI.** Mine uses seed 20261027; look01 used seed 0.

| comparison | Δ | mine | look01 | relative gain [CI, mine] |
|---|---|---|---|---|
| zavg_BEF − GLS | +0.103 | [+0.033, +0.219] | [+0.035, +0.209] | +42% [+12%, +92%] |
| zavg_BEF − Spearman | +0.199 | [+0.098, +0.328] | [+0.099, +0.321] | +136% [+48%, +259%] |
| zavg_BEF − Pearson | +0.211 | [+0.105, +0.346] | [+0.103, +0.338] | +157% [+58%, +291%] |
| B − GLS | +0.075 | [+0.008, +0.187] | [+0.004, +0.174] | +31% |
| zavg_BEF − B | +0.028 | [+0.018, +0.044] | [+0.018, +0.044] | +9% |

**Verdict recomputed (§3.7, IFT parts).**
- zavg_BEF passes against all three baselines.
- B fails against GLS on macro non-inferiority only: −19.6% relative.
- All six pass/fail outcomes match look01.

## Discrepancies / definition notes
- **Macro per-cluster AP.** My a-priori definition used within-cluster pairs only, and only 16 IFT clusters have a
  within-cluster negative. On that definition the values were 0.70–0.78 for every method, and B would pass against
  GLS (0.744 vs 0.722). look01 uses all pairs touching the cluster, with HT weights, over 56 clusters with ≥1
  positive. That definition reproduces look01 exactly and is now primary in the script; the within-only variant is
  kept as a sensitivity check. DESIGN §3.5 does not specify the definition. So B's failure against GLS depends on the
  definition, while zavg_BEF passes under both. zavg_BEF is −1.1% against GLS on the reproduced definition, within
  the −5% margin.
- **AP including unmapped pairs.** look01 treats every pair that touches an unmapped protein (cluster −1) as
  between-cluster (weight 1/f). If instead cluster −1 is one cluster, values are 0.002–0.005 higher, e.g. zavg_BEF
  0.3373. The ranking is unchanged.
- **Data provenance.** Dev parts (i) and (ii-dev) of §3.7 were not recomputed here; they were verified earlier in
  `zavg_BEF.json`. The run record of look01 shows `git_dirty: true`, with `src/` clean.

## Verdict
**Reproduced.** The IFT pair set, counts, f, all headline and secondary metrics for the five methods, and the §3.7
verdict were reproduced exactly from raw files and independent code. My bootstrap, with a different seed, gives CIs
within ±0.013 of look01's. All three zavg_BEF CIs exclude 0.
