## Trends (written after the screen; HT-AP = fold mean, SEEDS[0])

- **A1 standardisation dominates the spectral family.** Dividing each gene by its SD before estimating Σ lifts ridge
  whitening from 0.287–0.300 to 0.359–0.365 (τ = 0.1). With standardised Σ the result is flat in τ ∈ [0.01, 0.3]
  (0.359–0.365, LW 0.361) and in k ∈ {0, 5, 20}; without a ridge (τ = inf) standardising does nothing (Σ unused).
  Unstandardised: τ ≈ 1 is best for k ≤ 10 (0.327); GLS-like τ = 0 is worst among whitened layers (0.279; baseline GLS
  with exact NaN marginalisation 0.288); at k = 20 weak/no whitening wins (τ = 10/inf 0.33). Pearson improves
  steadily with top-k PC removal (k 0 → 20: 0.238 → 0.330) and drops at k = 40 (0.309); k = 40 hurts every τ.
  Line-centred GLS r (Wainberg convention) = gene-centred GLS (0.278 vs 0.279).
- **A3 row-z / mutual rank: small.** +0.006 to +0.03 on unstandardised bases (largest on Pearson), ≈ 0 on the
  standardised base (0.365 → 0.366 rowz, 0.363 mrbal). Mutual rank ≈ row-z (±0.01); balanced ≈ raw MR.
- **A4 second-order: the biggest gain.** Weighted cosine ≫ Jaccard (by 0.04–0.11). AP rises as the neighbour set
  shrinks (q 2 → 1 → 0.5 → 0.25: e.g. 0.339 → 0.352 → 0.362 → 0.369 on spec_k20_t0.1); q = 0.1 is better raw
  (0.412) but worse after row-z, so q ≈ 0.25 is the optimum. Gains need a whitened first-order base (+0.05–0.07 on
  τ = 0.1 / LW bases; on raw Pearson second-order is worse than the base, 0.204 vs 0.238).
- **Row-z of second-order (screen_v2) adds another +0.03–0.04**: best single layer
  rowz[socos_q0.25[spec_k20_t0.1_std]] 0.435 ± 0.142 (pooled AP 0.454) vs best baseline GLS 0.288.
- **Profile of the gain.** Second-order layers have larger fold SD (0.14–0.16 vs 0.11), lower HT AUROC (0.75–0.79 vs
  0.79) and lower macro per-cluster AP (0.33–0.34 vs 0.37 for the standardised spectral layers): they sharpen the top
  of the ranking, mostly in larger, densely connected clusters, rather than improving every complex. All improve every
  essentiality stratum (both non-essential 0.26 → 0.30 vs GLS).
- **Selection caveat.** screen_v2 (20 layers) was chosen after seeing screen_v1 (17) and round 1 of screen_v2 (3);
  the top of this table is dev-selected and optimistic.
