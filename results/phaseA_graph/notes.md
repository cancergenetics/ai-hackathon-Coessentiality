## Trends (written after the screen; HT-AP fold mean, SEEDS[0]; current best rowz[socos_q0.25] = 0.435)

- **The graph construction barely matters for second-order cosine.** With row-z, kNN symmetrisation (dir / union /
  mutual / avg) gives 0.435 / 0.435 / 0.435 / 0.437. Binary weights hurt (0.380), and so do soft-threshold powers of the
  balanced z (p4 0.399, p8 0.352 after row-z; p4 helps only without row-z, 0.419 vs 0.401). Increasing K hurts
  (union q0.5 0.429, mutual q1 0.422). The current best (linear weights, top-0.25%, row-z) is a local optimum
  within this family.
- **TOM / WGCNA does not beat cosine.** kNN TOM gives 0.32–0.43 (binary weights worst). Dense soft-threshold (β = 6
  on scaled cosine) gives TOM 0.395 / 0.422 (row-z) and cosine 0.419 / 0.420.
- **Diffusion: short-range smoothing gives a small, consistent gain; longer walks over-smooth.** Row-z cosine of
  diffusion profiles: heat t=1 0.442, heat t=3 0.441, PPR α=0.5 0.441, 1-step degree-normalised 0.432, N² 0.427,
  N³ 0.347. Raw kernel values (N²_ab, PPR_ab) are worse than profile cosines (0.37–0.42). Pooled-OOF Δ vs the
  current best is +0.007 to +0.017 (CI excludes 0 in seeds 2–4, lower bound −0.003 / −0.001 in seeds 0 / 1). Mean
  over 5 seeds is 0.460 vs 0.448.
- **Laplacian eigenmaps are best on the fold mean but not clearly better on pooled OOF.** Cosine of the top-d
  non-trivial eigenvectors of the normalised union kNN graph: d = 32 / 128 / 512 / 1024 → 0.321 / 0.429 / 0.460 /
  0.451, so the optimum is near d ≈ 512 (round 2 resolved the edge). Diffusion-map weighting λ^4 hurts (0.373),
  row-z does nothing (0.458), and a denser graph (q0.5, 0.455) or k = 0 base (0.451) give about the same. Mean over
  5 seeds is 0.463 (row-z 0.464) vs 0.448. Its profile differs from the current best: macro per-cluster AP 0.394
  vs 0.341, AUROC 0.791 vs 0.769, both non-essential 0.330 vs 0.302. But pooled-OOF Δ vs the current best is
  +0.028 / +0.030 / +0.001 / +0.004 / −0.004 (seeds 0–4), with every CI including 0. The fold-mean gain is mostly
  per-fold calibration, not a better global ranking.
- **Base sensitivity:** k = 0 or 5 PCs instead of 20 costs 0.005–0.01 in every family (flat, as in screen_v1).
- **Selection optimism:** nested (argmax over 4 folds, scored on the 5th) gives 0.452 over all 67 layers, and also
  0.452 within the embedding family (11 layers), vs 0.460 in-sample (optimism 0.009). Every outer fold selected
  lap512 (or its row-z). Nested held-out beats the current best in 5/5 folds (+0.004 to +0.033), but the current
  best's 0.435 is itself un-nested from earlier screens.
- **Runtime** (fresh process, all 596,597 pairs): lap512 94 s (eigsh 513 eigenpairs ≈ 80 s; 3.5 GB RSS), row-z heat1
  59 s (5.1 GB), vs current best 23 s and GLS 7.5 s. The 18k dense eigh needed for d > 1024 is impractical (26 min).
- **Recommendation:** no graph layer replaces the current best on the pre-registered pooled-OOF criterion.
  row-z cos_heat1 is the only consistent (small, +0.01) improvement. emb_lap512 has the highest fold means and
  better macro-cluster AP but is not significantly better pooled. It is a different view (global spectral
  coordinates vs local neighbourhood overlap), so it is the natural candidate to add to an A6 z-average or as a
  Phase-B feature, not a new primary.
- **Deviations / disclosure:** graph_v2 (7 columns) was added after seeing graph_v1 (embedding dimension at grid
  edge). All runs used `--allow-dirty` because another agent's untracked files were in src/ (src/metric_learning.py).
  src/layers_graph.py was committed before every run (commits 3bd38a6 and the eigsh-dimension follow-up).
