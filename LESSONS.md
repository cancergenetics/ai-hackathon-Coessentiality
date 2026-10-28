# LESSONS.md — process rules (rule + evidence). Pruned at M5 (retros: reviews/retro/M3a_retro.md, retro_final.md).

1. Reviewers and non-evaluating agents get no label-metric access: say "compute no AP/AUROC" in every brief. (R1: two reviewers probed all pairs before the IFT existed → contamination D1.)
2. Run graph/community code under fixed PYTHONHASHSEED, test regeneration in a subprocess, and commit small derived split files as the canonical copy. (Louvain tie-breaks made the IFT/dev assignment irreproducible.)
3. Run a power simulation before freezing any statistical gate. (D2: IFT CI gate had P(pass) ≤ 0.22 even at +20%.)
4. Label downstream work on unfrozen upstream configs "provisional" and budget a rerun. (Phase B v1 used 12 a-priori layers, not the Phase A best.)
5. Every headline Δ ships with macro per-cluster AP, Δ without the top-10 clusters, ≥5 seed repeats and a nested-by-fold check. (B: +0.155 overall vs +0.039 n.s. without top-10; greedy-ensemble gain vanished when nested.)
6. Verifiers re-implement from raw data importing nothing from the producer except the split, and sweep neighbouring configs. (Exact reproductions a2f8622/460fd36/b914a8e; "CI>0 in 5/5 seeds" for zavg_BEF−B held in only 2/5 at E d=256.)
7. Pilot a new supervised family on one fold against its initialisation; stop if |Δ| < 0.005. (Phase B v1, Phase C, B v2 stackers all null after full 5-seed runs.)
8. Pre-register the IFT candidate rule (D3) and the exact look command before running it; one look. (Look #1 was clean and independently re-derived.)
9. Update the tree header/checkboxes in the same commit as each result, and regenerate report tables at each phase end. (Tree went stale; T4 no-top-10 CIs vs baselines missing at report time.)
10. torch and LightGBM in one pytest process can deadlock on OpenMP; run those test files in separate processes. (Final suite run.)
