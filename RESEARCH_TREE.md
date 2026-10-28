# RESEARCH_TREE.md

Legend: [ ] planned · [~] in progress · [x] done · [-] abandoned/negative · [!] key finding

## Status summary (2026-10-27, M5: project complete)
Final method zavg_BEF — label-free equal-weight z-average of three views of gene-standardised, top-20-PC-filtered, ridge-whitened CRISPRGeneEffect profiles (F first-order correlation; B second-order neighbourhood cosine + genome-wide row-z; E 512-d Laplacian eigenmap of the genome-wide kNN graph). Dev (cluster-grouped CV, 5 seeds) HT-AP 0.470 vs GLS 0.312 / Spearman 0.282 / Pearson 0.260. Single pre-registered IFT look: 0.345 vs GLS 0.242 (+42%, Δ CI [+0.035,+0.209]), Spearman 0.146, Pearson 0.134; clearly-beats criterion met vs all three; reproduced exactly by independent code. Runtime 137 s raw→scores for 635k pairs.

Supervision (stacking, gene features, metric learning) never beat the label-free geometry. Main caveats: gain concentrated in large dense complexes (IFT macro AP ≈ GLS), small IFT (n_eff≈8), D1 contamination disclosed, OR same-chromosome artefact slightly amplified. REPORT.md written; withheld test to be scored by the user via scripts/predict_pairs.py --method zavg:B,E,F.

## Decision notes
- D0 (M0): Evaluate at natural prevalence (all negatives kept in eval folds). Primary metric AUPRC (average precision). Do not commit HuMAP/DepMap-derived pair lists; commit scripts + seeds + checksums of generated splits.

- D1 (R1): Algorithms and data reviewers evaluated label-free variants (shrunk-Σ whitening ~0.30 vs GLS 0.255 AP) on ALL scorable pairs before IFT existed. No selection made from it; IFT confirmation remains, but whitening-family prior is now IFT-informed. Disclosed in REPORT.

- D2 (v3.1): IFT power too low for CI gate (n_eff 7.7 clusters; P(pass)≤0.22 at +20%). CI gate moved to pooled dev OOF; IFT = point-estimate ≥10% gate. Decided before any IFT look.

- D3 (pre-registered before Phase B v2): IFT candidates chosen by dev-only rule. C_single = rowz[socos_q0.25[spec_k20_t0.1_std]]. C_comb = best combination from Phase B v2 (by 5-seed mean of nested held-out HT-AP). Primary = C_comb iff its pooled-OOF Δ vs C_single has CI excluding 0 in ≥3/5 seeds AND macro-cluster AP not lower; else primary = C_single and C_comb is secondary. Fixed-sequence: primary tested first.

- D4 (after IFT look #1): Final method = zavg_BEF (label-free; no fitted parameters, so 'refit on all train pairs' is a no-op; hyper-parameters frozen). Further method changes would be exploratory only (no further IFT looks planned).

## Tree
- [x] M0 Setup: env, PROMPT.md, DATA.md, profile (`results/profile/data_profile.json`)
- [x] M1 Literature review → LITERATURE.md (+ errata from R1 prior-work review)
- [x] M1 Infrastructure (commit 9bc81a2): mapping 1,114/1,151 IDs (2 ambiguous histones dropped); 3,191 pos / 593,406 neg scorable; 28 tests pass
  - [x] Baselines, all scorable pairs (prev 0.0053): Pearson r 0.202, Spearman ρ 0.222, GLS signed t 0.255 [0.238,0.272], GLS two-sided 0.241; GLS+OR correction worse (0.234). Signed > absolute everywhere. Runtimes 0.4–6 s. Σ rank-deficient (1357/1361) → 4 samples dropped via pivoted QR.
- [x] M2 DESIGN.md + up to 3 review rounds
  - [x] R1: 6 reviewers (reviews/round1/*) → DESIGN v2
  - [x] R2: 2 reviewers → DESIGN v3 (HT-weighted AP, fixed-sequence IFT testing, label-free ensemble default primary); round 3 skipped (refinements only)
  - [x] v3.1 amendment after power check (D2)
- [x] M2b Evaluation infra v2 (commits 1155d55, 47dd58b; 118 tests): 247 clusters (giant comp → 7 Louvain communities), IFT 847 pos/40,786 neg (57 clusters), dev 2,299 pos/320,229 neg (190 clusters)
  - [x] Dev-CV baselines (seed 0, HT-AP mean±sd over folds): Pearson r 0.238±0.095, Spearman ρ 0.260±0.095, GLS signed 0.288±0.098 (5 seeds: 0.260/0.282/0.312). OR-PC removal hurts GLS.
- [x] M3 Method development
  - [x] Phase A layer cache: 128 label-free layers (commits 920b8c0, 6b2f6e3), 39 tests
  - [!] Phase A screen (commits 34be32c, 7bd4c4b; results/phaseA/): best = rowz[socos_q0.25[spec_k20_t0.1_std]] — gene-standardised Σ, top-20 PC removal + ridge 0.1 whitening → genome-wide top-0.25% neighbour cosine (second-order) → genome-wide row-z. Dev HT-AP 0.435±0.142 vs GLS 0.288; pooled-OOF ΔAP vs GLS +0.155..+0.179 over 5 seeds, CI lower bound ≥+0.05. Runtime 23 s vs GLS 7.5 s. VERIFIED independently (a2f8622, results/verify/): exact reproduction from raw CSV, leakage audit pass, gene-shuffle control 0.010 ≈ prevalence. CAVEAT: gain carried by large dense complexes — without top-10 clusters (64% of OOF positives) Δ=+0.039 [−0.013,+0.087]; macro per-cluster AP 0.326 vs GLS 0.325.
    - [x] Trends: gene standardisation before Σ +0.06; flat in τ, k≤20; second-order +0.05–0.07; row-z of second-order +0.03.
    - [-] Label-free z-average ensembles: no gain in nested check (0.428 vs 0.428).
    - [!] OR control partial fail: 7.2% OR×OR pairs above null 99th pct (GLS 6.6%), driven by same-chromosome OR clusters. Chromosome-proximity control passes.
  - [x] Phase B v1 (results/phaseB/): LGBM over 12 a-priori layers 0.383 vs label-free z-mean of same 0.357 (Δ +0.006, CI incl 0). Gene-stat-only 0.024 (≈2.5× prev), chromosome-only 0.009 → no protein memorisation. Held-fold shuffle passes; all-label shuffle with monotone constraints 0.145 (constraint encodes 'similar=likely'; unconstrained → prevalence) — recorded finding. Supervision ≈ no gain so far.
  - [x] Reproducibility bug: cluster assignment depended on PYTHONHASHSEED (Louvain tie-breaks); canonical = PYTHONHASHSEED=0 = hash used by all results; enforced + CSV committed.
  - [-] Phase C supervised metric learning (commits 952b265, 47cd366, 26623ab; results/phaseC/): L1 cell-line weights, L2 low-rank (BCE/InfoNCE), L3 second-order on learned space — all within ±0.003 of initialisation; training memorises genes, inner-val AP flat. Side finding: outside top-10 clusters first-order whitened base ≈ best (+0.004) with higher macro AP (0.371 vs 0.341).
  - [!] Phase B v2 (e242a7c; results/phaseB_v2/): zavg_BEF (equal-weight null-z mean of B, E=emb_lap512, F=spec_k20_t0.1_std) 5-seed HT-AP 0.470, macro 0.395, no-top10 0.360 vs B 0.448/0.351/0.332, GLS 0.312/0.349/0.296. Δ vs B +0.014..+0.021 (CI>0 5/5 seeds); Δ vs GLS +0.175..+0.193. Nested 0.468. LR/LGBM stackers ≤ z-averages; gene features no help. D3 → PRIMARY zavg_BEF, SECONDARY B. Runtime 125 s (all pairs).
  - [x] Final predictor (b8fdf0f..e724ac9; results/predict/): scripts/predict_pairs.py, raw CSV → scores, matches cache, invariance tests; single 28 s, zavg:B,E,F 137 s on 635k pairs.
  - [x] Independent verification of zavg_BEF (460fd36; results/verify/): reproduced to 2.6e-5 per fold with independent E re-implementation; leakage audit pass; gene-shuffle control at prevalence; predictor output identical. Fragility: +0.02 over B stable for E d∈{256,1024}, q=0.5 (24/25 folds), but 'CI>0' holds in 2–5/5 seeds depending on E config → Δ vs B is modest; Δ vs baselines robust.
  - [x] final_eval aligned to v3.1 (ff0dd52; 197 tests)
- [!] M4 IFT look #1 (results/ift/look01_2026-10-27.json): 847 pos / 40,786 neg, 57 clusters. HT-AP zavg_BEF 0.345, B 0.317, GLS 0.242, Spearman 0.146, Pearson 0.134. zavg_BEF−GLS +0.103 [+0.035,+0.209] (+42%), −Spearman +0.199 [+0.099,+0.321], −Pearson +0.211 [+0.103,+0.338]. Macro-cluster AP 0.365 vs GLS 0.369 (non-inferior). PRIMARY CONFIRMED: clearly beats all three baselines. B fails vs GLS on macro non-inferiority (0.2965). zavg_BEF−B +0.028 [+0.018,+0.044].
  - [x] A7 graph family (commits 3bd38a6, c79e37f, ea7c7f5; results/phaseA_graph/): 69 layers. emb_lap512 (Laplacian eigenmap of genome-wide union-kNN graph) HT-AP 0.463 (5-seed mean) vs current best 0.448, nested 0.452; pooled-OOF Δ vs best +0.028..−0.004, CIs incl 0 → not a replacement. Macro-cluster AP 0.394 vs 0.341 (better on small complexes). rowz[cos_heat1] diffusion +0.007..+0.017 (CI excl 0 in 3/5 seeds), 59 s. TOM/soft-threshold/N³ diffusion no gain. Runtimes: lap512 94 s, best 23 s.
- [x] M4 Independent verification (a2f8622, 460fd36, b914a8e)
- [x] M5 REPORT.md, report tables (8abe43c), final retro, LESSONS pruned
- [x] M6 Manuscript (PLOS Comp Biol format, manuscript/): manuscript.md/.docx, S1 (prompt), S2 (methods that did not work), S3 (agent log), Fig1–5; two independent review rounds (reviews/manuscript/round{1,2}.md); revision analyses in results/revision/ (controls on zavg_BEF, post-hoc PC-removed-Pearson reference = IFT look "2-posthoc", runtimes, bootstrap coverage 94.8%). Public MVP repo at release/ai-hackathon-Coessentiality/ → github.com/cancergenetics/ai-hackathon-Coessentiality (gitignored here; uncommitted, published by the user). No dataset files, no licence file.
