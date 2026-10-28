# Retro M3a (fresh-process review; after Phase A, Phase B v1, verification)

## 1. What should we stop doing?

1. **Fixing statistical gates before a power check.** DESIGN v3 specified an IFT CI gate; only afterwards did a power simulation show n_eff 7.7 clusters and P(pass) <= 0.22 at +20%, forcing the v3.1 amendment (D2, commits 047dd58 -> d94db6f). Run feasibility/power before freezing gates.
2. **Running downstream phases on unfrozen upstream choices.** Phase B v1 (commit b3480c2) used 12 "a-priori provisional" layers, not the Phase A best (0.435). Its headline "supervision ~ no gain" (LGBM 0.383 vs z-mean 0.357) does not transfer to the real best layer and must be rerun.
3. **Headlining a seed-0 dev number from a 148-layer screen.** Fold sd is 0.14 against a 0.435 mean; GLS itself moves 0.260/0.282/0.312 across seeds. The selected config is optimistic by construction (verify README caveat 1). Multi-seed numbers came late.
4. **Reporting aggregate gains without concentration breakdowns.** Only the verifier showed that 64% of OOF positives sit in 10 clusters and that without them delta is +0.039 [-0.013, +0.087] with macro AP level (0.326 vs 0.325). The macro / no-top-cluster columns existed but were not in the headline.
5. **Verbose research tree.** RESEARCH_TREE.md bullet lines now carry full metric strings and trend lists duplicating phaseA/README; it is the memory file and should be short pointers.

## 2. What worked well?

- **Independent re-implementation** (results/verify): exact reproduction from raw CSV without importing src layers/metrics, leakage audit, and it surfaced the concentration caveat. Best-value step of the project.
- **Negative controls with consequences**: gene-shuffle 0.010 ~ prevalence 0.009; all-label shuffle with monotone constraint scoring 0.145 exposed that the constraint itself encodes "similar = likely"; OR/chromosome controls partially failed and were recorded rather than hidden.
- **Pre-committed decisions before looks**: D2 was decided before any IFT look; ift_looks.jsonl records the single contaminated look; IFT never touched.
- **Cheap label-free layer cache** (128+ layers x 596k pairs, manifest) made a 148-layer screen with 1,090 logged trials (trials.jsonl) tractable at 23 s per scorer.
- **Leakage-aware design reviewed early**: cluster-grouped CV + HT-weighted AP came from review rounds; the gene-only (0.024) and chromosome-only (0.009) probes show no protein memorisation.
- **Tests caught a real bug**: PYTHONHASHSEED-dependent Louvain clusters, found by a failing test, fixed by committing the canonical assignment.

## 3. Proposed LESSONS.md rules

1. Run a power/feasibility simulation before freezing any statistical gate in DESIGN. Evidence: D2, IFT n_eff 7.7, gate moved after v3.
2. Do not build on a layer/config until Phase screening freezes it; if you must, label it provisional and budget a rerun. Evidence: Phase B prov12 vs Phase A best.
3. Every headline delta must ship with macro per-cluster AP and delta without the top-10 clusters. Evidence: verify found +0.155 -> +0.039 [-0.013, +0.087].
4. Report >= 5 seeds for any selected-config claim; treat seed-0 screen winners as optimistic. Evidence: GLS 0.260-0.312 across seeds; fold sd 0.14.
5. Verifier re-implements from raw inputs and imports nothing from the producer except the split. Evidence: exact reproduction at 0.426/0.272 plus shuffle control and cluster audit.

## 4. Prune candidates in current LESSONS.md

- #1 (Apple Silicon/Rosetta): one-off environment fix, already solved; move to DATA/env notes.
- #2 (profile raw inputs): generic good practice, already implied by PROMPT.
- #3 (tell reviewers not to compute metrics): keep only merged into the "reviewers get no label files" rule.
- #4 + #5: merge into one rule ("fixed PYTHONHASHSEED, commit canonical split file, test in subprocess with different hash seed").
