# Retro final (fresh-process review; since M3a: A7, Phase C, B v2, predictor, IFT look #1, verifications)

## 1. What should we stop doing?

1. **Full phases on a family that already returned null.** B v1 (LGBM vs z-mean +0.006, CI incl. 0), Phase C (all rungs within +-0.003 of init; training memorises genes) and B v2 (LR/LGBM 0.462/0.466 vs zavg 0.470) gave the same answer three times. Pilot on one fold vs init; stop at +-0.005.
2. **Letting the memory file go stale.** The RESEARCH_TREE header still says "M3a ... Now running A7 and Phase C"; "[ ] M4 Independent verification" is unticked despite 460fd36 and b914a8e. The tree is still ~1,200 words of metric strings.
3. **Retrofitting required columns.** LESSON 6 demands no-top-10 Δ per headline, yet tables.md T4 admits those CIs vs baselines "are not in the results" (only Δ vs B). Found at report time.
4. **Reading "CI>0 in 5/5 seeds" as robustness.** Verifier: seed-2 lower bound +0.0005; E d=256 gives 2/5, d=1024 and q=0.5 4/5. The +0.02 point gain is stable, significance is not, and runtime is 125 s vs B's 23 s.
5. **Expecting small-complex gains from A7/C.** Dev macro AP rose (0.395 vs GLS 0.349) but IFT macro is 0.365 vs 0.369; the +42% headline is again large-cluster driven.

## 2. What worked well

- **Pre-registered D3/D4 and one IFT look**; look "1-verify" logged before running; fixed-sequence gate correctly rejected secondary B (macro 0.296 vs 0.369).
- **Independent re-implementation**: zavg_BEF reproduced to 2.6e-5 per fold; E rebuilt with deliberate deviations; IFT verifier used its own bootstrap seed and the predictor as black box, counts exact.
- **M3a lessons adopted**: frozen layers, 5 seeds, macro and no-top-10 columns in T3, nested check (0.468 vs 0.470).
- **Fast committed negatives** (Phase C README, trials.jsonl) and shuffle controls at prevalence (0.009-0.010).
- **`predict_pairs.py`** with invariance tests matches cache and verifier; `40_report_tables.py` reads committed files only, so gaps show as "n/a".
- **Honest dev-to-IFT shrinkage** (zavg 0.470 -> 0.345, GLS 0.312 -> 0.242; still +42%).

## 3. Proposed new LESSONS rules

1. Pilot a new supervised family on one fold vs its init score before a full phase; stop if |Δ| < 0.005. (B v1 +0.006, Phase C ±0.003, B v2 stackers ≤ zavg.)
2. Verifier sweeps the selected config's neighbours and reports seeds-with-CI>0 for each. (E d=256: 2/5 vs claimed 5/5.)
3. Update RESEARCH_TREE header and checkboxes in the same commit as each result; keep ≤600 words of pointers. (Stale M3a header.)
4. Run the report-tables script at each phase end so required columns are computed in compare scripts, not retrofitted. (T4 gap.)

## 4. Pruning (LESSONS ≤10)

Current 9 + 4 = 13. Drop #1 (Apple Silicon; one-off, move to env notes) and #9 (generic delegation advice, no recurrence evidence). Merge #6 and #7 (headline Δ ships with macro AP, no-top-10 Δ, ≥5 seeds, nested check). Result: 10 rules.
