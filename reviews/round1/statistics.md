# Round 1 review: statistics (DESIGN.md v1)

Numbers come from running `src/splits.make_split` on the scorable pairs in `data_cache/pairs.pkl` (scratch script, nothing committed).

## Measured facts that drive the review
- Scorable pairs: 3,191 pos / 593,406 neg (prevalence 0.53%). There are 241 positive-graph components. The largest has **848 positives (26.6%)**, and the next ones have 325, 262, 179, 153 and 121. Kish effective number of components weighted by positives: **≈10** (≈20 without the largest). Pooled AUPRC is driven by ~10 complexes.
- Complex 5-fold, seed 0: test prevalence is **1.8–2.3% in four folds and 9.0%** in the fold holding the largest component. Within-component ("hard") negatives make up ~7% of tested negatives, against 1.4% in the full universe. Only 20% of pairs are ever tested.
- The greedy largest-first splitter gives **identical fold-level positive counts for seeds 0 and 1**. The big components are always grouped the same way. Seed-to-seed ARI (0.21) comes only from small components. The planned "5 repeats × 5 folds" is therefore close to one partition repeated.
- Gene (C3) split: natural prevalence, but only ~650 positives are ever tested (~130 per fold).
- `metrics.py` bootstraps **pairs i.i.d. (stratified)**, not components as DESIGN §3.5 says. With n_eff ≈ 10, the resulting CIs will be far too narrow.

## Proposed changes

**P1-1 Implement the component (cluster) bootstrap correctly.** Draw component multiplicities m_c. Give a within-component pair weight m_c and a between-component pair weight m_c1·m_c2 (Owen's pigeonhole/dyadic bootstrap), and feed these weights to `PresortedScores.ap(w)`. For CV, resample within each fold and keep the component draw identical across methods and across repeats (paired). Report SD, percentile CI and leave-largest-out estimate. *Why:* pairs share proteins and complexes; i.i.d. resampling understates variance several-fold.

**P1-2 Standardise the composition of each fold before averaging AP.** Raw fold APs are comparable neither across folds (2% vs 9%) nor with the withheld test (~0.5%). In fold k, give between-component negatives the weight w_k = (B_dev/P_dev)·P_k/B_k, where B is the number of between-component negatives and P the number of positives. Keep weight 1 on positives and within-component negatives. (Horvitz–Thompson: untested cross-fold negatives are exchangeable with tested ones under random component assignment.) **Primary metric = unweighted mean over folds (and repeats) of standardised AP.** Also report raw AP and prevalence per fold. Pool over folds only for label-free scores; for supervised scores pooling mixes fold-model scales. *Robustness (final comparisons only):* score cross-fold pairs (i,j) with a model trained on folds other than i and j (15 fits per repeat)., giving the full dev universe at natural composition.

**P1-3 Make the repeats real.** Assign components to folds in a seeded random order, or draw random assignments and accept the first within a balance tolerance. Pin the largest component; randomise the rest; record ARI between repeats.

**P1-4 Separate selection from confirmation.** Dev-CV Δ for the winner is biased upward by selection over A1–D and hyper-parameters. It cannot serve as evidence for "clearly beats" (DESIGN §3.5(ii) currently requires it). Proposed rules:
- Make IFT the only confirmatory test.
- Pre-declare at most 2 final candidates.
- Apply Holm across candidates. Across the three baselines the claim is an intersection–union test, so each baseline is tested at α = 0.05 with no further correction; state this in DESIGN.
- Log configurations evaluated per family; give baselines a comparable, logged sweep (NaN handling, sign, OR removal).

**P1-5 Check the power of the internal final test before building any method.** IFT covers 20% of positives with the largest component excluded, so its n_eff is probably about 4–6. Before method development: report the IFT partition's n_eff, prevalence and hard-negative share; estimate the component-bootstrap SD of ΔAP on dev for GLS vs Pearson; if power for a 20% relative gain is below 50%, enlarge IFT to 30% of positives now.
- Write the IFT decision rule in DESIGN: the one-sided 95% lower bound of the paired component-bootstrap ΔAP (standardised composition) must be > 0 against each baseline.

**P2-6 Tighten "clearly beats".** Criterion (i), higher in at least 4/5 folds, has a null probability of 0.19, so treat it as a consistency check, not inference. Add two conditions, fixed now:
- (a) The gain also holds for macro per-component AP and for AP without the largest component. (guards against a win carried by one giant component).
- (b) A pre-registered practical margin, e.g. relative ΔAP ≥ 10% over the best baseline variant on dev standardised AP.

**P2-7 Unmapped and partially scorable pairs.** There are 123 pos / 38,059 neg unmapped (3.9% vs 6.0%); excluding them in dev is fine. For the "all-pairs" number:
- Assign the same minimum-score tie block for every method, so Δ is unaffected to first order. State this.
- More important: original GLS drops genes with any NaN (1,352 genes have NaN). Every method must be evaluated on the **same pair set**, with an identical, pre-specified fallback for pairs a method cannot score. Report coverage per method, and never compare APs computed on different subsets.
- `predict_pairs.py` must apply the same rule to unseen test proteins.

**P2-8 Strengthen the controls.**
- (a) Add a degree-preserving label permutation (edge swaps within training folds). It isolates pair signal from protein propensity better than plain permutation, which also destroys degree.
- (b) Treat the gene-features-only model as a reference, not a near-prevalence null. Report the full model's Δ over it.
- (c) Chromosomal-proximity control. Cutting/copy-number artefacts make neighbours co-essential. Report a chromosome-features-only model and AP after excluding same-arm pairs closer than about 5 Mb.
- (d) Secondary "hard-negative AP": positives against within-component negatives only.

**P3-9 Secondary splits.** The C3 split has about 650 tested positives. Report it with the component bootstrap and do not use it to rank close variants.

**P3-10 Text/code consistency.** DESIGN §3.2 says cross-fold pairs are unused, but `splits.py` trains on pairs spanning two non-held-out folds (correct; update the text). Update the component count (232 → 241 on the scorable subset). Report positives, negatives and n_eff.

## Not needed
AP estimator (sklearn, block ties), 1,000 resamples, and single-look logged IFT are appropriate.
