# Does restricting to Avana-screened cell lines help the final method?

Date: 2026-10-29. A post-hoc check requested after the report, the manuscript and the single IFT look. Dev only: no IFT pair was loaded. The final method (zavg_BEF) is unchanged; nothing here was used to choose anything.

**Why.** CRISPR library effects are gene-specific (Rauscher 2018; Dempster 2019), and the classic co-essentiality papers used Avana-only data. The round-1 data review (`reviews/round1/data_biology.md`) tested library restriction only with Pearson, in a single pass, and recommended keeping all cell lines. This repeats the question with the final method and the standard dev protocol.

**What was run** (`scripts/65_library_subsets.py`):
- Three row subsets of `CRISPRGeneEffect` 26Q3. Library flags come from the CRISPRConfounders one-hot (`src.data.model_metadata`).
  - `all`: all 1,361 models (the published input).
  - `avana_flag`: 1,048 models screened with Avana, including 221 also screened with KY, Humagne or another library.
  - `avana_only`: 827 models screened with Avana and nothing else.
- On each subset, zavg_BEF (ported implementation, frozen config) and signed GLS were recomputed from scratch. Background genes, covariance, neighbour sets (K = 44 in every subset), graph and null were all re-estimated.
- The rebuild on `all` reproduces the published cached zavg_BEF to 9e-6.
- 648 genes are not in the Avana library, so they have no data in `avana_only`. 9 of them are in dev pairs: 7,224 pairs, 77 of 2,299 positives (3.3%). Their pairs get the subset's minimum score.
- Two evaluations:
  - **native**: each input scores what it can, so it includes the cost of losing those genes.
  - **matched**: the 7,224 pairs get the minimum score in every input, so all three inputs are compared on the same scorable pairs.
- Each (input, scorer, evaluation) ran `run_dev_cv` over 5 seeds (cluster scheme, logged in `results/trials.jsonl`). Differences come from a paired cluster bootstrap (1,000 replicates) of pooled dev HT-AP.

## Results

Dev HT-AP, mean over 5 seeds of the 5-fold mean (SD over seeds about 0.015–0.02):

| Input | zavg_BEF native | zavg_BEF matched | GLS native | GLS matched |
|---|---|---|---|---|
| All 1,361 lines | **0.470** | 0.459 | 0.312 | 0.305 |
| Avana-flagged 1,048 | 0.466 | 0.457 | 0.320 | 0.316 |
| Avana-only 827 | 0.422 | 0.422 | 0.286 | 0.286 |

Pooled dev HT-AP differences against all lines (95% cluster-bootstrap CI):

| Comparison | native | matched |
|---|---|---|
| zavg_BEF, Avana-flagged − all | −0.003 [−0.020, +0.016] | +0.000 [−0.017, +0.017] |
| zavg_BEF, Avana-only − all | **−0.049 [−0.073, −0.024]** | **−0.039 [−0.063, −0.018]** |
| GLS, Avana-flagged − all | +0.007 [−0.009, +0.025] | +0.010 [−0.004, +0.028] |
| GLS, Avana-only − all | **−0.026 [−0.044, −0.009]** | **−0.019 [−0.033, −0.006]** |

The gain of zavg_BEF over GLS holds on every input: +0.173 (all), +0.164 (Avana-flagged), +0.149 (Avana-only). All CIs exclude 0.

Per-cluster (macro) AP of zavg_BEF: 0.395 for all lines, 0.409 for Avana-flagged, 0.362 for Avana-only.

## Interpretation

- **Avana-flagged lines tie with all lines.** Dropping the 313 non-Avana lines neither helps nor hurts the final method (−0.003, CI ±0.02). GLS gains slightly (+0.007 to +0.010), but the CI includes 0. Macro AP is slightly higher (0.409 vs 0.395), with no CI computed. The joint Chronos correction plus component removal and whitening appears to absorb most library structure.
- **Avana-only lines are clearly worse**, for both methods. Most of the loss is not the 9 unscorable genes: on matched pairs the loss is still −0.039.
- **Cause not identified.** Avana-only has 40% fewer lines than all lines and 21% fewer than Avana-flagged, and it drops the multi-library lines, which are Avana-screened too. This check cannot separate "fewer lines" from "something about the dropped lines". A size-matched random subsample of all lines was not run.
- **Conclusion for the method:** no reason to restrict the input. Keeping all 1,361 lines is at least as good as either Avana subset, and it scores 648 more genes.

**Caveats.**
- Dev only, one post-hoc check.
- The same label-free scores serve all five seeds, so the SD over seeds reflects split variation only.
- The pooled bootstrap CIs are the better guide to uncertainty.

## Files

| file | content |
|---|---|
| `summary.json` | subset info, 5-seed summaries, bootstrap (native and matched) |
| `runs/*.json` | one `run_dev_cv` result per input × scorer × evaluation × seed (60) |

Dev scores per subset are cached in `data_cache/library_subsets/` (not committed).
