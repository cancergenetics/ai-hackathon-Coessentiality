# Revision round 1: analyses requested in `reviews/manuscript/round1.md`

Date: 2026-10-27. The final method (zavg_BEF) is unchanged. Dev analyses ran through `src/evaluate.py` and `log_trial`. The IFT was touched once, through `scripts/final_eval.py`, as a post-hoc reference (look `2-posthoc`). That look was logged in `results/ift_looks.jsonl` before any score was computed.

| file | content |
|---|---|
| `controls.json` | M9: OR and chromosome controls (`scripts/60_revision_dev.py --stages controls`) |
| `onion.json`, `onion_runs/` | B1: PC-removed Pearson on dev, 5 seeds (`--stages onion`) |
| `ift_posthoc_spec_k20_tinf_std.json` | B1: post-hoc IFT reference (`scripts/final_eval.py --posthoc-reference`) |
| `runtime.json`, `runtime/*.run.json` | M2: raw CSV → scores (`scripts/62_revision_runtime.py`) |
| `bootstrap_coverage.json` | F13: bootstrap coverage simulation (`scripts/63_bootstrap_coverage.py`) |
| `trials_counts.json` | F12: trial-log counts (`scripts/64_revision_trials.py`) |

## M9. OR and same-chromosome controls for the final method

The code is `or_control` and `chrom_control` from `scripts/14_phaseA_compare.py`, imported unchanged.
- Genome-wide scores for zavg_BEF and B come from `src.predict.PairScorer`. The null moments of B, E and F match the manifests exactly (difference 0).
- The B and GLS rows reproduce `results/phaseA/compare.json` to the last digit (`phaseA_reproduction`).

**OR control.** 329 OR genes, 53,956 OR×OR pairs, 12,857 of them on the same chromosome. Under background, about 1% of pairs would lie above the null 99th percentile.

| scorer | > null p99, all | same chromosome | different chromosome | > null p99.9 | mean percentile |
|---|---|---|---|---|---|
| **zavg_BEF** | **0.0842** | **0.2260** | **0.0398** | 0.0336 | 0.608 |
| B | 0.0721 | 0.2001 | 0.0320 | 0.0303 | 0.512 |
| GLS signed | 0.0660 | 0.1608 | 0.0364 | 0.0224 | 0.591 |

zavg_BEF amplifies the OR artefact somewhat more than B. Most of the excess is in same-chromosome pairs. Different-chromosome pairs (4.0%) are also above the 1% expected under background.

**Chromosome control.** Dev pairs, pooled HT-AP (f = dev protein fraction) and the SEEDS[0] fold mean.

| scorer | pooled all | pooled, same-chromosome pairs excluded | excluding same arm | same-chromosome pairs only | fold mean, all / excluding same chromosome |
|---|---|---|---|---|---|
| **zavg_BEF** | **0.445** | **0.455** | 0.455 | 0.316 | 0.456 / 0.468 |
| B | 0.426 | 0.435 | 0.436 | 0.310 | 0.435 / 0.445 |
| GLS signed | 0.272 | 0.278 | 0.278 | 0.203 | 0.288 / 0.293 |

The control passes: excluding same-chromosome pairs raises HT-AP for every scorer.

## B1. PC-removed Pearson ("onion" normalisation) as a reference row

The layers are cached Phase-A layers from `results/layers/manifest.json` (screen_v1):
- `spec_k{5,20}_tinf`: gene-centred, top-k PCs of the line×line second-moment matrix removed, τ = ∞ (no whitening), cosine. This equals Pearson after removing the top-k PCs.
- `spec_k{5,20}_tinf_std`: the same after gene standardisation.

Each layer was run through dev cluster CV with 5 seeds, giving 20 logged trials. The table shows 5-seed means. HT-AP, macro and no-largest are fold means; pooled and no-top10 use pooled OOF, as in Phase B v2. zavg_BEF and GLS are read from `results/phaseB_v2/runs`.

| scorer | HT-AP | macro-cluster AP | AP without largest component | pooled-OOF HT-AP | no-top-10 AP |
|---|---|---|---|---|---|
| spec_k5_tinf | 0.334 | 0.325 | 0.358 | 0.322 | 0.285 |
| spec_k20_tinf | 0.358 | 0.328 | 0.366 | 0.346 | 0.302 |
| spec_k5_tinf_std | 0.317 | 0.309 | 0.343 | 0.303 | 0.263 |
| **spec_k20_tinf_std** (best) | **0.363** | 0.342 | 0.382 | 0.355 | 0.310 |
| zavg_BEF (final) | 0.470 | 0.395 | 0.458 | 0.464 | 0.360 |
| GLS signed | 0.312 | 0.349 | 0.314 | 0.281 | 0.296 |

**Δ HT-AP vs zavg_BEF.** Layer minus zavg_BEF on pooled OOF, paired cluster bootstrap with 1,000 resamples, seeds 0–4. The 95% CI excludes 0 in 5/5 seeds for all four layers.

| layer | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| spec_k5_tinf | −0.152 [−0.257, −0.074] | −0.162 [−0.268, −0.084] | −0.155 [−0.205, −0.088] | −0.110 [−0.173, −0.051] | −0.130 [−0.177, −0.073] |
| spec_k20_tinf | −0.118 [−0.175, −0.071] | −0.127 [−0.183, −0.080] | −0.124 [−0.153, −0.083] | −0.104 [−0.154, −0.053] | −0.115 [−0.153, −0.071] |
| spec_k5_tinf_std | −0.168 [−0.281, −0.082] | −0.180 [−0.294, −0.092] | −0.179 [−0.240, −0.094] | −0.120 [−0.187, −0.050] | −0.158 [−0.220, −0.088] |
| spec_k20_tinf_std | −0.118 [−0.201, −0.055] | −0.127 [−0.211, −0.068] | −0.116 [−0.158, −0.068] | −0.084 [−0.129, −0.040] | −0.098 [−0.133, −0.056] |

**Δ vs GLS** (`delta_vs_gls`):
- k20 layers: +0.056 to +0.092. The CI excludes 0 in 4/5 seeds for spec_k20_tinf and in 2/5 for spec_k20_tinf_std.
- k5 layers: CIs include 0 in every seed.

The macro-cluster AP of every PC-removed layer is below GLS (0.309–0.342 vs 0.349).

**IFT post-hoc reference (look `2-posthoc`).** Before scoring, this record was appended to `results/ift_looks.jsonl`: `{"look": "2-posthoc", "date": "2026-10-27", "what": "post-hoc reference baseline (PC-removed Pearson) requested in manuscript review; not a candidate, no selection of the final method", "reference": "spec_k20_tinf_std"}`.
- Only the dev-best layer, spec_k20_tinf_std, was evaluated.
- It was evaluated once. The mode refuses to run a second time.
- There is no verdict logic.

| IFT | HT-AP [95% CI] | macro-cluster AP | AP without largest component |
|---|---|---|---|
| spec_k20_tinf_std | 0.267 [0.157, 0.486] | 0.320 | 0.360 |
| zavg_BEF | 0.345 [0.197, 0.590] | 0.365 | 0.470 |

- **Δ(reference − zavg_BEF)** = −0.078 [−0.151, −0.028], relative −22.7% [−34.6%, −10.2%] (paired cluster bootstrap, 1,000 resamples, SEEDS[0]).
- **Consistency check:** zavg_BEF's IFT HT-AP equals look 1 exactly (difference 0.0).
- **Context from look 1:** GLS 0.242, Spearman 0.146, Pearson 0.134. PC-removed Pearson is therefore the strongest reference after B, but it stays clearly below zavg_BEF.

## M2. Raw CSV → scores runtimes (`scripts/predict_pairs.py`)

**Setup**
- **Input:** the union of both train pair ID files, IDs only, shuffled with seed 0. That is 634,779 pairs, of which 596,597 mapped; SHA-256 is in `runtime.json`. The file itself is not committed because it is HuMAP-derived.
- **Runs:** one fresh process per method, run sequentially in the same machine state. The raw CSV was page-cache warmed first.
- **Machine:** 8-core arm64 laptop with desktop apps open; load average was 3–5 during the runs.
- **Timing:** wall time is the process wall clock including interpreter start; RSS is the `/usr/bin/time -l` maximum resident set size.

| method | wall s (process) | tool total s | matrix load + hash s | scoring by component s | peak RSS GB |
|---|---|---|---|---|---|
| `pearson` | 10.9 | 8.8 | 3.4 | P 2.99 | 2.29 |
| `spearman` | 10.7 | 8.6 | 3.4 | S 2.95 | 2.34 |
| `E` alone | 124.5 | 122.5 | 3.4 | E 116.7 | 2.79 |
| `F` alone | 14.7 | 12.3 | 3.7 | F 5.89 | 2.70 |
| `zavg:B,E,F` (final) | 118.0 | 116.0 | 4.1 | F 5.3, B 18.6, E 85.8 | 2.99 |
| `gls` | 18.6 | 16.7 | 2.9 | G 11.7 | 2.55 |

**Comparison with the earlier records in `results/predict/`:** zavg_BEF took 136.9 s with 4.06 GB, and GLS 24.2 s.

**Run-to-run variability is about ±20%.**
- E is the dominant and most variable cost: 86–117 s here, 103 s before.
- E alone costs more than E inside zavg, even though both compute the genome-wide graph and the null.

Quote ranges or round values ("~2 min for zavg_BEF; ~10 s for Pearson/Spearman; ~20 s for GLS").

## F13. Coverage of the paired cluster bootstrap

**Model.** `scripts/63_bootstrap_coverage.py` builds a toy version of the pooled-OOF dev design:
- 190 clusters with heavy-tailed sizes (2–30), 5 folds;
- between-cluster pairs only within a fold, with HT weight 5; within-cluster weight 1;
- about 58k pairs and about 1,070 positives per dataset;
- protein effects, a cluster signal, and a method gain that is larger in big clusters.

**Truth.** The population HT-AP, computed on the concatenation of 400 independent datasets (23.2M pairs).

**Coverage run.** 400 simulated datasets, each analysed with `metrics.cluster_bootstrap` exactly as in the project: 1,000 resamples, `base_w` = HT weights, percentile 95% CI.

| scenario (true AP_A / AP_B) | coverage, Δ (MC SE) | coverage, AP_A | coverage, AP_B | Δ bias | mean CI width, Δ |
|---|---|---|---|---|---|
| μ = 2.5 (0.290 / 0.460; like the real dev APs) | **0.948** (±0.011) | 0.970 | 0.953 | −0.003 | 0.119 |
| μ = 1.5 (0.066 / 0.164) | **0.928** (±0.013) | 0.950 | 0.935 | +0.002 | 0.101 |

Coverage is close to nominal 95%. The misses for Δ are one-sided: the whole CI lies below the true Δ in 4.5–6.3% of datasets and above it in 0.75–1.0%. The CI therefore errs towards understating a true gain rather than overstating it. The μ = 2.5 scenario was added after the first run (μ = 1.5) showed toy APs far below the real ones; both scenarios are reported.

## F12. Trial log counts (`results/trials.jsonl`)

The file has 1,690 lines: 845 start records, 844 end records and 1 free-text note. The note records 2 unlogged timing smoke runs from before logging was made unconditional.

| subset | start records | end records | end status `ok` | end status `ok_not_written` | end status `error` | starts without an end |
|---|---|---|---|---|---|---|
| before this revision | 823 | 822 | 464 | 358 | 0 | 1 (`L1line_cos_cluster_seed0`, aborted run) |
| this revision | 22 | 22 | 22 | 0 | 0 | 0 |
| full log | 845 | 844 | 486 | 358 | 0 | 1 |

- **What one trial is:** one evaluation call, i.e. a grid point × seed, or one bootstrap/control/nested-selection step.
- **`ok_not_written`:** the trial was evaluated and logged, but no result JSON was written. Examples are nested-selection steps and screen probes.
- **Universe:** every trial is on the dev universe.
- **IFT accesses:** they are not in this file. They are in `results/ift_looks.jsonl`: looks 0, 1 (started/done), 1-verify, 1-figure and 2-posthoc.

## Code changes

**`scripts/final_eval.py`:**
- New `--posthoc-reference NAME` mode (`posthoc_reference`, `posthoc_scorer`).
- NAME is restricted to the 4 PC-removed Pearson layers above.
- The mode writes the look record before scoring, computes `metric_suite` for NAME and zavg_BEF plus the bootstrap Δ, and has no verdict logic.
- It refuses to run a second time.
- `next_look` ignores string looks; this fix was already in the working tree and is committed here.

**Tests:** a toy test was added to `tests/test_final_eval.py`. `pytest tests/test_final_eval.py tests/test_process.py` gives 25 passed, including the AST test that only `final_eval.py` imports the IFT loader.
