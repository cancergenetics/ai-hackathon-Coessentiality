# Round 1 review: AI-agent failure modes

Read: PROMPT, DESIGN v1, TREE, LESSONS, DATA, src/, scripts/01-02, tests/. P0 = before any method result is trusted; P1 = before Phase B; P2 = optional.

## Defects found in current code/design
1. **Bootstrap contradicts DESIGN 3.5.** `metrics.bootstrap_auprc*` resamples pairs, not components. Pairs share proteins (a complex of n proteins gives ~n^2/2 positives), so CIs are far too narrow and "clearly beats" passes too easily. The baseline README already prints these CIs.
2. **Baseline variants are scored on all pairs**, with no dev/IFT partition yet (`02_run_baselines.py`), while DESIGN 3.6 says "chosen on dev only". After IFT exists these numbers cannot drive variant selection.
3. **Results record no git commit, dirty flag, data checksums or pair-universe hash** (promised in DESIGN 6). Repo has only the M0 commit.
4. **Caches are unvalidated.** `build_matrix_cache` returns if the `.npy` exists; `scores/<tag>.npz`, `pairs.pkl`, `uniprot_map.csv` are keyed by name only, so a changed mapping rule or file is silently ignored.
5. **NaN-to-0 on undefined correlation/GLS** is only logged. In library-restricted runs thousands of pairs may tie at 0 with no trace in the JSON.
6. **No independent test of the UniProt -> HGNC -> Entrez -> column -> `ia/ib` chain.** Toy-matrix tests would pass an off-by-one.

## Risks and safeguards

| Risk | Safeguard | Pri |
|---|---|---|
| Anti-conservative CIs (1) | Implement a node/component cluster bootstrap (resample components; keep pairs with both proteins drawn); test coverage on simulated clustered data; label pair-level CIs "anticonservative" until then. | P0 |
| Propensity leakage (negatives ~ all non-positive pairs) | Pre-set pass rule for control 3.7(b): the full model must beat the gene-stat-only model with a cluster CI excluding 0; report the gene-stat-only AUPRC in every table. Test that no gene-ID-derived feature (one-hot, embedding, lookup) reaches any model. Toy oracle: degree-only scorer high under random splits, ~prevalence under complex splits. | P0 |
| Fit on test data (PCA, imputation, scaling, early stopping, feature selection, mutual rank) | Single `fit(train_idx)` interface for every supervised step. Label-free genome-wide transforms allowed but flagged `transductive`. **Batch-invariance test**: a pair's score alone equals its score in a large file, independent of row order and (a,b) order; this protects the final `predict_pairs.py`. | P0 |
| Label leakage in supervised models | Test: shuffling labels of the held-out fold leaves predictions and AUPRC unchanged; shuffling all training labels gives AUPRC ~ prevalence (3.7a extended to every supervised model). | P0 |
| IFT bleed or tuning on IFT | IFT loader exists only inside `final_eval.py`; model code receives dev pairs only. `final_eval.py` appends to an IFT-look log and refuses a repeat look at the same method hash. Cap at 3 pre-named candidates. | P0 |
| Stale caches (4) | Cache metadata stores source SHA-256, code git hash and mapping-rule version; loaders verify or rebuild; `check_data.sh` runs at start of each script; result JSON embeds cache hashes. | P0 |
| Provenance (3) | Shared `run_record()` helper writes commit, dirty flag, seed, runtime, pair counts, hashes into every JSON; scripts refuse to run with a dirty tree unless `--allow-dirty` (flagged in output). Commit after each milestone. | P0 |
| ID/index misalignment (6) | End-to-end test: for 30 random scorable pairs, find columns in the raw CSV by HGNC symbol (independent of `ia`) and match cached-matrix Pearson. Assert known mappings (RPL/RPS, PSMA/PSMB, POLR2*) and cache `genes.txt` == raw header after each rebuild. | P0 |
| Untested code feeding results | Rule: module tests pass (`pytest -q`, logged) before a result enters the tree; every new method gets a toy positive and negative control first. | P0 |
| Nominal independent review | Define it: reviewer reruns from a clean clone with caches deleted, writes its own AP and split-disjointness checks (not importing `src/metrics.py`), signs a note with commit hash and result diffs. Required before any "clearly beats". | P0 |
| Cherry-picked seeds/variants; winner's curse | Select only on `SEEDS[0]` dev-CV; final comparison on fresh split seeds 1-4 plus IFT. Every script appends all trials to `results/trials.jsonl`; REPORT states the trial count. Model seeds also from `SEEDS`, mean and SD reported. | P1 |
| Baseline sandbagging | Equal tuning budget logged for baselines (signed/abs, library subset, OR removal, rank-deficiency mode, imputation) and the new method. Add ablation: LightGBM over Pearson+Spearman+GLS+gene stats, so gains are attributed to new layers rather than supervision. Min-observation filter for correlations. Keep GLS-vs-`gls_reference` test. | P1 |
| Metric gaming | Primary fixed in code (per-fold-mean AUPRC, complex-grouped dev-CV); secondary metrics cannot be promoted later. Always report prevalence, pooled AP and lift (AP/prevalence), since the withheld test prevalence is unknown. | P1 |
| Ties/NaN/coverage (5) | Every result JSON carries `n_nan_to_zero`, `n_unscorable`, `n_tied_at_min`, plus the "all-pairs, min-score rule" AUPRC. | P1 |
| Pair-key ordering | Test (a,b) vs (b,a) invariance on the real file; `predict_pairs.py` canonicalises. | P1 |
| Context loss across agents | Agent handback format: commit, command, result path, 3-line result, "not verified" list. PI records it as a tree node; tree top carries a "resume pointer" to the active node. | P1 |
| Unverifiable claims | REPORT tables generated by script from `results/*.json` (each number cites path and key). LITERATURE claims that drive design (e.g. Fischer 2025 complex-level split) carry DOI/quote or are marked unverified. DATA.md "26Q3" is user-stated only; say so. | P1 |
| Chromosome shortcut in Phase B | Report with and without chromosome features. | P2 |
| Dev vs withheld shift | Component CV is pessimistic for shared proteins, optimistic for unseen ones; report C1/C2/C3; call IFT "internal", never "test-set". | P2 |

## Rules to add to DESIGN section 6
- No tree entry without {commit, clean tree, command, results path, tests passed}.
- New feature families, metrics, or any IFT-derived choice require a dated design revision first.
- A failed negative control is a finding; never edit the control to pass.
