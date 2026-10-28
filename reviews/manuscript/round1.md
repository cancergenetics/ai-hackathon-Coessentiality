# Review round 1: manuscript, captions, S1 Text, S2 Text

Reviewer: independent agent (referee, statistician and copy editor). I did not write the manuscript. Date: 2026-10-27.
Read: `manuscript/manuscript.md`, `manuscript.built.md`, `figures/captions.md`, `S1_Text.md`, `S2_Text.md`, `references_verified.md`, `build.py`, Fig1–5.png. Evidence checked: `results/report/tables.md`, `results/ift/look01_2026-10-27.json`, the READMEs of `results/{phaseA,phaseA_graph,phaseB,phaseB_v2,phaseC,verify,baselines_v2}`, `results/splits/clusters_summary.json`, `results/trials.jsonl`, `results/ift_looks.jsonl`, `RESEARCH_TREE.md`, `DESIGN.md`, `DATA.md`, `LESSONS.md`, `reviews/retro/*`, `reviews/*`, `tests/test_metrics_v2.py`, git log. No manuscript file was edited.

**Overall.** Most numbers are correct. I checked every number in Table 1, Table 2 (except runtimes), the IFT paragraph, the dev deltas and S1 Tables A, I and K against the committed files, and they match. The main problems are: (i) the text contradicts itself about small complexes, (ii) it is too confident about the dev evidence after selection and about the IFT's independence, (iii) runtimes and one verification claim are wrong, and (iv) the build output has PLOS formatting defects. Recommendation: major revision before submission.

---

## A. MUST-FIX (ordered by importance)

**M1. The small-complex claims contradict each other and the data.**
- Abstract: "for small complexes the score matched GLS". Discussion: "for small complexes the final score only matched GLS".
- Results heading: "Combining three views recovers small complexes". Fig 4 title: "zavg_BEF extends it to smaller clusters".
- Evidence:
  - On dev, zavg_BEF did not match GLS on small complexes; it beat it. Macro AP was 0.395 vs 0.349, and AP without the top-10 clusters was 0.360 vs 0.296 (phaseB_v2 README, T3).
  - On the IFT, macro AP was equal (0.365 vs 0.369). But AP without the largest component was 0.470 vs 0.350, a larger gain than on all pairs (+0.120 vs +0.103; look01 `ap_ht_no_largest_component`).
- So "most of the gain came from large complexes" holds for dev (Fig 4C: +0.22 on top-10 pairs vs +0.064 elsewhere). It does not hold for the IFT split by largest component.
- Fix: use one consistent statement everywhere. Suggested: "The gain was largest for big complexes. On smaller complexes it was smaller in development data (macro AP 0.395 vs 0.349) and absent on the internal test by macro AP (0.365 vs 0.369)."
- Rename the Results heading, for example "Combining three views helps smaller complexes in development data".

**M2. Runtimes in Table 2 and the Runtime section are not "raw file to scores".**
- The Table 2 caption says "Runtime: raw file to scores for 634,779 pairs". The Runtime section says "under 1 s for Pearson".
- T7 lists Pearson, Spearman, E and F through `predict_pairs` as "n/a (not in results)".
- Pearson's 0.40 s (T3) is from the cached matrix for 596,597 pairs. It excludes matrix loading, which alone takes 3.1–4.9 s (T7). No all-pairs runtime exists for Spearman.
- F's 6 s and E's 103 s are component times inside the zavg run (T7). They are not standalone runs.
- Fix: footnote each cell with its basis, or give only measured raw-to-score values (B 28 s, zavg_BEF 137 s, GLS 24 s) and mark the rest as component or cached times.

**M3. "Independent code, to within 10⁻¹⁶" overstates how independent the IFT verification was** (Results, end of "Confirmation on held-out complexes").
- The verifier computed the metrics with its own code, but from the scores of the producer's `predict_pairs.py`, run as a black box. Those metrics match to ≤1e-16.
- The verifier's own re-implementation of the scores agreed with the predictor only to r = 0.9999997 (max abs diff 0.057), with the same AP to 4 d.p. (`results/verify/README.md` §IFT).
- Reword: "An agent that had not produced the result recomputed all IFT metrics with its own code (identical to 10⁻¹⁶). Its independent re-implementation of the scores gave the same AP to four decimals."

**M4. The dev cross-validation is called "the main statistical evidence"** (Results, final IFT paragraph). It is the data on which selection happened.
- About 300 variants and 822 logged trials were selected on dev seed 0, and seed 0 is part of the 5-seed mean.
- The nested checks covered selection only within a family (67 graph layers; 4 z-averages). They did not cover the whole pipeline (k, τ, q, d, the choice of members).
- DESIGN v3.1 says the dev selection bias "is why the IFT point-estimate gate remains".
- The five seeds re-split the same 190 clusters, so they are not independent replications. DESIGN §3.7 says they are a "split-variance check, not fresh evidence".
- Reframe: the dev CV estimates the effect size with optimistic CIs. The IFT is the only test free of selection, but it is small (Kish n_eff 7.7) and partly exposed (D1).

**M5. "Pre-registered" needs a qualifier, and the D1 exposure belongs in Results.**
- The criteria were fixed in a version-controlled internal document (DESIGN v3.1, commit d94db6f, 13:29). They were not registered on a public registry, and the same agent system wrote and used them.
- Say this once, in Methods: "pre-specified in a version-controlled design document before the test was opened". Use "pre-specified" elsewhere.
- D1 (reviewers computed AP on all pairs, including future IFT pairs, before the split) appears only in the Discussion. Mention it in "Confirmation on held-out complexes" and qualify "held-out" in the Abstract.

**M6. The macro-AP definition behind B's failure was not pre-specified.**
- `results/verify/README.md`: DESIGN §3.5 does not define macro per-cluster AP.
- Under a within-cluster-only definition, B passes against GLS (0.744 vs 0.722). zavg_BEF passes under both definitions.
- The definition used (all pairs touching the cluster, HT-weighted, 56 clusters) was fixed in code (ff0dd52) before the look. Say so in Methods and note that B's failure depends on this definition.
- Results ("This confirms on independent complexes the pattern...") should be softened to "is consistent with".

**M7. The D3 selection rule is described inconsistently.**
- Methods: "the best combination by nested development AP".
- RESEARCH_TREE D3: "by 5-seed mean of nested held-out HT-AP".
- What was applied (phaseB_v2 README, "D3 decision (applied literally)"): the argmax over 8 combinations of the 5-seed mean of fold-mean held-out HT-AP. This was not a nested selection; nested selection among all 8 sometimes picked LR or LGBM.
- State the rule that was applied.
- Also disclose that the candidate members (B, E, H, F) were chosen after seeing dev results. S1 §8: F was added after the Phase C side finding. D3 fixed only the final argmax.

**M8. The margin of zavg_BEF over B is misstated in both directions.**
- Discussion: "the +0.02 margin of zavg_BEF over B depended on the exact eigenmap settings". This is wrong. The size was stable (0.466–0.470; 24/25 folds won under every E variant). Only its significance depended on E (CI > 0 in 2/5, 4/5 or 5/5 seeds; verify README, "Robustness").
- Results: "95% CI above zero in all five". Add "(seed 2 lower bound +0.0005)".

**M9. The controls were run on B, not on the final method.**
- The OR control (7.2% vs GLS 6.6%) and the same-chromosome control are B-only (T6, phaseA/compare.json). No committed OR or chromosome control exists for zavg_BEF.
- State that these controls refer to B, or run them on zavg_BEF.
- "The excess came from pairs on the same chromosome (20% vs 3.2%)": different-chromosome OR pairs (3.2%) also exceed the 1% expectation. Say "mostly".

**M10. Build and file defects in `manuscript.built.md` / docx.**
- (a) Reference 26 (Ledoit–Wolf) has lost its DOI. `build.py` drops the DOI when `"doi" in` the Vancouver string, and "Le**doi**t" contains "doi". Use a regex such as `\bdoi:`.
- (b) Reference 21 prints "doi:none (book)". Filter out `none`.
- (c) "# Figure captions" appears twice: manuscript.md adds the heading and captions.md has its own.
- (d) The captions.md preamble ("Figures are generated by `scripts/50_make_figures.py`...") and the file paths in every caption get into the manuscript.
- (e) The Fig 3 caption cites `source_data/Fig5_ablation_seed0.csv`. The file is now `Fig3_ablation_seed0.csv` (git status: renamed, not committed).
- (f) manuscript.md, S2_Text.md, build.py, the new figures and source_data are uncommitted. Commit them before you rely on them.

**M11. Two citations do not support the sentence they are attached to.**
- Introduction ¶4: "AUPRC, which is more informative than ROC analysis at low prevalence {saito2015, mcdermott2024}". McDermott et al. 2024 argue against the general claim that AUPRC is better under class imbalance. The credit line in references_verified.md says to cite them for "AUPRC preferentially rewards ... sub-populations with more positives; justifies ... macro per-component AUPRC". Move the citation to the sentence on macro AP (Results ¶ HT weights) and keep only saito2015 here.
- Introduction ¶2: "DepMap has screened more than a thousand cell lines with genome-wide CRISPR libraries {tsherniak2017, dempster2021chronos}". Tsherniak 2017 used RNAi in about 500 lines. Cite the 26Q3 release or data portal (1,361 models) for the count, and keep tsherniak2017 for DepMap as a resource.

**M12. The agent-setup description cannot be verified, and part of it contradicts the record.**
- "Human input consisted of the prompt, the data files and one 'continue' message." The prompt (S2) asks only for REPORT.md, not a manuscript. The references, S1 Text, figures and this manuscript (commits 23b8fb7–0a678aa, after M5 at 17:53) therefore needed further human instruction. State it.
- "25 sub-agents (18 Opus, 7 Sonnet)" and the role table cannot be checked from the repository; no agent log is committed. The parts I could match are 8 reviewer files, 2 retros and 3 verification commits.
- The table appears to cover the project only up to M5. The reference-verification, S1-drafting, figure and manuscript-review agents are missing.
- Add a supplementary table: agent, model, brief (one line), commits. Or say the counts come from the session log.
- "First to last commit took about 5.5 hours." The first commit was at 12:16, M5 (REPORT) at 17:53 (5.6 h), and the last commit at 21:26 (9.2 h). Write: "5.6 h from setup to final report; manuscript preparation took a further ~3.5 h."

**M13. The Methods notation is wrong or undefined** (Final method, Filter).
- X is defined, but Σ uses Y and G, which are undefined. If Y is cell lines × genes, then Σ = YYᵀ/G, not YᵀY/G (that is gene × gene). Define y_a and G = |BG|.
- State that the intercept direction is excluded from the sum defining W. Otherwise the ridge term gives it non-zero weight and "cosine (equal to the Pearson correlation)" fails.
- "With k=0, τ=0 this is GLS whitening": only GLS-like. Genes are standardised here and scores are cosines, not regression t-statistics. Fig 3 shows 0.279 for this setting vs 0.288 for GLS.

**M14. Claims about PC removal are not supported by the tests that were run.**
- Results ("Standardising genes before whitening"): "fell when 40 components were removed". k = 40 was tested only without gene standardisation (S1 Table B). With standardisation, only k ∈ {0, 5, 20} was tested.
- "Flat for 0–20" is a seed-0 range (0.359–0.365), but 0.395 in the same paragraph is a 5-seed mean. Label both.
- Over 5 seeds, F with k = 20 and with k = 0 both give 0.395 (phaseB_v2 README). PC removal adds nothing to F. Fig 3 shows +0.004.
- The Discussion still lists "removing dominant principal components" as one of the prior ideas that contributed. Say its contribution was negligible here.

**M15. Data provenance is stated as fact but unconfirmed in DATA.md.**
- The hu.MAP labels are "hu.MAP 3.0 (assumed; to confirm with user)", and the licence of these specific files is unverified.
- The HGNC download date is unknown.
- DepMap 26Q3 is "as stated by user". The manuscript says "as stated by the data provider", which reads as DepMap itself.
- Confirm these, or state them as assumptions.

**M16. PLOS format blockers.**
- The abstract contains a bulleted list. The PLOS abstract is plain prose; make it one paragraph or two.
- The abstract uses "DepMap" without expanding it, and "CI" undefined.
- Figure captions must sit directly after the paragraph that first cites each figure, not in an end section.
- Data and Code availability contain placeholders. Funding and Competing-interests statements are missing.
- Fig 3's caption contains a bullet list. Convert it to prose.

---

## B. Major (should fix)

1. **No strong prior-literature normaliser as a reference row.** The paper claims improvement over "conventional" scores. The obvious published alternative is Pearson after PC removal (Boyle 2018; Hassan 2023 "onion" layer). It was computed (S1 Table C: k = 20, τ = ∞, 0.330 at seed 0, vs 0.365 for F) but is not in Table 2. Add it as a reference row (5 seeds and IFT if available, otherwise dev seed 0), or explain its absence. A referee will ask for it, and for BaCoN or mutual rank (S1 Table F).
2. **B has the lowest IFT AUROC of all methods**: 0.674 vs 0.726–0.739 for the baselines. On dev, B's AUROC is also below GLS (0.755 vs 0.765). The neighbourhood score sharpens the top of the ranking but hurts the global ranking. Discuss this in one sentence; it matters to users who threshold low.
3. **Report the IFT AP including unmapped pairs** (zavg 0.332, GLS 0.233; T5). The withheld test will include such pairs.
4. **Give the uncertainty of the headline relative gains.**
   - The Author summary says "about 50% ... and 42%" with no interval. The IFT relative gain CI is +12.6% to +90%; quote it.
   - "About 50%" is the ratio of 5-seed fold means (0.470/0.312). Say so.
   - The absolute IFT AP CI is [0.197, 0.590]; give it once.
5. **Explain the design choices in the main text (one sentence each):**
   - Why components above 40 proteins are split: hu.MAP complexes have ≤30 subunits, so a 144-protein component merges overlapping complexes.
   - Why Louvain, and with which resolution (1).
   - Why 5 folds and a 25% IFT.
   - Why equal weights. Currently only in S1 §6: there is nothing to fit, and learned weights did not win.
   - Why the final method is label-free.
   - The Kish n_eff formula.
   - The definition of the essentiality strata. Currently only in the Fig 5 caption; move it to Methods.
6. **Selection optimism.** "Nested checks suggest an optimism of 0.002–0.01 AP" should say these checks were within-family and do not cover the whole configuration search (see M4).
7. **"Supervision did not help."**
   - Limit the claim to ~1,900 training positives per fold and to these model classes. The Discussion's "With about 3,000 positive pairs" is the total, not the training size per fold.
   - The stackers had higher macro AP (0.415–0.418 vs 0.395). Mention it in the main text; S1 already does.
   - The monotone-LGBM all-label shuffle anomaly (0.146; S1 §7.4) is a control that did not behave as expected. Give it one sentence in Controls.
8. **"Signed scores beat absolute ones ..., so negative correlation carried no information."** The data show only that absolute values did not help on balance. Say "so negative correlations added no useful signal on balance".
9. **"A model using only gene-level statistics scored 0.024 ... so protein identity did not leak."** The 2.6× prevalence reflects transferable essentiality propensity. The leakage evidence is the split comparison (S1 Table L). Rephrase.
10. **"The IFT values are lower than development values for every method, which reflects the composition of the 57 held-out clusters."** This was not tested. Say "probably reflects" and give a supporting fact, for example that 4 of the top-10 clusters and 2 giant-component communities are in the IFT (clusters_summary.json).
11. **Overstatements to remove:**
    - "That shared signal is more stable than any single pairwise correlation." Untested.
    - "Small complexes have few partners to share, so neighbourhood overlap helps them little." A hypothesis stated as fact; say "probably because".
12. **Two choices "were not obvious"** (Discussion). Standardising variables before a covariance or PCA is standard practice. Soften to "had the largest effect", or cite a prior use.
13. **Credit gaps** (from the credit lines in references_verified.md):
    - The olfactory-receptor PC correction originates with Boyle 2018, not only Wainberg (Results, "Conventional baselines").
    - The same-chromosome exclusion should cite kim2019 next to lazar2024 (Controls).
    - Wainberg also flagged syntenic artefacts.
    - No citation for the Nyström extension, the cluster bootstrap, ARPACK or the software used (networkx/Louvain implementation, scikit-learn, LightGBM, PyTorch, NumPy/SciPy). PLOS CB expects software to be cited; this needs new verified references.
14. **Safeguard wording.**
    - "Only one script could load the IFT pairs." The test enforces that only `final_eval.py` imports the IFT loader. The verifier rebuilt the IFT pairs from raw files (look "1-verify"). Say "import the IFT loader" and list all logged looks (0, 1, 1-verify, 1-figure).
    - The look-1 run record has `git_dirty: true` (src clean). Disclose it next to the "no silent code changes" safeguard.
    - Of the milestones, only 2 had a fresh-agent retrospective, although the prompt asked for one at each. Disclose it or drop the implied completeness.
15. **The process-error list is incomplete.** The retros record more errors:
    - The research tree went stale.
    - The no-top-10 CIs against the baselines were never computed (T4 note).
    - Phase B v1 used provisional layers.
    - Some runs overrode the clean-tree check.
    Add the first two, or say "main errors".

---

## C. Factual discrepancies (all; location → stated → correct, source)

| # | Location | Stated | Correct / evidence |
|---|---|---|---|
| F1 | Methods, Baselines | "would remove 1,356 of 1,361 lines" | removes 1,355 (leaves 6 of 1,361): `results/baselines_v2/README.md` l.57; S1 §1 agrees |
| F2 | Table 2 caption + Runtime | Pearson "<1", Spearman "~1", F 6, E 103 "raw file to scores" | not measured raw→scores; Pearson 0.40 s cached; Spearman all-pairs n/a; F/E are component times (T3, T7) |
| F3 | Results, Confirmation | "independent code, to within 10⁻¹⁶" | metrics ≤1e-16 on producer-predictor scores; independent scores to 4 d.p. (verify README) |
| F4 | Discussion | "+0.02 margin ... depended on the exact eigenmap settings" | size stable (0.466–0.470); significance depended (2–5/5 seeds) (verify README) |
| F5 | Methods, Candidate selection | "best combination by nested development AP" | argmax of 5-seed mean fold-mean HT-AP over 8 combinations (phaseB_v2 README "D3 decision") |
| F6 | Abstract, Discussion | small complexes "matched GLS" | dev macro 0.395 vs 0.349; no-top-10 0.360 vs 0.296; IFT no-largest 0.470 vs 0.350; only IFT macro ≈ (0.365 vs 0.369) |
| F7 | Results, Standardising | "fell when 40 components were removed" | k = 40 tested only without standardisation (S1 Table B; phaseA README) |
| F8 | Results, Controls | OR/chromosome controls presented for "the neighbourhood score" | B only; none for zavg_BEF (T6) |
| F9 | Methods, Agent setup | "first to last commit took about 5.5 hours" | 12:16→21:26 = 9.2 h; 12:16→17:53 (M5) = 5.6 h (git log) |
| F10 | Methods, Agent setup | Human input = prompt, data, one "continue" | the prompt does not request a manuscript; post-M5 work needed further instruction; counts unverifiable |
| F11 | Fig 3 caption | `source_data/Fig5_ablation_seed0.csv` | `source_data/Fig3_ablation_seed0.csv` |
| F12 | Methods, Hyper-parameter search | "821 completed trials" | `results/trials.jsonl`: 823 start, 822 end records (464 ok + 358 ok_not_written). REPORT.md says "about 1,100". Reconcile |
| F13 | Methods, Uncertainty | "Coverage on toy data was 93%" | no committed source; `tests/test_metrics_v2.py::test_cluster_bootstrap_coverage_toy` asserts only ≥85%, uses raw (not HT) weighting, 60 datasets. Commit the number or drop it |
| F14 | Results ¶ benchmark | largest component "holds 26% of positive pairs" | 848/3,191 = 26.6% → 27% (clusters_summary.json); 26% only if divided by all 3,314 |
| F15 | Results ¶ benchmark | "set aside 25% of positive pairs" | 26.5% (847/3,191; `ift_frac_of_positives` 0.265). Say "about 25% (26.5%)" |
| F16 | Methods, Baselines | GLS covariance "estimated from all genes" | selected variant (exact marginalisation): Σ from the 17,082 genes without NaN (baselines_v2 README l.57) |
| F17 | Methods, Filter | "With k=0, τ=0 this is GLS whitening" | GLS-like only (gene-standardised, cosine); 0.279 vs GLS 0.288 (Fig 3) |
| F18 | Methods, Filter | Σ = YᵀY/G | YYᵀ/G if Y is lines × genes; Y, G, y_a undefined |
| F19 | Fig 3 caption | "whitening equivalent to GLS" | not equivalent (0.279 vs 0.288); say "GLS-like" and explain the gap (imputation + cosine vs per-pair marginalisation + t-statistic) |
| F20 | Fig 1A / caption | kNN graph "union of both directions" feeds B; neighbours "under F" | B uses directed top-K lists (Methods B step 2); union is for E only; neighbours are ranked by S (row-z-symmetrised F) |
| F21 | Fig 4C caption | "Without the largest clusters, B is close to F and GLS" | B 0.339, F 0.343, GLS 0.299: close to F, not GLS |
| F22 | Results, Shared neighbourhoods / Combining | 73/80 and 93/57 of 177 clusters | correct, but seed 0 only; say so |
| F23 | Results, Standardising | "flat ... ridge 0.01–0.3" next to 0.395 | flatness is seed 0 (0.359–0.365); 0.395 is the 5-seed mean |
| F24 | Data, Methods | "release 26Q3 as stated by the data provider"; "hu.MAP 3.0" | DATA.md: 26Q3 "as stated by user"; hu.MAP 3.0 "assumed; to confirm with user" |
| F25 | Intro ¶2 | ">1,000 lines ... {tsherniak2017, dempster2021chronos}" | the count is supported by the 26Q3 data (1,361), not by Tsherniak 2017 (RNAi) |
| F26 | Intro ¶4 | {mcdermott2024} supports "AUPRC more informative" | it argues the opposite in general; credit line says use for macro AP |
| F27 | Discussion | "Proteins without a DepMap profile ... 3.7%" | correct (123/3,314); "multi-copy histone" includes the 2 ambiguous histones, which did map but to several genes; say "ambiguous or missing" |
| F28 | Agent setup table | "Retrospectives at milestones: 2" | correct count, but the prompt asks for one at each milestone; also the table omits post-M5 agents |
| F29 | Built refs | ref 26 missing DOI; ref 21 "doi:none (book)" | build.py bug (M10) |
| F30 | S1 §9 | "computed label-free AP on all scorable pairs" | AP needs labels; say "computed AP of label-free scores" |
| F31 | S1 §2.1 | "tuning k and τ gained little (0.241–0.332)" | best non-standardised (0.332) is +0.053 over GLS-like 0.279, close to the standardisation step (+0.061); "little" understates it |

Checked and correct (no action): Table 1 (all cells; clusters_summary.json); Table 2 AP, macro, no-top-10, IFT AP, macro and AUROC for all rows; IFT Δs and CIs (+0.103 [+0.035, +0.209], +0.199 [+0.099, +0.321], +0.211 [+0.103, +0.338]); +42%; P@R0.25 0.77 vs 0.35; strata 0.276/0.388/0.394 vs 0.214/0.349/0.263; B IFT values; Kish 7.7; power 0.215; 241 components, 247 clusters, 121 crossing edges; 1,114/1,151 mapped, 2 ambiguous + 35 missing; 27,046 unlabelled pairs; dev Δs vs GLS +0.175..+0.193, vs B +0.014..+0.021 and +0.024..+0.031; nested 0.468; stackers 0.462/0.466; gene-only 0.024; shuffle controls 0.009–0.011; OR 7.2%/6.6%/20%/3.2%; chromosome 0.426→0.435; 137 s, 4.1 GB, GLS 24 s, E 103 s inside zavg; K = 44; 17,525 BG genes; rank 1,357/1,361; S1 Tables A, B, C, G, H, I, K, L, M spot-checked against READMEs.

---

## D. PLOS Computational Biology compliance

| Item | Status |
|---|---|
| Title ≤ 250 chars (PLOS) / ≤ 200 requested | 137 chars: OK |
| Short title ≤ 70 | 55 chars: OK |
| Abstract ≤ 300 words | 277 words: OK, **but contains a bullet list** (not allowed in the abstract field); "DepMap" not expanded; "CI" undefined |
| Author summary 150–200 words | 183 words: OK. Uses "precision–recall area" (technical for lay readers); add the CI for 42% |
| Section order | Abstract, Author summary, Introduction, Results, Discussion, Materials and methods, Data/Code availability, Acknowledgments, References, Supporting information: OK (PLOS CB allows Methods after Discussion). Add Funding and Competing interests (submission form or manuscript) |
| Heading levels ≤ 3 | 2 levels plus bold run-in headings: OK |
| Footnotes | none: OK |
| Figure captions placement | **Not compliant**: PLOS requires each caption directly after the paragraph that first cites the figure; they are in an end section with a duplicated heading |
| Figures cited in order | Fig 1B is cited before Fig 1A, and Fig 2A before Fig 1A; Fig 2C before 2B. Acceptable, but cite Fig 1 (or 1A) in the Introduction's last paragraph |
| Tables | Table 1 and 2 cited in order, titled: OK |
| Captions | Bold title + legend: OK. Remove file paths, script names and internal phase names ("Phase B v2", "look 1-figure") from captions; Fig 3 bullet list → prose |
| S-files | Names "S1 Text", "S2 Text" OK, with captions in SI section: OK. S2 is cited first (Introduction) and S1 later (Results). Swap the numbering so S1 is the prompt (recommended) |
| Data availability | Placeholder "[URL/DOI to be added]"; hu.MAP "from its website": give the URL (humap3.proteincomplexes.org). DESIGN D0 said not to commit derived pair lists, but the release ships predictions for all training pairs; confirm the hu.MAP file licence (DATA.md: "licence of these specific files unverified") |
| Code availability | Placeholder; PLOS CB requires code at submission (GitHub + Zenodo DOI) |
| AI-use disclosure | Present (Intro, Agent setup, Acknowledgments): OK |
| Abbreviations at first use | Undefined: ROC (Intro ¶4), AUROC (Table 2), HGNC (Methods, Data), CI (Abstract), kNN/OOF/PR in captions (OOF is defined in Fig 2B; kNN is not defined). DepMap is not expanded in the Abstract |
| Equations | LaTeX in Methods; check that pandoc renders them as editable Word equations |
| References | Vancouver, numbered in order: OK, apart from the two build defects (M10). kish1965: section of the formula not verified (references_verified.md note) |

---

## E. Clarity edit list (six patterns only)

Patterns: (1) rare word combinations or coined phrases, (2) inanimate subjects with human verbs, (3) unusual noun-to-verb conversions, (4) missing "that" where two nouns collide, (5) stacked nouns, (6) long subject–verb gaps.

| # | Location | Original | Pattern | Proposed |
|---|---|---|---|---|
| E1 | Intro ¶2 | "Co-essentiality networks built from these screens recover known complexes and have assigned functions to uncharacterised genes" | 2 | "...recover known complexes and have been used to assign functions to uncharacterised genes" |
| E2 | Results, benchmark ¶2 | "First, a protein's label propensity, meaning how many positive partners it has, can be learned from its identity and transferred between random pair splits." | 6, 1 | "First, a model can learn from a protein's identity how many positive partners it has, and carry this over between random pair splits." |
| E3 | Results, Standardising | "When each gene was first standardised to unit variance, the top 20 eigencomponents were removed and the rest whitened with a ridge term, first-order similarity (F) reached 0.395" | 6 | "We first standardised each gene to unit variance, then removed the top 20 eigencomponents and whitened the rest with a ridge term. The resulting first-order similarity (F) reached 0.395" |
| E4 | Results, Supervision ¶1 | "improve on the label-free geometry" | 1 | "improve on the label-free scores" |
| E5 | Discussion ¶3 | "labels did not add information beyond the unsupervised geometry" | 1 | "labels did not add information beyond the label-free scores" |
| E6 | Results, Controls | "Pairs of olfactory receptor genes, which are not essential and should behave as background, exceeded the null 99th percentile in 7.2% of cases under B" | 6 | "Olfactory receptor genes are not essential, so pairs of them should score as background. Under B, 7.2% of these pairs exceeded the null 99th percentile" |
| E7 | Results, Controls | "a known CRISPR proximity artefact" | 5 | "a known proximity artefact of CRISPR screens" |
| E8 | Discussion, process bullets | "A power simulation showed the pre-registered significance gate on the internal test could not be passed." | 4 | "A power simulation showed that the pre-specified significance gate on the internal test would rarely be passed (probability ≤ 0.22)." (also fixes "could not") |
| E9 | Discussion, process bullets | "**Pre-split data exposure.**" | 5 | "**Data exposure before the split.**" |
| E10 | Discussion, process bullets | "**Unpassable significance gate.**" | 1 | "**A significance gate that could rarely be passed.**" |
| E11 | Methods, Data ¶1 | "DepMap CRISPRGeneEffect (Chronos), release 26Q3 as stated by the data provider (the files carry no version stamp), with Model, ModelCondition, CRISPRConfounders and ScreenGeneEffect." (no verb) | 6 | "We used DepMap CRISPRGeneEffect (Chronos), release 26Q3 as stated by the human investigators (the files carry no version stamp), together with Model, ModelCondition, CRISPRConfounders and ScreenGeneEffect." |
| E12 | Methods, B step 1 | "and balance it, $S_{aj}=(z_{aj}+z_{ja})/2$" | 3, 1 | "and symmetrise it, $S_{aj}=(z_{aj}+z_{ja})/2$" |
| E13 | Methods, Evaluation design, criterion 2 | "on pooled out-of-fold development predictions" | 5 | "on the pooled out-of-fold predictions for the development set" |
| E14 | Methods, Software | "the protein–cluster–split assignment" | 5 | "the assignment of proteins to clusters and splits" |
| E15 | Results heading | "Combining three views recovers small complexes" | 1 | "Combining three views helps smaller complexes in development data" (also M1) |
| E16 | Fig 5 title | "The single pre-registered internal-final-test look confirms the gain." | 5 | "A single pre-specified evaluation on the internal final test confirms the gain." |
| E17 | Fig 5B | "HT-AP by pair essentiality stratum" | 5 | "HT-AP by the essentiality stratum of the pair" |
| E18 | Fig 4 caption ¶1 | "Dev pooled out-of-fold predictions for split seed 0 (cached label-free Phase B v2 scores, ...)" | 5 | "Pooled out-of-fold predictions on the dev set (split seed 0)" |
| E19 | Results, Confirmation ¶3 | "Its effective size is small" (after "57 held-out clusters") | 6 | "The IFT's effective size is small" (the referent is ambiguous after a long sentence) |

Left alone on purpose: technical terms (HT-AP, row-z, whitening, Laplacian eigenmap, Kish), "leakage-aware" (from the prompt), and "clearly beats" (quoted pre-specified term).

---

## F. Brevity (cut or merge without loss)

1. **The AI-agent framing appears four times**: Abstract ¶1, Author summary ¶2, Introduction ¶1 and Methods. Cut Introduction ¶1 to one sentence: "The project was run by AI agents with minimal human input (Methods: Agent setup; S2 Text)."
2. Results, Shared neighbourhoods ¶2: delete "That shared signal is more stable than any single pairwise correlation." It is unsupported.
3. Results, Combining ¶3: "The z-scores use fixed means and standard deviations from 200,000 random genome-wide gene pairs" repeats Methods (Combination). Keep only "No label information enters the combination (Methods)."
4. **Runtime section**: it repeats the Table 2 column. Fold it into one sentence in the Table 2 caption or the Discussion.
5. Discussion, last sentence ("The method will be evaluated independently on the withheld hu.MAP test pairs; we did not access them.") repeats the Abstract and Methods (Data). Keep it in Methods and the Abstract.
6. Independent verification is described in Results (Confirmation), Discussion ("the most useful safeguard") and Methods (safeguards). Keep the Results sentence and the Methods bullet; cut the Discussion sentence or merge it into the process paragraph.
7. Results, Supervision: the three-bullet list of metric-learning variants → one sentence ("Learned cell-line weights, low-rank projections and second-order scores on the learned space all stayed within ±0.003 of their starting point (S1 Text).").
8. Methods, Hyper-parameter search: the four bullets repeat S1 §2–5. One sentence with the counts is enough.
9. Agent setup bullet "read their summaries rather than raw logs": low information; cut.
10. Captions: remove the preamble paragraph, every "Source: `results/...`" clause, the Fig 4 closing sentence ("The per-cluster means equal the pooled macro AP in ..."), and the Fig 5A logging sentence ("The curves come from a re-plot ... logged as look '1-figure' ..."). Put the provenance in the repository README or an S-file.
11. Discussion, prior-ideas list: topological overlap and SNN are cited with the same three references in Results and again in the Discussion. Keep one (the Discussion list) and cut the Results sentence to "This follows topological-overlap ideas (see Discussion)", or the reverse.

---

## G. Counts per category

- **Must-fix:** 16 (M1–M16)
- **Major (should fix):** 15 (B1–B15)
- **Factual discrepancies:** 31 (F1–F31), several overlapping with must-fix items
- **PLOS compliance issues:** 9: abstract bullets; abstract abbreviations; caption placement; captions with paths and bullets; figure citation order (minor); S-file citation order (minor); data/code placeholders; Funding/Competing interests missing; undefined abbreviations
- **Clarity edits:** 19 (E1–E19): pattern 1 ×6, pattern 2 ×1, pattern 3 ×1, pattern 4 ×1, pattern 5 ×6, pattern 6 ×5 (some hits count under two patterns)
- **Brevity cuts:** 11
