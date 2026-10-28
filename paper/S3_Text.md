# S3 Text. Agent log

## Setup

All work ran in one Claude Code desktop session. A principal-investigator agent (PI; Claude Opus 5.5) received the task prompt (S1 Text). It wrote the design and record files (DESIGN.md, RESEARCH_TREE.md, LESSONS.md, REPORT.md), made every decision, and delegated work to the sub-agents listed below.

Each sub-agent received a written brief that set out:

- the goal;
- the files to read;
- the files it owned;
- the rules: no access to internal-test pairs outside the evaluation script, fixed seeds, logging of every trial, and committing its own files only.

Sub-agents reported back with short summaries. The PI read those summaries, not the raw logs. Several sub-agents ran in parallel when their files did not overlap. The PI also forwarded follow-up instructions to running agents; for example, it changed the primary metric to Horvitz–Thompson weighting mid-task.

## Human input

1. The task prompt (S1 Text) and the data files.
2. One message, "continue", during the research phase.
3. After the final report (commit 5341f12), a second prompt asked for this manuscript, a public code repository and the supplementary files.

## Sources

The list below comes from the session record and the git history of the project repository. Commit times are local times on 27 Oct 2026 (first commit 12:16; final report 17:53; manuscript phase from about 21:00). Short hashes refer to that repository.

## Research phase: 25 sub-agents (18 Opus 5.5, 7 Sonnet 5.5)

| # | Role | Model | Brief (one line) | Output / commits |
|---|---|---|---|---|
| 1 | Literature review | Opus | Search the literature and write LITERATURE.md with design implications and licences | LITERATURE.md (b4c7937) |
| 2 | Infrastructure v1 | Opus | ID mapping, caches, metrics, split generators, Pearson/Spearman/GLS baselines, toy tests | 9bc81a2 |
| 3 | Design review R1: external PI | Opus | Review DESIGN v1; propose in-scope changes | docs/reviews/round1_external_pi.md |
| 4 | Design review R1: statistics | Opus | Evaluation protocol, bootstrap, multiplicity, power | round1_statistics.md |
| 5 | Design review R1: algorithms and compute | Opus | Method choices and feasibility | round1_algorithms_compute.md |
| 6 | Design review R1: data and biology | Sonnet | DepMap and hu.MAP data issues | round1_data_biology.md |
| 7 | Design review R1: prior work | Sonnet | Check literature claims; find missed work | round1_prior_work.md |
| 8 | Design review R1: AI-agent failure modes | Sonnet | Leakage, silent bugs, process safeguards | round1_ai_failure_modes.md |
| 9 | Infrastructure v2 | Opus | Implement DESIGN v2/v3: clusters, IFT, HT-AP, cluster bootstrap, logging, IFT isolation, power check | 1155d55, 47dd58b |
| 10 | Design review R2: PI and statistics | Opus | Second-round review of DESIGN v2 | round2_pi_statistics.md |
| 11 | Design review R2: failure modes and algorithms | Sonnet | Second-round review of DESIGN v2 | round2_ai_algorithms.md |
| 12 | Method: label-free layers | Opus | Build and cache 128 label-free similarity layers; tests | 920b8c0, 6b2f6e3 |
| 13 | Method: layer screen | Opus | Screen layers on dev, label-free ensembles, controls | 34be32c, 7bd4c4b |
| 14 | Method: supervised stacking v1 | Opus | Gene features, LightGBM/LR scorers, controls, leakage gap | 68908d0–b3480c2 |
| 15 | Verifier 1 | Opus | Re-implement the best layer from raw data; leakage audit; controls | a2f8622 |
| 16 | Method: graph layers | Opus | kNN graphs, diffusion, spectral embeddings; nested check | 3bd38a6, c79e37f, ea7c7f5 |
| 17 | Method: metric learning | Opus | Supervised metric learning ladder (time-boxed) | 952b265, 47cd366, 26623ab |
| 18 | Retrospective 1 | Sonnet | "What should we stop doing / what worked well?" | docs/retros/retro_M3a.md |
| 19 | Method: supervised stacking v2 | Opus | Final combination round on frozen layers; apply rule D3 | e242a7c |
| 20 | Engineering: predictor | Opus | Standalone scorer from the raw CSV; invariance tests | b8fdf0f–e724ac9 |
| 21 | Verifier 2 | Opus | Re-implement zavg_BEF (incl. eigenmap); robustness to settings | 460fd36 |
| 22 | Engineering: final evaluation | Opus | Align the IFT script with DESIGN v3.1 (dry run only) | ff0dd52 |
| 23 | Verifier 3 | Opus | Re-derive the IFT evaluation from raw files | b914a8e |
| 24 | Engineering: report tables | Sonnet | Generate report tables from committed results only | 8abe43c |
| 25 | Retrospective 2 | Sonnet | Final retrospective | docs/retros/retro_final.md |

The PI ran the internal-test evaluation itself (commit 8267142), using the command and candidates fixed beforehand.

## Manuscript phase

| # | Role | Model | Brief (one line) | Output / commits |
|---|---|---|---|---|
| 26 | Code repository | Opus | Build the public MVP repository, port the method, regenerate predictions, privacy check | github.com/cancergenetics/ai-hackathon-Coessentiality (built locally; published by the authors); a88a929 |
| 27 | Figures | Opus | Figures 1–5 from committed results; IFT re-plot logged | f90806c, 0a678aa |
| 28 | References | Sonnet | Verify references in Vancouver style | 23b8fb7 |
| 29 | Supplementary draft | Opus | Draft S2 Text from committed results only | c920ff1 |
| 30 | Manuscript review | Opus | Fact-check, overstatement, PLOS compliance, clarity edits | manuscript/review_round1.md |
| 31 | Revision analyses | Opus | Controls on zavg_BEF, post-hoc reference baseline, runtimes, bootstrap coverage | results/revision/ |
| 32 | Software references | Sonnet | Verify software and method references | 3dd8b32 |
| 33 | Manuscript review, round 2 | Opus | Second independent check of the revised manuscript | manuscript/review_round2.md |

## Main errors and how they were caught

| Error | How it was caught | Consequence |
|---|---|---|
| Round-1 reviewers computed AP of label-free scores on all pairs before the IFT existed | PI noticed it in their reports | Disclosed (look 0); no selection |
| Louvain cluster assignment depended on PYTHONHASHSEED | Failing regeneration test, reported by sub-agent 14 | Canonical assignment committed; seed enforced |
| IFT significance gate had too little power | Power simulation by sub-agent 9 | Gate amended before the IFT was opened |
| Several supervised families run at full scale despite null pilots | Retrospectives | Recorded in LESSONS.md |
| final_eval.py would crash on non-integer look IDs | Sub-agent 27 | Fixed |
| Manuscript overstated small-complex gains and mislabelled runtimes | Sub-agent 30 | Revised |
