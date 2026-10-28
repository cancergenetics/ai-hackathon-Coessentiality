# How the agents ran this project

The project was run as a test of autonomous AI-agent research. Humans wrote the task ([PROMPT.md](../PROMPT.md)), supplied the data files, sent one "continue" message during the research phase, and later asked for the manuscript and this repository in a second prompt. Everything else was done by agents: literature, design, code, experiments, evaluation, verification, the report, the manuscript and this repository.

## Set-up

- **PI agent:** Claude Opus 5.5, running in the Claude Code desktop app.
  - Kept the research record: the research tree with decision notes, the trial log and a lessons file.
  - Wrote task briefs, reviewed and integrated sub-agent results, and made every decision.
  - Its own context stayed lean: it read agents' summaries rather than raw logs.
- **Sub-agents:** 25 in the research phase (18 Claude Opus 5.5, 7 Claude Sonnet 5.5) and 8 more in the manuscript phase (33 in total; listed in S3 Text of the manuscript). Each got a written brief and returned a summary; results were committed to git.

| role | agents |
|---|---|
| Literature review | 1 |
| Infrastructure (data loading, ID mapping, baselines; evaluation splits and metrics) | 2 |
| Design review, round 1: external PI, statistics, algorithms/compute, data/biology, prior work, AI-agent failure modes | 6 |
| Design review, round 2 | 2 |
| Method scientists: Phase A layers, Phase A screen, Phase B v1 (supervised stacking), graph family (A7), Phase C (metric learning), Phase B v2 | 6 |
| Engineering: standalone predictor, final-evaluation alignment, report tables | 3 |
| Independent verifiers | 3 |
| Retrospectives ("what should we stop doing?", "what worked well?") | 2 |

## Safeguards

- **Design review.** The design document went through two review rounds by independent reviewer agents; the prompt allowed up to three. Each reviewer proposed concrete changes or said none were needed. A further amendment, v3.1, followed a power analysis and was made before any test look.
- **Withheld data.** The hu.MAP test set was withheld from the agents by design and never accessed. All development used hu.MAP train pairs.
- **Leakage-aware evaluation.**
  - Dev used cluster-grouped cross-validation, keeping complexes disjoint, with a fixed seed list.
  - About 25% of the positives (whole clusters) were set aside as an internal final test.
  - The candidate methods and the decision rule were pre-specified (in a version-controlled design document, not a public registry) before the internal test was looked at. It was looked at once (one disclosed earlier exposure is documented in the report).
- **Independent verification.** Every headline result was re-derived by an agent that did not produce it, starting from the raw files and using its own code. Verification covered exact reproduction, a leakage audit and negative controls.
- **Controls and toy tests.** About 200 tests used toy data with known answers, plus positive and negative controls: label and gene shuffles, chromosome-only and gene-statistics-only models, and an olfactory-receptor artefact check.
- **Process logs.**
  - The research tree recorded decisions, with milestone summaries.
  - An append-only trial log recorded 845 started and 844 completed evaluations.
  - An internal-test look log, design reviews, retrospectives and a lessons file were also kept.
  - All of these stay in the private research repository. This public repository contains the final method, its tests, the predictions and the summary tables.

## What the humans did

- Wrote the task prompt.
- Downloaded and supplied the DepMap, HGNC and hu.MAP train files.
- Will run the independent evaluation on the withheld hu.MAP test set.
