# ai-hackathon-Coessentiality — predicting protein co-complex membership from DepMap co-essentiality (AI hackathon project)

## Disclaimer

(Human-written text starts)

This is an exercise in assessing the bioinformatics capabilities of agentic AI. This exercise was performed by the Ryan lab at University College Dublin, collaborating as part of a hackathon run by the Beltrao Lab at ETH Zurich, where we used Claude to run bioinformatics projects. All results in this project were generated entirely by AI with minimal human intervention. This is not intended to be viewed as a traditional publication and bioinformatics tool.

The goal of this specific project is to develop an approach to predict co-complex membership from gene co-essentiality. It was executed by Claude Opus 5.5 with delegation to Opus 5.5/Sonnet 5.5 subagents, and finished within 24 hours on a personal 13-inch MacBook Pro equipped with an Apple M2 chip and 24 GB of unified memory.

Currently none of the results have been verified by a human. We plan to verify the agent’s methodology and benchmarking, and we will update this section accordingly.

The rest of this repository is entirely AI generated. It includes code for implementation of the predictive model and a AI-written report in the style of a journal article, detailing the final method, the path followed to reach it, and benchmarking against other methods.

(Human-written text ends)

## What is here

| file / directory | contents |
|---|---|
| `PROMPT.md` | the project brief, verbatim |
| `REPORT.md` | final report: approach, results, limitations, follow-ups |
| `RESEARCH_TREE.md` | research log: status summaries, decision notes D0–D4, tree of what was tried |
| `DESIGN.md` | design (v3.1, after two review rounds) with the revision log |
| `LITERATURE.md` | literature review (with errata from the prior-work review) |
| `DATA.md` | data sources, versions, checksums and licences |
| `LESSONS.md` | process lessons for running agentic research projects |
| `manuscript/` | AI-written manuscript in PLOS Computational Biology format (DOCX + Markdown source), S1–S3 Texts, figures and figure source data |
| `docs/` | agent set-up (`AGENTS.md`), cluster/split assignment format, data checksums |
| `reviews/` | design reviews (rounds 1–2), retrospectives, manuscript reviews (rounds 1–2) |
| `src/` | research code: ID mapping, splits and clusters, metrics, baselines, label-free layers (incl. the final method), supervised models, evaluation harness, predictor |
| `scripts/` | numbered pipeline scripts (`00_`–`64_`), the internal-test evaluation (`final_eval.py`), the pair scorer (`predict_pairs.py`), independent verification scripts (`verify/`) |
| `tests/` | unit, toy-data (known-answer) and process tests for the research code |
| `results/` | committed metrics and logs: baselines, every method phase, verification, the single internal-test evaluation (`results/ift/`), trial and test-access logs, split assignment, genome-wide predictions |
| `mvp/` | minimal standalone implementation of the final method (`coessential` package) with its own tests; easiest way to score new pairs |
| `datasets/` | placeholders only: where to put the DepMap, HGNC and hu.MAP files (git-ignored) |

## Results in brief

Task: predict whether two proteins are in the same complex (hu.MAP 3.0 train pairs) from DepMap CRISPR co-essentiality, beating Pearson, Spearman and GLS. Evaluation keeps complexes disjoint between training and test; the metric is AUPRC with Horvitz–Thompson weights (HT-AP) at the natural prevalence (~0.5%).

| method | dev HT-AP (cluster-grouped 5-fold CV, 5 seeds) | internal test HT-AP (one pre-specified evaluation) |
|---|---|---|
| **zavg_BEF (final, label-free)** | **0.470** | **0.345** |
| GLS (Wainberg et al. 2021) | 0.312 | 0.242 |
| Spearman | 0.282 | 0.146 |
| Pearson | 0.260 | 0.134 |

zavg_BEF averages three z-scored similarities of standardised, PC-filtered, whitened gene-effect profiles: direct correlation (F), overlap of genome-wide co-dependency neighbourhoods (B) and proximity in a Laplacian eigenmap of the neighbour graph (E). On the internal test, zavg_BEF − GLS = +0.103 (95% cluster-bootstrap CI +0.035 to +0.209). Dev numbers are optimistic (method selected on dev); the internal test is small (≈8 effective clusters) and was partly exposed early on; gains are concentrated in large complexes. The withheld hu.MAP test set was never accessed. Details: `REPORT.md` and `manuscript/`.

The final method builds on published ideas: covariance whitening / GLS (Wainberg et al. 2021; Gheorghe & Hart 2022), principal-component removal and multi-layer similarity (Boyle et al. 2018; Hassan et al. 2023), per-gene balancing (mutual rank, Obayashi & Kinoshita 2009; BaCoN, Rohde et al. 2025), topological overlap and shared-nearest-neighbour similarity (Ravasz et al. 2002; Zhang & Horvath 2005; Jarvis & Patrick 1973) and Laplacian eigenmaps (Belkin & Niyogi 2003). Full references: `manuscript/references_verified.md`.

## Data

No data files are included in this repository. To reproduce, download the files listed in `DATA.md` and place them under `datasets/` as described in [`datasets/README.md`](datasets/README.md):

- `datasets/DepMAP/` — DepMap Public 26Q3 `CRISPRGeneEffect.csv` (plus `Model.csv`, `ModelCondition.csv`, `CRISPRConfounders.csv`, `ScreenGeneEffect.csv` for the full research pipeline), <https://depmap.org/portal/data_page/>;
- `datasets/HGNC/` — `hgnc_complete_set.txt`, <https://www.genenames.org/download/archive/>;
- `datasets/HuMAP/` — hu.MAP 3.0 `Train_positive_pairs.txt`, `Train_negative_pairs.txt` (evaluation only), <https://humap3.proteincomplexes.org/>.

All data, caches (`data_cache/`) and derived score matrices are git-ignored. `scripts/check_data.sh` verifies the checksums.

## Quick start

Score your own protein pairs with the standalone implementation (two tab-separated UniProt accessions per line, no header):

```bash
pip install -r mvp/requirements.txt
python mvp/scripts/predict_pairs.py --pairs pairs.txt --method zavg_BEF --out scores.tsv          # run from the repository root
python mvp/scripts/predict_genome_wide.py --top 200000 --out top_pairs.tsv.gz                   # genome-wide ranking
(cd mvp && python -m pytest)                                                                    # toy-data tests
```

The research pipeline (Python 3.12; `pip install -r requirements.txt`) reproduces every number: `scripts/01_build_cache.py` → `scripts/03_make_splits.py` (run with `PYTHONHASHSEED=0`) → baseline and layer scripts → `scripts/21_phaseB_v2.py`; see `REPORT.md` and `RESEARCH_TREE.md` for the order and outputs. `scripts/predict_pairs.py --method zavg:B,E,F` is the research version of the scorer. The precomputed genome-wide top 200,000 pairs are in `results/predictions/`. Research tests: `python -m pytest tests --ignore=tests/test_supervised.py --ignore=tests/test_metric_learning.py`, then run those two files separately (PyTorch and LightGBM can deadlock on OpenMP in one process); tests that need the raw data or `data_cache/` are skipped when those are absent.

## Data sources and attribution

The predictions are derived from DepMap data (Broad Institute; CC BY 4.0): credit DepMap when reusing them (the 26Q3 release, and Dempster et al. 2021, *Genome Biol* 22:343). Gene symbols and UniProt IDs come from HGNC. `results/splits/protein_assignment.csv` is derived from the hu.MAP 3.0 train positive pairs (Fischer et al. 2025, *Mol Syst Biol*); it lists proteins and their evaluation clusters, not pairs or labels.
