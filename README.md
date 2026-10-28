# ai-hackathon-Coessentiality

## Disclaimer

(Human-written text starts)

This is an exercise in assessing the bioinformatics capabilities of agentic AI. This exercise was performed by the Ryan lab at University College Dublin, collaborating as part of a hackathon run by the Beltrao Lab at ETH Zurich, where we used Claude to run bioinformatics projects. All results in this project were generated entirely by AI with minimal human intervention. This is not intended to be viewed as a traditional publication and bioinformatics tool.

The goal of this specific project is to develop an approach to predict co-complex membership from gene co-essentiality. It was executed by Claude Opus 5.5 with delegation to Opus 5.5/Sonnet 5.5 subagents, and finished within 24 hours on a personal 13-inch MacBook Pro equipped with an Apple M2 chip and 24 GB of unified memory.

Currently none of the results have been verified by a human. We plan to verify the agent’s methodology and benchmarking, and we will update this section accordingly.

The rest of this repository is entirely AI generated. It includes code for implementation of the predictive model and a AI-written report in the style of a journal article, detailing the final method, the path followed to reach it, and benchmarking against other methods.

(Human-written text ends)

---

The original task prompt is in [PROMPT.md](PROMPT.md); the agent and sub-agent set-up is described in [AGENTS.md](AGENTS.md).

## What the method does

The task was to predict whether two proteins belong to the same protein complex, using only gene co-essentiality: how similarly the two genes behave when knocked out across 1,361 cancer cell models (DepMap CRISPR screens, release 26Q3). Labelled complex pairs came from hu.MAP 3.0.

The final score, **`zavg_BEF`**, needs no labels. Each gene's knockout profile is first cleaned of dominant shared effects: the profiles are standardised, the 20 strongest cell-line components are removed, and the rest is whitened with a ridge-regularised covariance (a regularised version of GLS, Wainberg et al. 2021). Three similarity views are then computed on these cleaned profiles:

- **F (direct):** the correlation of the two genes' cleaned profiles.
- **B (shared partners):** whether the two genes have the same strongest co-essential partners elsewhere in the genome. Each gene keeps its top 0.25% neighbours (about 44 genes); the score compares the two neighbour lists, leaving out the pair's own link.
- **E (network position):** closeness of the two genes in a 512-dimensional spectral embedding (Laplacian eigenmap) of the genome-wide neighbour graph.

Each view is converted to a z-score against 200,000 random gene pairs. `zavg_BEF` is the mean of the three z-scores. The method is deterministic, and a pair's score does not depend on which other pairs are scored. The full specification is in [`src/coessential/method.py`](src/coessential/method.py).

## Prior work this method builds on

The final method combines published ideas; the agents' contribution was to combine and tune them on a complex-disjoint benchmark.

- Co-essentiality predicts co-complex membership: Pan et al. 2018 (Cell Syst); Boyle et al. 2018 (Mol Syst Biol); Kim et al. 2019 (Life Sci Alliance).
- Whitening the cell-line covariance (GLS): Wainberg et al. 2021 (Nat Genet); whitening as the dominant normalisation step: Gheorghe & Hart 2022 (BMC Bioinformatics).
- Removing dominant principal components and combining normalised similarity layers ("onion"): Boyle et al. 2018; Hassan et al. 2023 (Mol Syst Biol).
- Per-gene balancing of similarity distributions (row-z): the logic of mutual rank (Obayashi & Kinoshita 2009, DNA Res) and BaCoN (Rohde et al. 2025, Mol Syst Biol).
- Second-order / shared-neighbour similarity: topological overlap (Ravasz et al. 2002, Science; Zhang & Horvath 2005, Stat Appl Genet Mol Biol) and shared-nearest-neighbour similarity (Jarvis & Patrick 1973, IEEE Trans Comput).
- Spectral embedding of the neighbour graph: Laplacian eigenmaps (Belkin & Niyogi 2003, Neural Comput).
- Evaluation: pair-input leakage (Park & Marcotte 2012, Nat Methods); large complexes dominating pooled precision–recall (Rahman et al. 2021, Mol Syst Biol); Horvitz–Thompson weighting (1952); Louvain communities (Blondel et al. 2008).

Full references with DOIs are in the manuscript (`paper/`).

## Headline results

Evaluation used hu.MAP 3.0 *train* pairs only, with splits that keep complexes disjoint. The primary metric is HT-AP: average precision with Horvitz–Thompson weights, which restore the natural share of between-complex negatives (prevalence ~0.5%). Full tables are in [results/summary.md](results/summary.md).

| method | dev HT-AP (cluster-grouped 5-fold CV, 5 seeds) | internal final test HT-AP (one pre-specified look) |
|---|---|---|
| **zavg_BEF** | **0.470** | **0.345** |
| GLS (Wainberg et al. 2021) | 0.312 | 0.242 |
| Spearman | 0.282 | 0.146 |
| Pearson | 0.260 | 0.134 |

- **Internal final test.** It held 847 positive and 40,786 negative pairs from held-out complexes. zavg_BEF − GLS = +0.103 HT-AP, 95% cluster-bootstrap CI [+0.035, +0.209]. The pre-specified "clearly beats" criterion was met against all three baselines.
- **Withheld test set.** The hu.MAP 3.0 test set was withheld from the agents and never accessed. Its evaluation is pending and will be done independently with `scripts/predict_pairs.py` (below).
- **Runtime.** About 2 min (118–137 s in two runs; run-to-run variation about ±20%) from the raw CSV to scores for 635k pairs on a 13-inch MacBook Pro with an Apple M2 chip and 24 GB memory. GLS: about 19–24 s; Pearson and Spearman: about 11 s.
- **Post-hoc reference.** Pearson correlation after removing the top 20 principal components (Boyle et al. 2018; Hassan et al. 2023) was added after the internal test, at a reviewer's request. It scored 0.363 on dev and 0.267 on the internal test, 0.078 below zavg_BEF (95% CI 0.028–0.151).

## Install

```bash
git clone https://github.com/cancergenetics/ai-hackathon-Coessentiality.git
cd ai-hackathon-Coessentiality
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # or: pip install -e .
python -m pytest                    # toy-data tests (~5 s)
```

Python ≥ 3.10. The published numbers were produced with Python 3.12 on arm64 macOS.

## Data (not included; download yourself)

No DepMap, hu.MAP or HGNC files are distributed with this repository. Download them and place them in the git-ignored [`datasets/`](datasets/README.md) folder:

| path | source | needed for |
|---|---|---|
| `datasets/DepMAP/CRISPRGeneEffect.csv` (DepMap **26Q3**, as stated by the project's human investigators; the file carries no version stamp) | DepMap portal, <https://depmap.org/portal/data_page/> (Downloads → DepMap Public 26Q3 → CRISPRGeneEffect.csv) | everything |
| `datasets/HGNC/hgnc_complete_set.txt` | HGNC, <https://www.genenames.org/download/archive/> (complete HGNC dataset, TSV) | UniProt → gene mapping, gene symbols |
| `datasets/HuMAP/Train_positive_pairs.txt`, `Train_negative_pairs.txt` | hu.MAP 3.0, <https://humap3.proteincomplexes.org/> | evaluation only |

Each sub-folder's README gives the SHA-256 checksum of the file used in this project. Other DepMap releases or HGNC versions will give somewhat different scores.

## Usage

Score a pair file (two tab-separated UniProt accessions per line, no header):

```bash
python scripts/predict_pairs.py --pairs pairs.txt --method zavg_BEF --out scores.tsv
# defaults: --depmap datasets/DepMAP/CRISPRGeneEffect.csv --hgnc datasets/HGNC/hgnc_complete_set.txt
```

- **Output columns:** `uniprot_a, uniprot_b, score, mapped`, in input order.
- **Methods:** `zavg_BEF` (default), `B`, `E`, `F` (raw component values), and the baselines `gls` (t), `pearson` (r) and `spearman` (ρ).
- **Extra columns:** `--components` adds `z_B, z_E, z_F`; `--add-gls` adds `gls_t`.
- **Unscorable pairs** get `mapped = 0` and score = (minimum mapped score − 1). Pass `--unmapped na` to leave them empty instead. Unscorable means a protein without a unique DepMap gene, or both proteins on the same gene.

Rank all gene pairs genome-wide (genes with < 10% missing values; about 2–3 min, ~2.5 GB RAM):

```bash
python scripts/predict_genome_wide.py --top 200000 --out top_pairs.tsv.gz
```

Evaluate a score file on labelled pairs:

```bash
python scripts/evaluate.py --scores scores.tsv --pos Test_positive_pairs.txt --neg Test_negative_pairs.txt
# internal final test of this project (needs your copy of the hu.MAP train files + the committed cluster assignment):
cat datasets/HuMAP/Train_positive_pairs.txt datasets/HuMAP/Train_negative_pairs.txt > train_pairs.txt
python scripts/predict_pairs.py --pairs train_pairs.txt --method zavg_BEF --out train_scores.tsv
python scripts/evaluate.py --scores train_scores.tsv \
    --pos datasets/HuMAP/Train_positive_pairs.txt --neg datasets/HuMAP/Train_negative_pairs.txt \
    --assignment data/protein_assignment.csv --side ift     # -> ap_ht 0.345
```

- **Default output:** AP and AUROC on mapped pairs, plus AP with unmapped pairs placed at the minimum score.
- **With `--assignment`:** adds HT-weighted AP and macro per-cluster AP.

The hyper-parameters are frozen at the published values. The `--k/--tau/--q/--dim` options exist for experiments, and changing them changes the method.

## Precomputed predictions

[`predictions/`](predictions/README.md) holds `genome_wide_top_pairs.tsv.gz`: the 200,000 top-scoring gene pairs genome-wide by zavg_BEF, with symbols, Entrez and UniProt IDs and the B/E/F z-scores. It was generated from the raw DepMap 26Q3 CSV with this code and matches the research code to within 3×10⁻⁵ in z. Scores for the hu.MAP pairs are not distributed; generate them from your own copy of the files with `scripts/predict_pairs.py` (see Usage).

`data/protein_assignment.csv` is the protein → cluster / split assignment used for evaluation (see [data/README.md](data/README.md)).

## Limitations

- **Gains are concentrated in large, dense complexes.** On dev, smaller complexes also improved (macro AP 0.395 vs 0.349 for GLS); on the internal test, macro AP only equals GLS (0.365 vs 0.369).
- **Dev estimates are optimistic.** About 300 label-free variants were screened on dev, and the members of the final average were chosen after seeing dev results.
- **The internal test is small** (effective number of clusters ≈ 8), so its CIs are wide. It was also partly exposed: before it was split off, two design-review agents computed AP of label-free scores on all pairs (no selection was made from those numbers).
- **About 3.7% of positive train pairs involve a protein without a unique DepMap gene** (immunoglobulin segments, mtDNA-encoded genes, haemoglobins, two multi-copy histones). These pairs get the minimum score.
- **CRISPR proximity artefacts are slightly amplified.** Of olfactory-receptor gene pairs, 8.4% score above the null 99th percentile under zavg_BEF (expected 1%; GLS 6.6%), mostly pairs on the same chromosome.
- **Results depend on the DepMap release and on hu.MAP 3.0's complex composition** (complexes ≤ 30 subunits).

## How to cite

Repository: <https://github.com/cancergenetics/ai-hackathon-Coessentiality> (see [CITATION.cff](CITATION.cff)). An AI-written manuscript describing the method is in [paper/](paper/README.md).

If you use the predictions, please also cite the data sources:

- DepMap: cite the 26Q3 release as described on the DepMap portal, and Dempster et al. 2021, *Genome Biol* 22:343 (Chronos).
- hu.MAP 3.0: Fischer et al. 2025, *Mol Syst Biol* 21:911–943.
- HGNC.
- For the GLS baseline: Wainberg et al. 2021, *Nat Genet* 53:638–649.

## Data sources and attribution

- The predictions are derived from DepMap data (Broad Institute; released under CC BY 4.0), so credit DepMap when reusing them: cite the 26Q3 release as described on the DepMap portal and Dempster et al. 2021, *Genome Biol* 22:343.
- Gene symbols and UniProt IDs come from HGNC.
- `data/protein_assignment.csv` is derived from the hu.MAP 3.0 train positive pairs (Fischer et al. 2025, *Mol Syst Biol*); it lists proteins and their evaluation clusters, not pairs or labels.
- No raw DepMap, HGNC or hu.MAP file, and no pair label, is included in this repository.

## Process records

- [PROMPT.md](PROMPT.md): the original task prompt, verbatim.
- [AGENTS.md](AGENTS.md): how the agent team was organised and checked.
- [paper/](paper/README.md): the AI-written manuscript (PLOS Computational Biology format) with supporting texts and figures.
