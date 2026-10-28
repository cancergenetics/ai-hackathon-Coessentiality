# Cluster and split assignment (`results/splits/protein_assignment.csv`)

This is the canonical protein-level assignment used to evaluate the method. It is copied unchanged from the research
repository, and was generated deterministically with PYTHONHASHSEED=0. It contains one row for each of the 1,151 UniProt
accessions in the hu.MAP 3.0 train pair files.

The clusters are built from the train **positive** pairs, so this file encodes the train complex structure. It does not list pairs or labels.

| column | definition |
|---|---|
| `protein` | UniProt accession |
| `component`, `component_size` | connected component of the positive-pair graph among mapped proteins (-1 / 0: unmapped) |
| `cluster`, `cluster_size` | evaluation cluster: the components, except that the 144-protein giant component is split into 7 Louvain communities (247 clusters) |
| `louvain_split` | True if the cluster came from splitting the giant component |
| `stratum` | stratum for the stratified dev/IFT assignment: `giant` (Louvain communities of the giant component), `top10` (the 10 other clusters with the most within-cluster positives), `pos_q1`–`pos_q3` (tertiles of within-cluster positives among the rest), `no_pos` (no within-cluster positive), `unmapped` |
| `cluster_within_pos` | number of positive train pairs within the cluster |
| `side` | `dev` (development cross-validation) or `ift` (internal final test, about 25% of positives by whole clusters, looked at once) |
| `mapped` | True if the protein maps to a unique DepMap gene |

Evaluation pairs are pairs with both proteins on the same side. For a test set of whole clusters drawn at protein fraction
f, the HT weight is 1 for pairs within a cluster and 1/f for pairs between clusters. f is the share of mapped proteins with
`side == "ift"` (0.265).

Example: `python mvp/scripts/evaluate.py --assignment results/splits/protein_assignment.csv --side ift ...` (see the main README); the research harness reads the same file via `src/clusters.py`.

Raw input files (DepMap, HGNC, hu.MAP) are **not** distributed; put them under `datasets/` (git-ignored; see [datasets/README.md](../datasets/README.md)).
