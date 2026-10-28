# datasets/ (placeholder — not distributed)

No DepMap, hu.MAP or HGNC files are included in this repository. Download them yourself and place them here; everything in this folder except these README files is git-ignored.

```
datasets/
├── DepMAP/CRISPRGeneEffect.csv            # required (DepMap Public 26Q3)
├── HGNC/hgnc_complete_set.txt             # required (UniProt → gene mapping, symbols)
└── HuMAP/Train_positive_pairs.txt         # optional, evaluation only
    HuMAP/Train_negative_pairs.txt
```

The scripts use these paths by default; pass `--depmap` / `--hgnc` to use other locations. See each sub-folder's README for the source and the SHA-256 checksum of the file used in this project.
