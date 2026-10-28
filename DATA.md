# DATA.md — datasets used

All raw files live in `datasets/` (gitignored; never committed). Derived caches in `data_cache/` (gitignored).
Checksums: SHA-256, computed 2026-10-27 with `shasum -a 256` (re-check: `scripts/check_data.sh`).

| File | Source | Version | SHA-256 | License |
|---|---|---|---|---|
| DepMAP/CRISPRGeneEffect.csv | DepMap Portal (Broad), depmap.org/portal/data_page | 26Q3 (as stated by user; files carry no version stamp) | 1e0c492944a7297c0eda179081a1e81a1cfe41a97628facd7eddbdda1a7504ee | CC BY 4.0 (DepMap public data terms) |
| DepMAP/ScreenGeneEffect.csv | DepMap Portal | 26Q3 (user) | 08e15ab81d7450db70c48c7781f312f6cebc068da7077bea775569fb516a3f04 | CC BY 4.0 |
| DepMAP/CRISPRConfounders.csv | DepMap Portal | 26Q3 (user) | 0e098325d53403d7852f3a773477d20a508a9bf29f43e10fe28bf0f550b35779 | CC BY 4.0 |
| DepMAP/Model.csv | DepMap Portal | 26Q3 (user) | 65eb1895800e71a963a3ab56e5d08c8d006cb0d9e7aa68f7e34a3810a2a3cbc7 | CC BY 4.0 |
| DepMAP/ModelCondition.csv | DepMap Portal | 26Q3 (user) | a1edda78cd60718e5c99690ce8ab80315b65ddc9346643338d2c1c774fc51326 | CC BY 4.0 |
| HuMAP/Train_positive_pairs.txt | hu.MAP (humap3.proteincomplexes.org) train split, provided by user | hu.MAP 3.0 (as stated by the user; not independently confirmed) | 77276dbac8ca108cac4a9c196b0d26bfac3912b616d9647b44adacdc88092fc9 | CC BY 4.0 per hu.MAP 3.0 paper/HF dataset card; licence of these specific files unverified |
| HuMAP/Train_negative_pairs.txt | as above | as above | c37e27dfe12ddfcecc34e084719a4f8b2aaf47e106455e3da46d7199aed836f3 | as above |
| HGNC/hgnc_complete_set.txt | HGNC, genenames.org | download date unknown (provided by user) | 0f2d3f035e0e6583c055a1d9e008f275376a4b66f72af380dfc2ac730272c247 | CC0 (genenames.org/about/license; per LITERATURE.md) |

## Notes / explicit assumptions
- Prompt refers to `Train_positive_pairs_pairs.txt`; the file on disk is `Train_positive_pairs.txt`. We use the on-disk file.
- HuMAP pairs are UniProt accessions (no isoforms). Mapping: UniProt -> HGNC (Approved, `uniprot_ids`) -> Entrez ID -> DepMap column `SYMBOL (ENTREZ)`. Mapping matches on Entrez, not symbol.
- Profile of raw data: `results/profile/data_profile.json` (scripts/00_profile_data.py).
- HuMAP Test pairs are withheld by design and are never accessed.
