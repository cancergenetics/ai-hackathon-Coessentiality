"""Build derived caches (data_cache/, gitignored) and write the ID-mapping / metadata report.

Caches: <matrix>.f32.npy + .genes.txt + .samples.txt for CRISPRGeneEffect and ScreenGeneEffect,
<matrix>.samples_meta.csv (metadata join), gene_annotation.csv (HGNC per DepMap gene), pairs.pkl (canonical
pair table), uniprot_map.csv.
Report (committed, counts + ID lists only, no pair lists): results/baselines/data_mapping.json

Usage: ./.env/bin/python scripts/01_build_cache.py [--force]
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from src.config import CACHE, RESULTS  # noqa: E402
from src import data  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    CACHE.mkdir(exist_ok=True)
    rep = {}
    for m in data.MATRICES:
        data.build_matrix_cache(m, force=args.force)
        X, samples, genes = data.load_matrix(m, mmap=True)
        meta = data.sample_metadata(m, samples)
        meta.to_csv(CACHE / f"{m}.samples_meta.csv")
        r = dict(shape=list(X.shape))
        if m == "CRISPRGeneEffect":
            r["libraries"] = meta.libraries.value_counts().to_dict()
            r["n_multi_library_models"] = int((meta.n_libraries > 1).sum())
            r["models_missing_confounders"] = int(meta.ScreenNNMD.isna().sum())
        else:
            r["library_code_counts"] = meta.library_code.value_counts().to_dict()
            r["library_vs_confounders"] = {
                code: dict(consistent=int((g.library_consistent == True).sum()),  # noqa: E712
                           inconsistent=int((g.library_consistent == False).sum()),  # noqa: E712
                           model_not_in_confounders=int(g.library_consistent.isna().sum()))
                for code, g in meta.groupby("library_code")}
            r["n_models"] = int(meta.ModelID.nunique())
        r["lineage_missing"] = int(meta.OncotreeLineage.isna().sum())
        rep[m] = r
    ann = data.gene_annotation(genes)
    ann.to_csv(CACHE / "gene_annotation.csv")
    rep["hgnc_gene_annotation"] = dict(
        genes=len(ann), with_hgnc=int(ann.hgnc_id.notna().sum()), with_chrom=int(ann.chrom.notna().sum()),
        chrom_counts_top=ann.chrom.value_counts().head(5).to_dict(),
        note="HGNC 'location' is a cytoband (e.g. 19q13.43); HGNC has no base-pair coordinates, so genomic "
             "distance is only approximable at band resolution.")
    pairs, umap = data.load_pairs(genes)
    pairs.to_pickle(CACHE / "pairs.pkl")
    umap.to_csv(CACHE / "uniprot_map.csv")
    rep["mapping"] = data.mapping_report(pairs, umap)
    rep["mapping_rule"] = data.__doc__.split("ID mapping rule")[1].split("Pairs with an unmapped")[0].strip()
    rep["runtime_s"] = round(time.time() - t0, 1)
    out = RESULTS / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_mapping.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != "mapping_rule"}, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
