"""Data loading: ID mapping, canonical pair table, cached DepMap matrices, sample metadata, HGNC locations.

All paths come from src.config. Raw inputs are never modified; derived caches go to data_cache/ (gitignored).

ID mapping rule (UniProt -> DepMap column), documented and deterministic
-----------------------------------------------------------------------
1. HGNC rows with status == "Approved"; ``uniprot_ids`` split on '|' (one row per UniProt accession).
2. Each UniProt accession gets the set of candidate Entrez IDs of all HGNC genes listing it.
3. Candidates are intersected with the Entrez IDs of DepMap columns (``SYMBOL (ENTREZ)``; matched on
   Entrez, never on symbol, because DepMap symbols can lag HGNC renames).
4. Exactly one candidate in DepMap -> mapped. Zero -> ``not_in_depmap`` (or ``not_in_hgnc`` if step 2 is
   empty). Several -> ``ambiguous``: dropped and logged, because the protein is encoded by several
   paralogous gene copies (e.g. replication-dependent histones) and choosing one copy's knockout profile
   would be arbitrary.
5. Two different UniProt accessions mapping to the same DepMap column are allowed but logged; any pair
   whose two proteins land on the same column is flagged ``same_gene`` and is not scorable.

Pairs with an unmapped protein stay in the pair table with ``scorable == False``; evaluation excludes them
from scoring by default and reports their count.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CACHE, DEPMAP, HGNC_FILE, NEG_FILE, POS_FILE

log = logging.getLogger(__name__)

MATRICES = ("CRISPRGeneEffect", "ScreenGeneEffect")

# ScreenGeneEffect index is "SC-NNNNNN.XXrr". The two-letter code XX was checked against the CRISPRConfounders
# library one-hot of model ACH-NNNNNN (see screen_metadata / results/baselines/data_mapping.json):
# AV->Avana 1049/1049 consistent, KY->KY 322/322, CD->HumagneCD 220/220, TK->TKOv3 4/4, BN->Brunello 12/13.
SCREEN_CODE_TO_LIBRARY = {"AV": "Avana", "KY": "KY", "CD": "HumagneCD", "BN": "Brunello", "TK": "TKOv3"}


# ----------------------------------------------------------------------------------------------- HGNC
def load_hgnc() -> pd.DataFrame:
    h = pd.read_csv(HGNC_FILE, sep="\t", dtype=str, low_memory=False)
    return h[h.status == "Approved"].reset_index(drop=True)


def depmap_columns(matrix: str = "CRISPRGeneEffect") -> pd.Index:
    return pd.read_csv(DEPMAP / f"{matrix}.csv", nrows=0).columns[1:]


def parse_depmap_column(col: str) -> tuple[str, str]:
    """'A1BG (1)' -> ('A1BG', '1')."""
    m = re.match(r"^(.*) \((\d+)\)$", col)
    if not m:
        raise ValueError(f"unexpected DepMap column {col!r}")
    return m.group(1), m.group(2)


_LOC_RE = re.compile(r"^(\d{1,2}|X|Y)(?:([pq])(\d+(?:\.\d+)?)?)?")


def parse_location(loc) -> dict:
    """Parse an HGNC cytogenetic location string.

    '19q13.43' -> chrom '19', arm 'q', band '13.43', band_num 13.43. 'mitochondria' -> chrom 'MT'.
    Ranges ('1p36.33-p36.32') keep the first band; 'Xp22.33 and Yp11.2' (PAR) keeps the first locus.
    'cen', 'reserved', 'unplaced', 'not on reference assembly', missing -> chrom None where undetermined.
    HGNC provides NO base-pair coordinates, so genomic distance can only be approximated at cytoband
    resolution (same chromosome / same arm / |band_num difference|); this is coarse (bands span Mb).
    """
    out = dict(chrom=None, arm=None, band=None, band_num=np.nan)
    if not isinstance(loc, str):
        return out
    s = loc.strip()
    if s.startswith("mitochondria"):
        out["chrom"] = "MT"
        return out
    m = _LOC_RE.match(s)
    if not m:
        return out
    out["chrom"], out["arm"], out["band"] = m.group(1), m.group(2), m.group(3)
    if out["band"]:
        out["band_num"] = float(out["band"])
    return out


def gene_annotation(genes: pd.Index | list[str]) -> pd.DataFrame:
    """Per-DepMap-column HGNC annotation (matched on Entrez): symbol, hgnc_id, location, chrom, arm, band,
    locus_group, gene_group. Genes absent from HGNC get NaN."""
    h = load_hgnc().dropna(subset=["entrez_id"]).drop_duplicates("entrez_id")
    h = h.set_index("entrez_id")[["hgnc_id", "symbol", "location", "locus_group", "gene_group"]]
    h = h.rename(columns={"symbol": "hgnc_symbol"})
    parsed = [parse_depmap_column(c) for c in genes]
    out = pd.DataFrame(dict(gene=list(genes), depmap_symbol=[p[0] for p in parsed],
                            entrez=[p[1] for p in parsed]))
    out = out.join(h, on="entrez")
    loc = pd.DataFrame([parse_location(x) for x in out.location])
    return pd.concat([out, loc], axis=1).set_index("gene")


# ------------------------------------------------------------------------------------------ ID mapping
def build_uniprot_map(uniprot_ids, dep_cols: pd.Index) -> pd.DataFrame:
    """Map UniProt accessions to DepMap columns following the module-level rule. One row per accession."""
    h = load_hgnc()
    m = h[["symbol", "entrez_id", "uniprot_ids"]].dropna(subset=["uniprot_ids", "entrez_id"]).copy()
    m["uniprot"] = m.uniprot_ids.str.split("|")
    m = m.explode("uniprot")
    cand = m.groupby("uniprot").entrez_id.apply(lambda s: sorted(set(s)))
    ent2col = {parse_depmap_column(c)[1]: c for c in dep_cols}
    col2idx = {c: i for i, c in enumerate(dep_cols)}
    rows = []
    for u in sorted(set(uniprot_ids)):
        c = cand.get(u, [])
        indep = [e for e in c if e in ent2col]
        if not c:
            status = "not_in_hgnc"
        elif not indep:
            status = "not_in_depmap"
        elif len(indep) > 1:
            status = "ambiguous"
        else:
            status = "mapped"
        col = ent2col[indep[0]] if status == "mapped" else None
        rows.append(dict(uniprot=u, status=status, hgnc_entrez=";".join(c), depmap_entrez=";".join(indep),
                         gene=col, gene_idx=col2idx[col] if col else -1))
    df = pd.DataFrame(rows).set_index("uniprot")
    dup = df[df.status == "mapped"].groupby("gene").size()
    df["shared_gene"] = df.gene.isin(dup[dup > 1].index)
    for st in ("not_in_hgnc", "not_in_depmap", "ambiguous"):
        ids = df.index[df.status == st].tolist()
        if ids:
            log.info("UniProt %s (%d): %s", st, len(ids), ids)
    if df.shared_gene.any():
        log.info("UniProt sharing a DepMap gene: %s", df[df.shared_gene].gene.to_dict())
    return df


# ------------------------------------------------------------------------------------------ pair table
def load_pairs(dep_cols: pd.Index | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonical pair table and UniProt map.

    Columns: a, b (UniProt, a < b lexicographically), label (1 pos / 0 neg), gene_a, gene_b (DepMap column or
    None), ia, ib (column index in the cached matrix gene order, -1 if unmapped), scorable (both mapped and
    different genes), same_gene. Row order: positives then negatives, each in file order.
    Gene indices refer to CRISPRGeneEffect column order; ScreenGeneEffect has identical columns (asserted
    in build_matrix_cache).
    """
    if dep_cols is None:
        dep_cols = depmap_columns("CRISPRGeneEffect")
    parts = []
    for f, lab in ((POS_FILE, 1), (NEG_FILE, 0)):
        d = pd.read_csv(f, sep="\t", header=None, names=["x", "y"], dtype=str)
        a = np.where(d.x.values < d.y.values, d.x.values, d.y.values)
        b = np.where(d.x.values < d.y.values, d.y.values, d.x.values)
        parts.append(pd.DataFrame(dict(a=a, b=b, label=np.int8(lab))))
    p = pd.concat(parts, ignore_index=True)
    if (p.a == p.b).any():
        raise ValueError("self pairs present")
    dup = p.duplicated(["a", "b"], keep=False)
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicated/conflicting pairs")
    umap = build_uniprot_map(pd.unique(p[["a", "b"]].values.ravel()), dep_cols)
    p["gene_a"] = umap.gene.reindex(p.a).values
    p["gene_b"] = umap.gene.reindex(p.b).values
    p["ia"] = umap.gene_idx.reindex(p.a).values.astype(np.int64)
    p["ib"] = umap.gene_idx.reindex(p.b).values.astype(np.int64)
    p["same_gene"] = (p.ia >= 0) & (p.ia == p.ib)
    p["scorable"] = (p.ia >= 0) & (p.ib >= 0) & ~p.same_gene
    return p, umap


def mapping_report(pairs: pd.DataFrame, umap: pd.DataFrame) -> dict:
    """Counts for the record (no pair lists)."""
    rep = dict(n_uniprot=len(umap), status_counts=umap.status.value_counts().to_dict(),
               unmapped_ids={st: umap.index[umap.status == st].tolist()
                             for st in ("not_in_hgnc", "not_in_depmap", "ambiguous")},
               ambiguous_detail=umap.loc[umap.status == "ambiguous", ["hgnc_entrez", "depmap_entrez"]]
               .to_dict(orient="index"),
               multi_entrez_in_hgnc=int((umap.hgnc_entrez.str.count(";") > 0).sum()),
               multi_entrez_resolved_by_depmap=umap.loc[(umap.hgnc_entrez.str.count(";") > 0)
                                                        & (umap.status == "mapped"), "gene"].to_dict(),
               shared_gene=umap.loc[umap.shared_gene, "gene"].to_dict(),
               n_mapped_genes=int(umap.gene.nunique()))
    for lab, name in ((1, "pos"), (0, "neg")):
        s = pairs[pairs.label == lab]
        rep[name] = dict(total=len(s), scorable=int(s.scorable.sum()), unmapped=int((~s.scorable & ~s.same_gene).sum()),
                         same_gene=int(s.same_gene.sum()))
    return rep


# --------------------------------------------------------------------------------------- matrix cache
def _cache_paths(matrix: str) -> dict[str, Path]:
    return dict(X=CACHE / f"{matrix}.f32.npy", genes=CACHE / f"{matrix}.genes.txt",
                samples=CACHE / f"{matrix}.samples.txt", meta=CACHE / f"{matrix}.meta.json")


def build_matrix_cache(matrix: str, force: bool = False) -> None:
    assert matrix in MATRICES
    p = _cache_paths(matrix)
    if p["X"].exists() and not force:
        return
    CACHE.mkdir(exist_ok=True)
    df = pd.read_csv(DEPMAP / f"{matrix}.csv", index_col=0)
    ref = depmap_columns("CRISPRGeneEffect")
    if not df.columns.equals(ref):
        raise ValueError(f"{matrix} columns differ from CRISPRGeneEffect columns")
    np.save(p["X"], df.values.astype(np.float32))
    p["genes"].write_text("\n".join(df.columns) + "\n")
    p["samples"].write_text("\n".join(df.index) + "\n")
    p["meta"].write_text(json.dumps(dict(shape=list(df.shape), nan_frac=float(df.isna().values.mean()),
                                         dtype="float32", source=str(DEPMAP / f"{matrix}.csv"))))


def load_matrix(matrix: str = "CRISPRGeneEffect", mmap: bool = False) -> tuple[np.ndarray, pd.Index, pd.Index]:
    """Returns (X samples x genes float32, sample ids, gene columns). Builds the cache on first use."""
    build_matrix_cache(matrix)
    p = _cache_paths(matrix)
    X = np.load(p["X"], mmap_mode="r" if mmap else None)
    genes = pd.Index(p["genes"].read_text().rstrip("\n").split("\n"))
    samples = pd.Index(p["samples"].read_text().rstrip("\n").split("\n"))
    assert X.shape == (len(samples), len(genes))
    return X, samples, genes


# ------------------------------------------------------------------------------------- sample metadata
LIBS = ["Avana", "HumagneCD", "KY", "TKOv3", "Brunello"]


def _models() -> pd.DataFrame:
    return pd.read_csv(DEPMAP / "Model.csv", dtype=str).set_index("ModelID")


def _confounders() -> pd.DataFrame:
    return pd.read_csv(DEPMAP / "CRISPRConfounders.csv", index_col=0)


def model_metadata(samples: pd.Index) -> pd.DataFrame:
    """CRISPRGeneEffect rows (ACH- model IDs) joined with CRISPRConfounders (screen QC, library one-hot) and
    Model.csv lineage. ``libraries`` is '+'-joined library names (models screened with >1 library were
    combined by DepMap)."""
    cf = _confounders().reindex(samples)
    md = _models().reindex(samples)[["CellLineName", "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype"]]
    out = pd.concat([md, cf], axis=1)
    lib = cf[[f"Library{l}" for l in LIBS]].fillna(0).values > 0
    out["libraries"] = ["+".join(l for l, b in zip(LIBS, row) if b) for row in lib]
    out["n_libraries"] = lib.sum(1)
    out.index.name = "sample"
    return out


def screen_metadata(samples: pd.Index) -> pd.DataFrame:
    """ScreenGeneEffect rows 'SC-NNNNNN.XXrr' -> ModelID ACH-NNNNNN (numeric-ID correspondence; validated by
    library agreement with CRISPRConfounders), library from code XX, replicate/screen number rr, lineage.
    ``library_consistent`` is True when the confounder one-hot of the model includes that library."""
    s = pd.Series(samples, index=samples)
    m = s.str.extract(r"^SC-(\d+)\.([A-Z]{2})(\d+)$")
    if m.isna().any().any():
        raise ValueError("unexpected ScreenGeneEffect index format")
    out = pd.DataFrame(index=samples)
    out["ModelID"] = "ACH-" + m[0]
    out["library_code"] = m[1]
    out["library"] = m[1].map(SCREEN_CODE_TO_LIBRARY)
    out["screen_number"] = m[2].astype(int)
    cf = _confounders()
    flag = [cf.at[mid, f"Library{lib}"] if mid in cf.index else np.nan
            for mid, lib in zip(out.ModelID, out.library)]
    out["library_consistent"] = pd.Series(flag, index=samples).map({1.0: True, 0.0: False})
    md = _models().reindex(out.ModelID)
    out["OncotreeLineage"] = md.OncotreeLineage.values
    out["CellLineName"] = md.CellLineName.values
    out.index.name = "sample"
    return out


def sample_metadata(matrix: str, samples: pd.Index) -> pd.DataFrame:
    return model_metadata(samples) if matrix == "CRISPRGeneEffect" else screen_metadata(samples)
