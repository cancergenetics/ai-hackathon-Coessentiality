"""Input files and identifier mapping.

DepMap gene-effect matrix
-------------------------
``CRISPRGeneEffect.csv`` (DepMap portal): rows = cell-line models (first column = model ID), columns =
``SYMBOL (ENTREZ)`` genes, values = gene effect (Chronos). Read with pandas and stored as float32 (cell lines x genes).

UniProt -> DepMap gene mapping (deterministic)
---------------------------------------------
1. HGNC rows with ``status == "Approved"``; ``uniprot_ids`` split on ``|`` (one row per UniProt accession).
2. Each accession gets the set of Entrez IDs of all HGNC genes that list it.
3. The candidates are intersected with the Entrez IDs of the DepMap columns. Matching is on Entrez, never on the
   symbol, because DepMap symbols can lag HGNC renames.
4. Exactly one candidate in DepMap -> ``mapped``. None -> ``not_in_depmap`` (or ``not_in_hgnc`` if step 2 found
   nothing). Several -> ``ambiguous`` (the protein is encoded by several paralogous gene copies, e.g. replication-
   dependent histones; picking one copy's knockout profile would be arbitrary), so it is not scored.
5. A pair is scorable only if both proteins are mapped to two *different* DepMap genes.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

HGNC_COLUMNS = ["symbol", "status", "entrez_id", "uniprot_ids"]


def load_depmap(path) -> tuple[np.ndarray, pd.Index]:
    """(X float32 cell lines x genes, gene columns 'SYMBOL (ENTREZ)')."""
    df = pd.read_csv(path, index_col=0)
    genes = df.columns
    X = df.values.astype(np.float32)
    return X, genes


def load_hgnc(path) -> pd.DataFrame:
    """Approved HGNC rows (all columns read as strings)."""
    h = pd.read_csv(path, sep="\t", dtype=str, usecols=lambda c: c in HGNC_COLUMNS)
    return h[h.status == "Approved"].reset_index(drop=True)


def parse_depmap_column(col: str) -> tuple[str, str]:
    """'A1BG (1)' -> ('A1BG', '1')."""
    m = re.match(r"^(.*) \((\d+)\)$", col)
    if not m:
        raise ValueError(f"unexpected DepMap column {col!r}")
    return m.group(1), m.group(2)


def build_uniprot_map(uniprot_ids, genes: pd.Index, hgnc: pd.DataFrame) -> pd.DataFrame:
    """One row per UniProt accession: status, gene (DepMap column or None), gene_idx (column index or -1)."""
    m = hgnc[["entrez_id", "uniprot_ids"]].dropna().copy()
    m["uniprot"] = m.uniprot_ids.str.split("|")
    m = m.explode("uniprot")
    cand = m.groupby("uniprot").entrez_id.apply(lambda s: sorted(set(s)))
    ent2col = {parse_depmap_column(c)[1]: c for c in genes}
    col2idx = {c: i for i, c in enumerate(genes)}
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
        rows.append(dict(uniprot=u, status=status, gene=col, gene_idx=col2idx[col] if col else -1))
    df = pd.DataFrame(rows, columns=["uniprot", "status", "gene", "gene_idx"]).set_index("uniprot")
    for st in ("not_in_hgnc", "not_in_depmap", "ambiguous"):
        ids = df.index[df.status == st].tolist()
        if ids:
            log.info("UniProt %s (%d): %s", st, len(ids), ids)
    return df


def gene_annotation(genes: pd.Index, hgnc: pd.DataFrame) -> pd.DataFrame:
    """Per DepMap column: entrez, symbol (HGNC approved symbol, else the DepMap symbol), uniprot (HGNC
    ``uniprot_ids``, '|'-separated, may be empty)."""
    h = hgnc.dropna(subset=["entrez_id"]).drop_duplicates("entrez_id").set_index("entrez_id")
    parsed = [parse_depmap_column(c) for c in genes]
    ent = [p[1] for p in parsed]
    sym = h.symbol.reindex(ent).values
    uni = h.uniprot_ids.reindex(ent).values
    return pd.DataFrame(dict(gene=list(genes), entrez=ent,
                             symbol=[s if isinstance(s, str) else p[0] for s, p in zip(sym, parsed)],
                             uniprot=[u if isinstance(u, str) else "" for u in uni]))


def read_pairs(path) -> pd.DataFrame:
    """Two tab-separated UniProt columns, no header (extra columns ignored, blank lines skipped)."""
    a, b = [], []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            f = line.rstrip("\r\n").split("\t")
            if len(f) < 2 or not f[0].strip() or not f[1].strip():
                raise ValueError(f"{path}:{n}: expected two tab-separated UniProt IDs")
            a.append(f[0].strip())
            b.append(f[1].strip())
    return pd.DataFrame(dict(uniprot_a=a, uniprot_b=b), dtype=object)


def map_pairs(pairs: pd.DataFrame, genes: pd.Index, hgnc: pd.DataFrame):
    """Gene indices (ia, ib; -1 = unmapped) for a frame with uniprot_a / uniprot_b, and the UniProt map."""
    ids = pd.unique(pairs[["uniprot_a", "uniprot_b"]].values.ravel())
    umap = build_uniprot_map(ids, genes, hgnc)
    ia = umap.gene_idx.reindex(pairs.uniprot_a).values.astype(np.int64)
    ib = umap.gene_idx.reindex(pairs.uniprot_b).values.astype(np.int64)
    return ia, ib, umap


def ensure_exists(path, what: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p} (see README, 'Data')")
    return p
