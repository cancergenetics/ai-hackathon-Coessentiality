"""Parsing helpers; mapping/pair-table invariants on the real files (skipped if data absent)."""
import numpy as np
import pytest

from src.config import HGNC_FILE, POS_FILE
from src.data import parse_depmap_column, parse_location


def test_parse_location():
    assert parse_location("19q13.43") == dict(chrom="19", arm="q", band="13.43", band_num=13.43)
    d = parse_location("1p36.33-p36.32"); assert (d["chrom"], d["arm"], d["band"]) == ("1", "p", "36.33")
    d = parse_location("Xp22.33 and Yp11.2"); assert d["chrom"] == "X"
    assert parse_location("mitochondria")["chrom"] == "MT"
    d = parse_location("6p21.3 alternate reference locus"); assert d["band"] == "21.3"
    d = parse_location("10 not on reference assembly"); assert d["chrom"] == "10" and d["arm"] is None
    assert parse_location("unplaced")["chrom"] is None
    assert parse_location(np.nan)["chrom"] is None


def test_parse_depmap_column():
    assert parse_depmap_column("A1BG (1)") == ("A1BG", "1")
    assert parse_depmap_column("HLA-A (3105)") == ("HLA-A", "3105")
    with pytest.raises(ValueError):
        parse_depmap_column("A1BG")


@pytest.mark.skipif(not (HGNC_FILE.exists() and POS_FILE.exists()), reason="raw data absent")
def test_pair_table_invariants():
    from src.data import load_pairs
    p, umap = load_pairs()
    assert (p.a < p.b).all()
    assert not p.duplicated(["a", "b"]).any()
    assert set(p.label.unique()) == {0, 1}
    assert ((p.ia >= 0) == p.gene_a.notna()).all()
    assert (p.scorable == ((p.ia >= 0) & (p.ib >= 0) & (p.ia != p.ib))).all()
    assert (umap.loc[umap.status == "mapped", "gene_idx"] >= 0).all()
    assert (umap.loc[umap.status != "mapped", "gene_idx"] == -1).all()
