"""All-pairs matrices (coessential.all_pairs) on toy data: values, symmetry, gene set, pair table."""
import numpy as np
import pandas as pd

from coessential import all_pairs
from coessential.method import CoessentialityModel
from toy import TOY, modules


def test_all_pairs_matrices_match_pair_scores(tmp_path):
    X, _ = modules(seed=3)
    X[:30, 7] = np.nan                                       # a low-coverage (non-background) gene
    G = X.shape[1]
    m = CoessentialityModel(X, TOY)
    genes = pd.Index([f"S{i} ({1000 + i})" for i in range(G)])
    hg = tmp_path / "hgnc.txt"
    rows = [(f"HGNC:{i}", f"S{i}", "Approved", str(1000 + i), "" if i in (2, 50) else f"Q{i}") for i in range(G)]
    pd.DataFrame(rows, columns=["hgnc_id", "symbol", "status", "entrez_id", "uniprot_ids"]).to_csv(
        hg, sep="\t", index=False)
    tab, counts = all_pairs.protein_genes(genes, hg, m.nan_frac, TOY.max_nan)
    assert counts["n_excluded_no_uniprot"] == 2 and counts["n_protein_genes"] == G - 2
    assert tab.low_coverage.sum() == 1 and tab.depmap_index[tab.low_coverage == 1].tolist() == [7]
    gidx = tab.depmap_index.values
    mats, pairs, st = all_pairs.score_all_pairs(m, gidx, threshold=2.0)
    n = len(gidx)
    iu = np.triu_indices(n, 1)
    z, Z = m.zavg(gidx[iu[0]], gidx[iu[1]])
    assert np.allclose(mats["zavg_BEF"][iu], z, atol=1e-5)
    for c in "BEF":
        assert np.allclose(mats[f"z_{c}"][iu].astype(float), Z[c], atol=1e-2 + 1e-3 * np.abs(Z[c]))
    for M in mats.values():
        assert np.isnan(np.diag(M)).all()
        assert np.array_equal(M, M.T, equal_nan=True)
    assert np.array_equal(np.vstack([m.zavg_block(r) for r in m._blocks()])[np.ix_(gidx, gidx)][iu],
                          (sum(np.vstack([m.z_block(r)[c] for r in m._blocks()]) for c in "BEF") / 3.0)[np.ix_(gidx, gidx)][iu])
    assert st["pair_table_rows"] == int((z >= 2.0).sum()) and (pairs["a"] < pairs["b"]).all()
    assert (np.diff(pairs["zavg_BEF"]) <= 0).all()
    pt = all_pairs.pair_table(pairs, tab)
    assert pt["rank"].tolist() == list(range(1, len(pt) + 1))
    assert pt.low_coverage_any.sum() == ((pairs["a"] == 6) | (pairs["b"] == 6)).sum()   # gene 7 -> protein index 6
