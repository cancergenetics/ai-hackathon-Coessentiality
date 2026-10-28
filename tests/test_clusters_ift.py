"""§3.1-3.2 clusters and IFT/dev assignment (toy + real assignment file), unlabelled pairs, end-to-end ID chain."""
import itertools

import numpy as np
import pandas as pd
import pytest

from src import clusters as C
from src.config import CACHE, DEPMAP, NEG_FILE, POS_FILE, SEEDS
from src.toy import make_toy

HAVE_CACHE = (CACHE / "pairs.pkl").exists() and C.ASSIGN_FILE.exists()


# ------------------------------------------------------------------------------------------------- toy
@pytest.fixture(scope="module")
def toy():
    return make_toy(seed=SEEDS[0])


def test_louvain_splits_large_components(toy):
    u, X, assign = toy
    assert assign.cluster_size.max() <= C.MAX_CLUSTER_SIZE
    assert assign.loc[assign.louvain_split, "component_size"].min() > C.MAX_CLUSTER_SIZE
    # deterministic
    pairs = u.pairs[["a", "b", "label"]]
    cl1, cl2 = C.build_clusters(pairs), C.build_clusters(pairs)
    assert cl1.equals(cl2)
    st = C.cluster_edge_stats(pairs, cl1)
    assert st["cross_community_positive_edges"] > 0          # the planted giant has inter-module edges


def test_toy_sides_disjoint_and_stratified(toy):
    u, X, assign = toy
    assert set(assign.side) == {"dev", "ift"}
    assert assign.groupby("cluster").side.nunique().max() == 1      # clusters never split across sides
    s = assign.drop_duplicates("cluster")
    for st, d in s.groupby("stratum"):
        if st != "no_pos":
            assert (d.side == "ift").any() or len(d) < 3, st


# ------------------------------------------------------------------------------------------------ real
@pytest.fixture(scope="module")
def real():
    allp = pd.read_pickle(CACHE / "pairs.pkl")
    return allp, C.load_assignment()


@pytest.mark.skipif(not HAVE_CACHE, reason="caches absent")
def test_assignment_hash_and_regeneration(real):
    import os, subprocess, sys
    code = ("import pandas as pd; from src import clusters as C; from src.config import SEEDS, CACHE; "
            "r, _ = C.build_all(pd.read_pickle(CACHE / 'pairs.pkl'), seed=SEEDS[0]); print(C.sha256_bytes(C._csv_bytes(r)))")
    env = dict(os.environ, PYTHONHASHSEED="0")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True,
                         cwd=C.RESULTS.parent).stdout.split()[-1]
    want = C.SHA_FILE.read_text().split()[0]
    assert out == want
    committed = C.RESULTS / "splits" / "protein_assignment.csv"
    assert C.sha256_bytes(committed.read_bytes()) == want


@pytest.mark.skipif(not HAVE_CACHE, reason="caches absent")
def test_ift_dev_protein_disjoint(real):
    allp, assign = real
    assert assign.protein.is_unique
    assert set(assign.side) == {"dev", "ift"}
    from src.universe import load_dev_universe
    u = load_dev_universe()
    dev_prots = set(u.pairs.a) | set(u.pairs.b)
    ift_prots = set(assign.protein[assign.side == "ift"])
    assert dev_prots.isdisjoint(ift_prots)
    assert (u.col("side") == "dev").all()


@pytest.mark.skipif(not HAVE_CACHE, reason="caches absent")
def test_cross_positive_edges_counted_in_neither(real):
    allp, assign = real
    sc = allp[allp.scorable].reset_index(drop=True)
    sides = C.pair_sides(sc, assign[assign.mapped])
    cross_pos = sc[(sides == "cross") & (sc.label == 1)]
    assert len(cross_pos) > 0
    from src.universe import load_dev_universe
    u = load_dev_universe()
    key = set(zip(u.pairs.a, u.pairs.b))
    assert not any(k in key for k in zip(cross_pos.a, cross_pos.b))
    summ = C.summary(sc, allp, assign)
    assert summ["pairs"]["dev"]["pos"] + summ["pairs"]["ift"]["pos"] + summ["pairs"]["cross"]["pos"] == len(sc[sc.label == 1])
    assert summ["pairs"]["cross"]["pos"] == len(cross_pos)
    assert summ["pairs"]["dev"]["pos"] == int(u.y.sum())
    assert summ["ift_max_cluster_share_of_positives"] <= 0.30
    assert summ["pairs"]["ift"]["pos"] >= 700


@pytest.mark.skipif(not (POS_FILE.exists() and (CACHE / "pairs.pkl").exists()), reason="raw data absent")
def test_unlabelled_pairs_never_negatives():
    pos = pd.read_csv(POS_FILE, sep="\t", header=None, dtype=str)
    neg = pd.read_csv(NEG_FILE, sep="\t", header=None, dtype=str)
    lab = {frozenset(r) for r in pos.values} | {frozenset(r) for r in neg.values}
    prots = sorted(set(pos.values.ravel()) | set(neg.values.ravel()))
    n_all = len(prots) * (len(prots) - 1) // 2
    assert n_all - len(lab) == 27046
    allp = pd.read_pickle(CACHE / "pairs.pkl")
    assert len(allp) == len(lab) == len(pos) + len(neg)
    negset = {frozenset(r) for r in neg.values}
    table_neg = {frozenset(r) for r in allp.loc[allp.label == 0, ["a", "b"]].values}
    assert table_neg == negset                                     # every table negative is a file negative
    # an explicit unlabelled pair is absent from the table in both orientations
    keys = set(zip(allp.a, allp.b))
    rng = np.random.default_rng(SEEDS[0])
    found = 0
    while found < 50:
        a, b = sorted(rng.choice(prots, 2, replace=False))
        if frozenset((a, b)) not in lab:
            assert (a, b) not in keys and (b, a) not in keys
            found += 1


@pytest.mark.skipif(not ((DEPMAP / "CRISPRGeneEffect.csv").exists() and HAVE_CACHE), reason="raw data absent")
def test_end_to_end_id_chain_from_raw_csv():
    """30 random scorable pairs: Pearson recomputed from the raw CSV selected by column NAME equals the cached-matrix
    pipeline result (UniProt -> HGNC -> Entrez -> DepMap column -> cached column index)."""
    from src.baselines import pearson_pairs
    from src.data import load_matrix
    allp = pd.read_pickle(CACHE / "pairs.pkl")
    sc = allp[allp.scorable].reset_index(drop=True)
    rng = np.random.default_rng(SEEDS[0])
    sub = sc.iloc[rng.choice(len(sc), 30, replace=False)]
    cols = sorted(set(sub.gene_a) | set(sub.gene_b))
    header = pd.read_csv(DEPMAP / "CRISPRGeneEffect.csv", nrows=0).columns
    pos = [0] + [i for i, c in enumerate(header) if c in set(cols)]
    raw = pd.read_csv(DEPMAP / "CRISPRGeneEffect.csv", usecols=pos, index_col=0)
    assert set(raw.columns) == set(cols)
    X, samples, genes = load_matrix("CRISPRGeneEffect", mmap=True)
    assert list(raw.index) == list(samples)
    cached = pearson_pairs(X, sub.ia.values, sub.ib.values)["signed"]
    for k, (_, r) in enumerate(sub.iterrows()):
        assert genes[r.ia] == r.gene_a and genes[r.ib] == r.gene_b
        xa, xb = raw[r.gene_a].values, raw[r.gene_b].values
        ok = ~np.isnan(xa) & ~np.isnan(xb)
        assert cached[k] == pytest.approx(np.corrcoef(xa[ok], xb[ok])[0, 1], abs=1e-5)
