"""Split generators: disjointness, no positive edge crossing, determinism, inner validation."""
import itertools

import numpy as np
import pandas as pd
import pytest

from src.config import SEEDS
from src.splits import inner_validation, make_split


@pytest.fixture(scope="module")
def toy_pairs():
    """60 'complexes' of 2-12 proteins + 30 singleton proteins; positives = random 70% of within-complex
    edges (spanning tree kept so each complex is one component); negatives = all other pairs."""
    rng = np.random.default_rng(SEEDS[0])
    prots, comp = [], {}
    pid = 0
    for c in range(60):
        for _ in range(rng.integers(2, 13)):
            comp[f"P{pid:04d}"] = c; pid += 1
    for _ in range(30):
        comp[f"P{pid:04d}"] = -1 - pid; pid += 1
    names = sorted(comp)
    rows = []
    for c in set(comp.values()):
        members = sorted(p for p in names if comp[p] == c)
        tree = {(members[i - 1], members[i]) for i in range(1, len(members))}
        for a, b in itertools.combinations(members, 2):
            if (a, b) in tree or rng.random() < 0.7:
                rows.append((a, b, 1))
    pos = {(a, b) for a, b, _ in rows}
    for a, b in itertools.combinations(names, 2):
        if (a, b) not in pos:
            rows.append((a, b, 0))
    return pd.DataFrame(rows, columns=["a", "b", "label"])


def _prots(df):
    return set(df.a) | set(df.b)


@pytest.mark.parametrize("scheme", ["gene", "complex"])
def test_protein_disjoint(toy_pairs, scheme):
    sp = make_split(toy_pairs, scheme, 5, SEEDS[0])
    for k in range(5):
        tr, te = toy_pairs[sp.train_mask(k)], toy_pairs[sp.test_mask(k)]
        assert len(te) > 0 and len(tr) > 0
        assert _prots(tr).isdisjoint(_prots(te))
        assert not (sp.train_mask(k) & sp.test_mask(k)).any()
        assert not (sp.c2_mask(k) & (sp.train_mask(k) | sp.test_mask(k))).any()
        # every pair is exactly one of train / test / C2 for fold k
        assert (sp.c2_mask(k) | sp.train_mask(k) | sp.test_mask(k)).all()


def test_complex_no_positive_crossing(toy_pairs):
    sp = make_split(toy_pairs, "complex", 5, SEEDS[0])
    pos = toy_pairs.label.values == 1
    assert (sp.group_a[pos] == sp.group_b[pos]).all()
    assert (sp.test_fold[pos] >= 0).all()
    st = sp.stats(toy_pairs.label)
    assert all(d["c2_pos"] == 0 for d in st)
    # balance: positives per fold within a factor 2 of the mean for this toy
    tp = np.array([d["test_pos"] for d in st])
    assert tp.max() < 2 * tp.mean()


def test_random_split_stratified(toy_pairs):
    sp = make_split(toy_pairs, "random", 5, SEEDS[0])
    assert (sp.test_fold >= 0).all()
    st = sp.stats(toy_pairs.label)
    tp = [d["test_pos"] for d in st]
    assert max(tp) - min(tp) <= 1
    for k in range(5):
        assert not (sp.train_mask(k) & sp.test_mask(k)).any()


@pytest.mark.parametrize("scheme", ["random", "gene", "complex"])
def test_deterministic(toy_pairs, scheme):
    a = make_split(toy_pairs, scheme, 5, SEEDS[0]).fingerprint()
    b = make_split(toy_pairs, scheme, 5, SEEDS[0]).fingerprint()
    c = make_split(toy_pairs, scheme, 5, SEEDS[1]).fingerprint()
    assert a == b and a != c


@pytest.mark.parametrize("scheme", ["random", "gene", "complex"])
def test_inner_validation(toy_pairs, scheme):
    sp = make_split(toy_pairs, scheme, 5, SEEDS[0])
    itr, val = inner_validation(toy_pairs, sp, 2, SEEDS[1])
    assert val.sum() > 0 and itr.sum() > 0
    assert not (itr & val).any()
    assert not ((itr | val) & ~sp.train_mask(2)).any()
    if scheme != "random":
        te = toy_pairs[sp.test_mask(2)]
        assert _prots(toy_pairs[val]).isdisjoint(_prots(toy_pairs[itr]))
        assert _prots(toy_pairs[val]).isdisjoint(_prots(te))
    if scheme == "complex":
        assert (toy_pairs[val].label == 1).sum() > 0
