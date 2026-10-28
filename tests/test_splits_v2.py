"""§3.3 cluster-grouped dev CV: protein disjointness, seed dependence, no cross-fold positives evaluated, inner split."""
import numpy as np
import pytest

from src.config import SEEDS
from src.evaluate import make_dev_split
from src.splits import inner_validation
from src.toy import make_toy, toy_dev


@pytest.fixture(scope="module")
def dev():
    u, _, _ = make_toy(seed=SEEDS[0])
    return toy_dev(u)


def _prots(p):
    return set(p.a) | set(p.b)


@pytest.mark.parametrize("scheme", ["cluster", "component", "gene"])
def test_disjoint_and_complete(dev, scheme):
    sp = make_dev_split(dev, scheme, SEEDS[0])
    p = dev.pairs
    for k in range(5):
        tr, te, c2 = sp.train_mask(k), sp.test_mask(k), sp.c2_mask(k)
        assert te.any() and tr.any()
        assert _prots(p[tr]).isdisjoint(_prots(p[te]))
        assert not (tr & te).any() and (tr | te | c2).all()


def test_cluster_folds_keep_clusters_whole(dev):
    sp = make_dev_split(dev, "cluster", SEEDS[0])
    cm = dev.cluster_map()
    fold_of = {}
    for prot, g in sp.protein_group.items():
        fold_of.setdefault(cm[prot], set()).add(g)
    assert all(len(v) == 1 for v in fold_of.values())
    # positives between clusters in different folds are never test pairs
    p = dev.pairs
    cross = (p.label.values == 1) & (sp.group_a != sp.group_b)
    assert (sp.test_fold[cross] == -1).all()


def test_seeds_give_different_assignments(dev):
    fps = {make_dev_split(dev, "cluster", s).fingerprint() for s in SEEDS}
    assert len(fps) == len(SEEDS)
    assert make_dev_split(dev, "cluster", SEEDS[0]).fingerprint() == make_dev_split(dev, "cluster", SEEDS[0]).fingerprint()


def test_balance(dev):
    st = make_dev_split(dev, "cluster", SEEDS[0]).stats(dev.y)
    tp = np.array([d["test_pos"] for d in st])
    assert tp.min() > 0 and tp.max() < 2.5 * tp.mean()


@pytest.mark.parametrize("scheme", ["cluster", "component"])
def test_inner_validation_same_type(dev, scheme):
    sp = make_dev_split(dev, scheme, SEEDS[0])
    itr, val = inner_validation(dev.pairs, sp, 1, SEEDS[1])
    p = dev.pairs
    assert val.any() and itr.any() and not (itr & val).any()
    assert not ((itr | val) & ~sp.train_mask(1)).any()
    assert _prots(p[val]).isdisjoint(_prots(p[itr]))
    assert _prots(p[val]).isdisjoint(_prots(p[sp.test_mask(1)]))
    cm = dev.cluster_map() if scheme == "cluster" else dev.component_map()
    assert {cm[x] for x in _prots(p[val])}.isdisjoint({cm[x] for x in _prots(p[itr])})
