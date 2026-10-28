"""Pair-level split generators. All deterministic given (pair table, scheme, n_folds, seed).

Input: a pair table with columns a, b (protein IDs) and label (0/1). Splits are computed on exactly the
rows passed in (the scripts pass the *scorable* pairs, so unmapped proteins never influence the partition).

Schemes
-------
random  : sklearn StratifiedKFold(shuffle=True, random_state=seed) over pairs. Proteins are shared between
          train and test (pair-level split; optimistic when per-protein propensity is learnable).
gene    : Park & Marcotte (2012) C3. Proteins are randomly permuted (seed) and split into K equal groups.
          Fold k test = pairs with BOTH proteins in group k; train = pairs with BOTH proteins outside group k
          (this includes pairs spanning two different non-k groups). Pairs with exactly one protein in group k
          are C2 for fold k: excluded from both train and test of that fold, and reported.
complex : proteins grouped by connected components of the positive-pair graph (a protein with no positive
          pair is its own component). Components are assigned to folds greedily, largest first (n positive
          pairs, then n proteins; ties broken by a seeded random permutation), each to the fold with the
          smallest normalised load pos_k/T_pos + prot_k/T_prot. Test/train/C2 then as in `gene`. By
          construction no positive edge crosses folds, so C2 pairs are all negatives.

cluster : DESIGN v2 §3.3 primary dev CV. Proteins grouped by a caller-supplied protein -> cluster map (§3.1
          clusters from src.clusters). Clusters are visited in a seeded RANDOM order (not largest first) and each is
          put in the fold with the smallest normalised load pos_k/T_pos + prot_k/T_prot after adding it (seeded
          tie-break). Positives counted are within-cluster positives of the pairs passed in. Test/train/C2 as in
          `gene`; positive edges between clusters in different folds are C2 for both folds (never evaluated).
component: §3.4(a) strict connected-component grouping; same randomised balanced algorithm with the
          protein -> connected-component map (computed from the pairs passed in when not supplied).

Inner validation: ``inner_validation(pairs, split, k, seed)`` re-applies the same scheme to fold k's training
pairs only and returns (inner_train_mask, val_mask) over the full table, using inner fold 0 as validation.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedKFold

SCHEMES = ("random", "gene", "complex", "cluster", "component")


@dataclass
class PairSplit:
    scheme: str
    n_folds: int
    seed: int
    test_fold: np.ndarray                 # per pair: fold in which the pair is a test pair, -1 = never (C2 cross)
    group_a: np.ndarray | None = None     # per pair: protein group of a (gene/complex schemes)
    group_b: np.ndarray | None = None
    protein_group: dict = field(default_factory=dict)
    cluster_map: dict | None = None       # protein -> cluster (cluster/component schemes; reused by inner split)

    def test_mask(self, k: int) -> np.ndarray:
        return self.test_fold == k

    def train_mask(self, k: int) -> np.ndarray:
        if self.group_a is None:
            return self.test_fold != k
        return (self.group_a != k) & (self.group_b != k)

    def c2_mask(self, k: int) -> np.ndarray:
        if self.group_a is None:
            return np.zeros(len(self.test_fold), bool)
        return (self.group_a == k) ^ (self.group_b == k)

    def stats(self, labels) -> list[dict]:
        y = np.asarray(labels).astype(bool)
        out = []
        for k in range(self.n_folds):
            te, tr, c2 = self.test_mask(k), self.train_mask(k), self.c2_mask(k)
            d = dict(fold=k, test_pos=int((te & y).sum()), test_neg=int((te & ~y).sum()),
                     train_pos=int((tr & y).sum()), train_neg=int((tr & ~y).sum()),
                     c2_pos=int((c2 & y).sum()), c2_neg=int((c2 & ~y).sum()))
            if self.protein_group:
                d["test_proteins"] = int(sum(g == k for g in self.protein_group.values()))
            out.append(d)
        return out

    def fingerprint(self) -> str:
        """SHA-256 of the fold assignment (for recording identical test pairs without committing pairs)."""
        h = hashlib.sha256(self.test_fold.astype(np.int16).tobytes())
        if self.group_a is not None:
            h.update(self.group_a.astype(np.int16).tobytes())
            h.update(self.group_b.astype(np.int16).tobytes())
        return h.hexdigest()


def _random_split(pairs, n_folds, seed):
    y = pairs.label.values
    tf = np.full(len(pairs), -1, np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        tf[te] = k
    return PairSplit("random", n_folds, seed, tf)


def _from_protein_groups(pairs, groups: dict, scheme, n_folds, seed):
    ga = pairs.a.map(groups).values.astype(np.int64)
    gb = pairs.b.map(groups).values.astype(np.int64)
    tf = np.where(ga == gb, ga, -1)
    return PairSplit(scheme, n_folds, seed, tf, ga, gb, dict(groups))


def proteins_of(pairs) -> np.ndarray:
    return np.array(sorted(set(pairs.a) | set(pairs.b)))


def _gene_split(pairs, n_folds, seed):
    prots = proteins_of(pairs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(prots))
    groups = {}
    for k, chunk in enumerate(np.array_split(perm, n_folds)):
        for i in chunk:
            groups[prots[i]] = k
    return _from_protein_groups(pairs, groups, "gene", n_folds, seed)


def positive_components(pairs) -> pd.DataFrame:
    """Connected components of the positive graph. Returns per-protein component id, sorted protein order."""
    prots = proteins_of(pairs)
    ix = {p: i for i, p in enumerate(prots)}
    pos = pairs[pairs.label == 1]
    r = np.array([ix[a] for a in pos.a]), np.array([ix[b] for b in pos.b])
    A = sp.coo_matrix((np.ones(len(pos)), r), shape=(len(prots),) * 2)
    _, lab = connected_components(A, directed=False)
    return pd.DataFrame(dict(protein=prots, component=lab))


def _complex_split(pairs, n_folds, seed):
    comp = positive_components(pairs)
    pc = dict(zip(comp.protein, comp.component))
    pos = pairs[pairs.label == 1]
    npos = pd.Series(pos.a.map(pc).values).value_counts()
    cs = comp.groupby("component").size().rename("n_prot").to_frame()
    cs["n_pos"] = npos.reindex(cs.index).fillna(0).astype(int)
    rng = np.random.default_rng(seed)
    cs["tie"] = rng.permutation(len(cs))
    cs = cs.sort_values(["n_pos", "n_prot", "tie"], ascending=[False, False, True])
    T_pos, T_prot = max(cs.n_pos.sum(), 1) / n_folds, cs.n_prot.sum() / n_folds
    load_pos, load_prot = np.zeros(n_folds), np.zeros(n_folds)
    fold_order = rng.permutation(n_folds)            # seeded tie-break between equally loaded folds
    comp_fold = {}
    for c, row in cs.iterrows():
        score = (load_pos + row.n_pos) / T_pos + (load_prot + row.n_prot) / T_prot
        k = fold_order[np.argmin(score[fold_order])]
        comp_fold[c] = int(k)
        load_pos[k] += row.n_pos
        load_prot[k] += row.n_prot
    groups = {p: comp_fold[c] for p, c in pc.items()}
    return _from_protein_groups(pairs, groups, "complex", n_folds, seed)


def _randomised_balanced(pairs, cmap: dict, scheme, n_folds, seed):
    prots = proteins_of(pairs)
    cid = np.array([cmap[p] for p in prots])
    pos = pairs[pairs.label == 1]
    ca, cb = pos.a.map(cmap).values, pos.b.map(cmap).values
    within = pd.Series(ca[ca == cb]).value_counts()
    ids = np.unique(cid)
    n_prot = pd.Series(cid).value_counts().reindex(ids).values.astype(float)
    n_pos = within.reindex(ids).fillna(0).values.astype(float)
    T_pos, T_prot = max(n_pos.sum(), 1) / n_folds, n_prot.sum() / n_folds
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    fold_order = rng.permutation(n_folds)
    load_pos, load_prot = np.zeros(n_folds), np.zeros(n_folds)
    cfold = {}
    for i in order:
        score = (load_pos + n_pos[i]) / T_pos + (load_prot + n_prot[i]) / T_prot
        k = fold_order[np.argmin(score[fold_order])]
        cfold[ids[i]] = int(k)
        load_pos[k] += n_pos[i]
        load_prot[k] += n_prot[i]
    groups = {p: cfold[c] for p, c in zip(prots, cid)}
    sp = _from_protein_groups(pairs, groups, scheme, n_folds, seed)
    sp.cluster_map = {p: cmap[p] for p in prots}
    return sp


def make_split(pairs: pd.DataFrame, scheme: str, n_folds: int = 5, seed: int = 0, clusters: dict | None = None) -> PairSplit:
    """clusters: protein -> group id, required for scheme 'cluster', optional for 'component'."""
    pairs = pairs.reset_index(drop=True)
    if scheme == "cluster":
        if clusters is None:
            raise ValueError("scheme 'cluster' needs a protein -> cluster map")
        return _randomised_balanced(pairs, clusters, scheme, n_folds, seed)
    if scheme == "component":
        if clusters is None:
            comp = positive_components(pairs)
            clusters = dict(zip(comp.protein, comp.component))
        return _randomised_balanced(pairs, clusters, scheme, n_folds, seed)
    if scheme == "random":
        return _random_split(pairs, n_folds, seed)
    if scheme == "gene":
        return _gene_split(pairs, n_folds, seed)
    if scheme == "complex":
        return _complex_split(pairs, n_folds, seed)
    raise ValueError(scheme)


def inner_validation(pairs: pd.DataFrame, split: PairSplit, k: int, seed: int, n_inner: int = 4):
    """Carve a validation set out of fold k's training pairs with the same scheme.

    Returns (inner_train_mask, val_mask) as boolean arrays over the rows of `pairs`. For gene/complex schemes,
    inner validation pairs have both proteins in the inner held-out group and inner training pairs have both
    proteins outside it, so they are protein-disjoint as well."""
    pairs = pairs.reset_index(drop=True)
    tr_idx = np.flatnonzero(split.train_mask(k))
    sub = pairs.iloc[tr_idx].reset_index(drop=True)
    inner = make_split(sub, split.scheme, n_inner, seed, clusters=split.cluster_map)
    itr = np.zeros(len(pairs), bool)
    val = np.zeros(len(pairs), bool)
    itr[tr_idx[inner.train_mask(0)]] = True
    val[tr_idx[inner.test_mask(0)]] = True
    return itr, val
