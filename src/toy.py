"""Synthetic universe with planted complexes, for tests and for exercising scripts/final_eval.py without real IFT.

Proteins = genes (one column each). Complexes of 2-12 proteins (+ a few singletons and one large 60-protein
"giant" complex that build_clusters splits by Louvain); positives = a spanning chain plus a random 60% of
within-complex pairs; negatives = all other pairs. X (samples x genes): lineage-block sample covariance shared by
all genes + a complex-specific latent factor for members (strength varies by complex) + noise; a block of
missing values mimics library missingness.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src import clusters as C
from src.universe import PairUniverse, annotate, common_essential


def make_toy(seed: int = 0, n_complex: int = 70, n_single: int = 15, n_samples: int = 80, giant: int = 60,
             signal: float = 0.8):
    rng = np.random.default_rng(seed)
    sizes = list(rng.integers(2, 13, n_complex)) + ([giant] if giant else [])
    members, comp = [], []
    pid = 0
    for c, k in enumerate(sizes):
        for _ in range(k):
            members.append(f"T{pid:05d}")
            comp.append(c)
            pid += 1
    for _ in range(n_single):
        members.append(f"T{pid:05d}")
        comp.append(-1 - pid)
        pid += 1
    comp = np.array(comp)
    G = len(members)
    blocks = np.repeat(np.arange(8), int(np.ceil(n_samples / 8)))[:n_samples]
    Sig = 0.3 * np.eye(n_samples) + 0.7 * (blocks[:, None] == blocks[None, :])
    L = np.linalg.cholesky(Sig)
    X = L @ rng.normal(size=(n_samples, G))
    strength = {c: signal * rng.uniform(0.2, 1.5) for c in set(comp) if c >= 0}
    for c, st in strength.items():
        idx = np.flatnonzero(comp == c)
        if c == len(sizes) - 1 and giant:            # giant: sub-modules so Louvain has structure
            for sub in np.array_split(idx, 4):
                X[:, sub] += st * rng.normal(size=(n_samples, 1))
            X[:, idx] += 0.3 * st * rng.normal(size=(n_samples, 1))
        else:
            X[:, idx] += st * rng.normal(size=(n_samples, 1))
    X -= 0.8 * (rng.random(G) < 0.3)[None, :]        # some "essential" genes
    X[:10, rng.choice(G, 15, replace=False)] = np.nan
    rows = []
    for c in sorted(set(comp[comp >= 0])):
        idx = np.flatnonzero(comp == c)
        if c == len(sizes) - 1 and giant:
            subs = np.array_split(idx, 4)
            chain = {(subs[j][-1], subs[j + 1][0]) for j in range(3)}
            for sub in subs:
                chain |= {(sub[i - 1], sub[i]) for i in range(1, len(sub))}
            for a, b in itertools.combinations(idx, 2):
                same = any(a in s_ and b in s_ for s_ in subs)
                if (a, b) in chain or (same and rng.random() < 0.6):
                    rows.append((a, b))
        else:
            chain = {(idx[i - 1], idx[i]) for i in range(1, len(idx))}
            for a, b in itertools.combinations(idx, 2):
                if (a, b) in chain or rng.random() < 0.6:
                    rows.append((a, b))
    pos = set(rows)
    ia, ib = np.triu_indices(G, 1)
    lab = np.array([(a, b) in pos for a, b in zip(ia, ib)], dtype=np.int8)
    names = np.array(members)
    pairs = pd.DataFrame(dict(a=names[ia], b=names[ib], label=lab, ia=ia, ib=ib))
    pairs["pair_id"] = np.arange(len(pairs))
    cl = C.build_clusters(pairs, max_size=40, seed=seed)
    assign, _ = C.assign_sides(pairs, cl, seed=seed)
    p = annotate(pairs, assign, common_essential(X))
    largest = int(assign.loc[assign.component_size.idxmax(), "component"])
    u = PairUniverse(p, "toy", f=1.0, largest_component=largest,
                     meta=dict(f_ift=float((assign.side == "ift").mean()), f_dev=float((assign.side == "dev").mean())))
    return u, X, assign


def toy_dev(u: PairUniverse) -> PairUniverse:
    d = u.subset(np.flatnonzero(u.col("side") == "dev"), name="toy_dev")
    d.f = u.meta["f_dev"]
    return d
