"""Toy gene-effect matrices with planted structure (known answers)."""
import math

import numpy as np

from coessential.method import Config

# small-data configuration: no PC removal (the planted modules ARE the top PCs), ~10 neighbours, 16-d embedding
TOY = Config(k=0, tau=0.1, q=2.5, d=16, n_null=5000)


def modules(n_lines=80, n_modules=6, size=15, n_noise=300, loading=1.0, seed=0):
    """Genes 0..n_modules*size-1 form modules (shared latent factor + noise); the rest are pure noise.
    Returns (X lines x genes, module label per gene, -1 = noise)."""
    rng = np.random.default_rng(seed)
    G = n_modules * size + n_noise
    X = rng.standard_normal((n_lines, G))
    lab = np.full(G, -1)
    for m in range(n_modules):
        cols = np.arange(m * size, (m + 1) * size)
        X[:, cols] += loading * rng.standard_normal((n_lines, 1))
        lab[cols] = m
    return X, lab


def shared_neighbour_toy(n_lines=300, seed=1):
    """Module M = genes 0..39 (factor f). Genes a = 40 and b = 41 load weakly on f with independent noise: they are
    moderately correlated with each other and share M as neighbours. Genes c = 42 and d = 43 load equally weakly on a
    private factor h: same direct correlation, but no common third-party neighbours. Genes 44.. are noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_lines, 424))
    f, h = rng.standard_normal(n_lines), rng.standard_normal(n_lines)
    X[:, :40] += f[:, None]
    X[:, 40:42] += 0.8 * f[:, None]
    X[:, 42:44] += 0.8 * h[:, None]
    return X


def all_pairs(G):
    iu = np.triu_indices(G, 1)
    return iu[0].astype(np.int64), iu[1].astype(np.int64)


def auroc(pos, neg):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


INF = math.inf
