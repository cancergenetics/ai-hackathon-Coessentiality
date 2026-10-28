"""Phase-A A7: label-free kNN-graph / diffusion / spectral-embedding layers (same contract as src/layers.py).

Every layer is a deterministic function of the gene-effect matrix X (through a base ``Spectral`` filter of
src/layers.py) and the two gene indices. Nothing here reads labels, the pair table or which genes are labelled.
Each layer is materialised as a full genome-wide score matrix M (genes x genes, float32), computed in fixed row
blocks of ``LayerEngine.block`` genes, so a pair's score never depends on which other pairs are requested
(batch invariance) and genome-wide row moments (row-z) come for free. Pair score = (M_ab + M_ba) / 2 (exactly
symmetric).

Graphs (nodes = all genes; only background (BG, <10% NaN) genes can be neighbours, as in src/layers.py)
------------------------------------------------------------------------------------------------------
``KNNGraph(base, q, sym, weight)``: each gene a keeps its top K = round(q/100 |BG|) BG neighbours (self excluded)
by the balanced base similarity B_aj = (z_aj + z_ja)/2 (``LayerEngine.neighbours(base, balanced=True)``; B is
symmetric). Edge weight w = 1 ('bin'), max(B, 0) ('lin') or max(B, 0)^p ('p<p>', soft threshold).
Symmetrisation of the directed kNN matrix W (rows = genes, columns = BG): 'dir' = W; 'union' = max(W, W');
'mutual' = min(W, W') (edge kept only if each is in the other's top K); 'avg' = (W + W')/2. Non-BG genes are never
neighbours, so their out-edges cannot be reciprocated: their rows keep the directed W edges and they have no
in-edges (they never act as intermediates for other genes). The BG x BG block of every symmetric graph is exactly
symmetric.
``SoftGraph(base, beta)``: dense WGCNA-style soft-threshold adjacency a_aj = clip(S_aj / s_ref, 0, 1)^beta on the raw
base cosine S (signed: negative similarity -> 0), BG columns only, zero diagonal. Whitened cosines are small (null
SD ~0.03), so S is first scaled by s_ref = the ``ref_q`` quantile of S over the fixed 200k genome-wide null pairs
(label-free) so that a in [0, 1] as WGCNA assumes.

Operators / profiles (``GraphLayer.op``)
----------------------------------------
'adj': profile rows = adjacency rows (W or a). For diffusion, N = D^-1/2 A D^-1/2 with D = row sums (symmetric on
BG; symmetric graphs only). 'pow<t>': N^t. 'ppr<alpha>': personalised PageRank / random walk with restart in the
symmetric normalisation, sum_{t>=1} alpha (1-alpha)^t N^t (t = 0 identity term dropped: it only touches the
diagonal, which every score excludes), truncated when the remaining coefficient mass < 1e-3. 'heat<t>': heat kernel
exp(-t (I - N)) = e^-t sum_k t^k/k! N^k, k >= 1, same truncation. Computed by repeated sparse x dense products
(dense G x G float32 iterates). 'lap<d>': Laplacian eigenmaps — the top d non-trivial eigenvectors of N_BG
(largest algebraic, the leading (trivial) one dropped; rows of D^-1/2 U and U have the same cosines); 'dmap<d>t<t>':
diffusion map, eigenvector coordinates scaled by lambda^t. Non-BG genes are embedded by the Nystrom extension
e_g = N_g,BG U / lambda. Eigenvectors from scipy eigsh (ARPACK) with a seeded start vector (SEEDS[0]).

Scores (``GraphLayer.score``)
-----------------------------
'cos': partner-excluded cosine of profile rows P_a, P_b over columns u not in {a, b} (generalises the A4
second-order layer: for the 'dir'/'lin' kNN graph with op 'adj' it equals ``SecondOrder(base, q, 'cos')``).
'kern': the kernel value P_ab itself (diffusion similarity); for a non-BG gene g (no in-edges) its column is filled
from its row, so M is symmetric. 'tom': topological overlap (WGCNA) on the adjacency A with weights in [0, 1]:
(sum_u a_au a_ub + a_ab) / (min(k_a, k_b) + 1 - a_ab), k = row sums, a_ab = max(A_ab, A_ba); kNN weights are scaled
to [0, 1] by the 99th percentile of the non-zero weights (binary graphs: unchanged). 'emb': cosine of embedding
rows (no partner exclusion: coordinates are not genes).
``rowz=True``: genome-wide row-z balancing, (z_ab + z_ba)/2 with row moments over BG \\ {a} (as src/layers.RowZ).
"""
from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from src import layers
from src.config import SEEDS
from src.layers import LayerEngine, Spectral

SYMS = ("dir", "union", "mutual", "avg")
TRUNC = 1e-3                    # diffusion series truncation: remaining coefficient mass
TOM_WREF_Q = 0.99               # kNN weights scaled to [0, 1] by this quantile of non-zero weights (TOM only)


# ------------------------------------------------------------------------------------------ configs
def _weight_ok(w):
    return w in ("bin", "lin") or re.fullmatch(r"p\d+(\.\d+)?", w) is not None


@dataclass(frozen=True)
class KNNGraph:
    base: Spectral
    q: float = 0.25
    sym: str = "union"
    weight: str = "lin"
    kind: ClassVar[str] = "knn"

    def __post_init__(self):
        if self.sym not in SYMS or not _weight_ok(self.weight):
            raise ValueError(self)

    @property
    def name(self):
        return f"knn_{self.sym}_{self.weight}_q{self.q:g}[{self.base.name}]"

    def to_dict(self):
        return dict(kind=self.kind, base=self.base.to_dict(), q=self.q, sym=self.sym, weight=self.weight)


@dataclass(frozen=True)
class SoftGraph:
    base: Spectral
    beta: float = 6.0
    ref_q: float = 0.9999
    kind: ClassVar[str] = "soft"

    @property
    def name(self):
        return f"soft_b{self.beta:g}[{self.base.name}]"

    def to_dict(self):
        return dict(kind=self.kind, base=self.base.to_dict(), beta=self.beta, ref_q=self.ref_q)


Graph = KNNGraph | SoftGraph
_OP = re.compile(r"^(adj|pow(\d+)|ppr([0-9.]+)|heat([0-9.]+)|lap(\d+)|dmap(\d+)t([0-9.]+))$")


def parse_op(op: str) -> dict:
    m = _OP.match(op)
    if not m:
        raise ValueError(f"bad op {op}")
    if op == "adj":
        return dict(kind="adj")
    if m.group(2):
        return dict(kind="pow", t=int(m.group(2)))
    if m.group(3):
        return dict(kind="ppr", alpha=float(m.group(3)))
    if m.group(4):
        return dict(kind="heat", t=float(m.group(4)))
    if m.group(5):
        return dict(kind="lap", d=int(m.group(5)), t=0.0)
    return dict(kind="dmap", d=int(m.group(6)), t=float(m.group(7)))


@dataclass(frozen=True)
class GraphLayer:
    graph: Graph
    op: str = "adj"
    score: str = "cos"
    rowz: bool = False
    kind: ClassVar[str] = "graph"

    def __post_init__(self):
        o = parse_op(self.op)
        if self.score not in ("cos", "kern", "tom", "emb"):
            raise ValueError(self.score)
        emb = o["kind"] in ("lap", "dmap")
        if emb != (self.score == "emb"):
            raise ValueError("score 'emb' <=> op lap/dmap")
        if self.score == "tom" and o["kind"] != "adj":
            raise ValueError("tom is defined on the adjacency")
        if self.score == "kern" and o["kind"] == "adj":
            raise ValueError("kern needs a diffusion op")
        if o["kind"] != "adj" and not (isinstance(self.graph, KNNGraph) and self.graph.sym != "dir"):
            raise ValueError("diffusion / embedding ops need a symmetric kNN graph")

    @property
    def core(self) -> "GraphLayer":
        return GraphLayer(self.graph, self.op, self.score, False)

    @property
    def name(self):
        n = f"{self.score}_{self.op}[{self.graph.name}]"
        return f"rowz[{n}]" if self.rowz else n

    def to_dict(self):
        return dict(kind=self.kind, graph=self.graph.to_dict(), op=self.op, score=self.score, rowz=self.rowz)


def as_config(c):
    if not isinstance(c, dict):
        return c
    c = dict(c)
    if c.get("kind") != "graph":
        return layers.as_config(c)
    g = dict(c["graph"])
    gk = g.pop("kind")
    g["base"] = layers.as_config(g["base"])
    graph = KNNGraph(**g) if gk == "knn" else SoftGraph(**g)
    return GraphLayer(graph, c["op"], c["score"], c["rowz"])


def base_of(cfg) -> Spectral:
    return cfg.graph.base if isinstance(cfg, GraphLayer) else cfg.base


def series_coeffs(o: dict) -> np.ndarray:
    """Coefficients c_t (t = 0..T) of the diffusion series; T = first t with remaining mass < TRUNC."""
    if o["kind"] == "pow":
        c = np.zeros(o["t"] + 1)
        c[-1] = 1.0
        return c
    if o["kind"] == "ppr":
        a = o["alpha"]
        T = max(1, math.ceil(math.log(TRUNC) / math.log(1 - a)))
        return np.array([a * (1 - a) ** t for t in range(T + 1)])
    if o["kind"] == "heat":
        t = o["t"]
        c = [math.exp(-t)]
        while 1.0 - sum(c) >= TRUNC:
            c.append(c[-1] * t / len(c))
        return np.array(c)
    raise ValueError(o)


# ------------------------------------------------------------------------------------------ engine
class GraphEngine:
    """Wraps a LayerEngine; caches graphs, operators, embeddings and the last full score matrices."""

    def __init__(self, eng: LayerEngine, cache_size: int = 1, seed: int = SEEDS[0]):
        self.eng, self.cache_size, self.seed = eng, cache_size, seed
        self._adj, self._emb, self._M = {}, {}, OrderedDict()
        self.timing: dict = {}

    # ------------------------------------------------------------------ graphs
    def _knn_W(self, g: KNNGraph) -> sp.csr_matrix:
        e = self.eng
        K = e.n_neighbours(g.q)
        Kmax = max(K, e.n_neighbours(2.0))                  # same selection pass as src/layers SecondOrder
        idx, val = e.neighbours(g.base, True, Kmax)
        idx, val = idx[:, :K], val[:, :K]
        v = np.clip(val, 0.0, None)
        if g.weight == "bin":
            w = np.ones_like(v)
        elif g.weight == "lin":
            w = v
        else:
            w = v ** float(g.weight[1:])
        W = sp.csr_matrix((w.ravel().copy(), idx.ravel().copy(), np.arange(0, e.G * K + 1, K)), shape=(e.G, e.G))
        W.sort_indices()
        if g.sym == "dir":
            return W
        bgm = sp.diags(e.bg.astype(np.float64))
        WT = (W.T @ bgm).tocsr()                            # transpose, columns restricted to BG
        if g.sym == "union":
            S = W.maximum(WT)
        elif g.sym == "mutual":
            S = W.minimum(WT)
        else:
            S = (W + WT) * 0.5
        S = S.tocsr()
        nb = ~e.bg                                          # non-BG rows keep their directed edges
        if nb.any():
            S = (sp.diags(e.bg.astype(np.float64)) @ S + sp.diags(nb.astype(np.float64)) @ W).tocsr()
        S.eliminate_zeros()
        S.sort_indices()
        return S

    def adjacency(self, g: Graph):
        """KNNGraph -> csr (float64); SoftGraph -> dense float32 (G x G, BG columns, zero diagonal)."""
        if g in self._adj:
            return self._adj[g]
        e = self.eng
        if isinstance(g, KNNGraph):
            A = self._knn_W(g)
        else:
            na, nb = layers.null_pairs(e)
            ref = float(np.quantile(e.compute(g.base, na, nb), g.ref_q))
            _, Z32 = e.spectral(g.base)
            A = np.empty((e.G, e.G), np.float32)
            for b in e._all_blocks():
                rows, S = e._block_S(Z32, b)
                np.clip(S / np.float32(ref), 0.0, 1.0, out=S)
                S **= np.float32(g.beta)
                S[:, ~e.bg] = 0.0
                S[np.arange(len(rows)), rows] = 0.0
                A[rows] = S
            self.timing[f"soft_ref[{g.name}]"] = ref
        self._adj = {g: A}                                   # keep one graph at a time (memory)
        return A

    def operator(self, g: KNNGraph) -> sp.csr_matrix:
        W = self.adjacency(g)
        d = np.asarray(W.sum(1)).ravel()
        with np.errstate(divide="ignore"):
            r = np.where(d > 0, d ** -0.5, 0.0)
        N = (sp.diags(r) @ W @ sp.diags(r)).tocsr().astype(np.float32)
        N.sort_indices()
        return N

    # ------------------------------------------------------------------ profiles
    def diffusion_profiles(self, g: KNNGraph, op: str) -> np.ndarray:
        c = series_coeffs(parse_op(op))
        N = self.operator(g)
        X = N.toarray()                                     # N^1 (dense float32)
        if np.count_nonzero(c[1:]) == 1:                    # pure power N^t
            for _ in range(2, len(c)):
                X = N @ X
            return X
        P = X * np.float32(c[1])
        for t in range(2, len(c)):
            X = N @ X
            P += X * np.float32(c[t])
        return P

    def embedding(self, g: KNNGraph, op: str, d_max: int = 512) -> np.ndarray:
        """One eigsh of d_max + 1 eigenpairs per graph serves every d <= d_max (d_max capped at |BG| - 2)."""
        o = parse_op(op)
        d_max = min(d_max, self.eng.n_bg - 2)
        if o["d"] > d_max:
            raise ValueError("d > d_max")
        key = (g, d_max)
        if key not in self._emb:
            e = self.eng
            N = self.operator(g).astype(np.float64)
            Nbb = N[e.bg_idx][:, e.bg_idx]
            v0 = np.random.default_rng(self.seed).standard_normal(e.n_bg)
            lam, U = eigsh(Nbb, k=d_max + 1, which="LA", v0=v0, tol=1e-8)
            o_ = np.argsort(lam)[::-1]
            lam, U = lam[o_], U[:, o_]
            E = np.zeros((e.G, d_max + 1))
            E[e.bg_idx] = U
            nb = np.flatnonzero(~e.bg)
            if len(nb):
                with np.errstate(divide="ignore", invalid="ignore"):
                    E[nb] = (N[nb][:, e.bg_idx] @ U) / np.where(np.abs(lam) > 1e-12, lam, np.inf)
            self._emb = {key: (lam, E)}
        lam, E = self._emb[key]
        lam, E = lam[1:o["d"] + 1], E[:, 1:o["d"] + 1]      # drop the leading (trivial) eigenvector
        if o["kind"] == "dmap":
            E = E * np.sign(lam) * np.abs(lam) ** o["t"]
        return E

    def profiles(self, layer: GraphLayer):
        o = parse_op(layer.op)
        g = layer.graph
        if o["kind"] == "adj":
            A = self.adjacency(g)
            if isinstance(g, SoftGraph):
                return A
            if layer.score == "tom" and g.weight != "bin":
                ref = np.quantile(A.data, TOM_WREF_Q)
                A = A.copy()
                A.data = np.minimum(A.data / ref, 1.0)
            return A.astype(np.float32).toarray()
        if o["kind"] in ("lap", "dmap"):
            return self.embedding(g, layer.op, max(512, o["d"]))    # d <= 512 share one eigsh(513)
        return self.diffusion_profiles(g, layer.op)

    # ------------------------------------------------------------------ full score matrix
    def full_scores(self, layer: GraphLayer):
        """(M float32 G x G, row means, row SDs over BG \\ self) for the core (non-rowz) layer."""
        core = layer.core
        if core in self._M:
            self._M.move_to_end(core)
            return self._M[core]
        while len(self._M) >= self.cache_size:
            self._M.popitem(last=False)
        e = self.eng
        G = e.G
        P = self.profiles(core)
        M = np.empty((G, G), np.float32)
        m, s = np.empty(G), np.empty(G)
        if core.score == "emb":
            E = np.asarray(P, dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                E = E / np.linalg.norm(E, axis=1, keepdims=True)
            E = np.nan_to_num(E).astype(np.float32)
        elif core.score == "cos":
            sq = np.einsum("ij,ij->i", P, P, dtype=np.float64)
            dg = np.diagonal(P).astype(np.float64)
        elif core.score == "tom":
            k = P.sum(1, dtype=np.float64)
        for b in e._all_blocks():
            rows = np.arange(b * e.block, min((b + 1) * e.block, G))
            ar = np.arange(len(rows))
            if core.score == "emb":
                Mb = E[rows] @ E.T
            elif core.score == "kern":
                Mb = P[rows].astype(np.float64)
            else:
                D = (P[rows] @ P.T).astype(np.float64)
                Pr = P[rows].astype(np.float64)
                PTr = P[:, rows].T.astype(np.float64)
                if core.score == "cos":
                    dot = D - dg[rows, None] * PTr - Pr * dg[None, :]
                    na2 = sq[rows, None] - dg[rows, None] ** 2 - Pr ** 2
                    nb2 = sq[None, :] - dg[None, :] ** 2 - PTr ** 2
                    den = np.sqrt(np.clip(na2, 0, None) * np.clip(nb2, 0, None))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        Mb = np.where(den > 0, dot / den, 0.0)
                else:                                       # tom
                    a = np.maximum(Pr, PTr)
                    Mb = (D + a) / (np.minimum(k[rows, None], k[None, :]) + 1.0 - a)
                del D, Pr, PTr
            Mb = np.asarray(Mb, dtype=np.float64)
            Mb[ar, rows] = 0.0
            M[rows] = Mb
        if core.score == "kern":                            # non-BG columns (no in-edges): fill from rows
            nb = np.flatnonzero(~e.bg)
            M[:, nb] = M[nb, :].T                           # (non-BG x non-BG pairs score 0)
        del P
        for b in e._all_blocks():
            rows = np.arange(b * e.block, min((b + 1) * e.block, G))
            m[rows], s[rows] = layers.row_moments(M[rows][:, e.bg_idx], e.bg_pos[rows])
        self._M[core] = (M, m, s)
        return self._M[core]

    def compute(self, layer, ia, ib) -> np.ndarray:
        layer = as_config(layer)
        if not isinstance(layer, GraphLayer):
            from src import layers_ext
            return layers_ext.compute(layer, ia, ib, self.eng)
        ia, ib = np.asarray(ia, dtype=np.int64), np.asarray(ib, dtype=np.int64)
        if ia.shape != ib.shape:
            raise ValueError("ia/ib shapes differ")
        if (ia == ib).any():
            raise ValueError("self pairs are undefined")
        M, m, sd = self.full_scores(layer)
        s = 0.5 * (M[ia, ib].astype(np.float64) + M[ib, ia].astype(np.float64))
        if layer.rowz:
            with np.errstate(invalid="ignore", divide="ignore"):
                s = 0.5 * ((s - m[ia]) / sd[ia] + (s - m[ib]) / sd[ib])
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        return s

    def release(self):
        self._adj, self._emb = {}, {}
        self._M.clear()


_GE: dict = {}


def get_graph_engine(eng: LayerEngine | None = None) -> GraphEngine:
    eng = eng or layers.get_engine()
    if id(eng) not in _GE:
        _GE[id(eng)] = GraphEngine(eng)
    return _GE[id(eng)]


def compute(cfg, ia, ib, engine: LayerEngine | None = None) -> np.ndarray:
    """Scores for any layer config (graph layers here; others dispatched to src.layers_ext)."""
    return get_graph_engine(engine).compute(cfg, ia, ib)
