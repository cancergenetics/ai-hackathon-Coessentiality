"""DESIGN v2 §3.1-3.2: approximate complexes (clusters) and the protein-level IFT / dev assignment.

Clusters (§3.1)
---------------
Universe = scorable proteins (UniProt IDs that map to a DepMap column and occur in >=1 scorable pair).
Graph = positive scorable pairs. Connected components; every component with > MAX_CLUSTER_SIZE (40) proteins is
split by ``networkx.community.louvain_communities(seed=SEEDS[0])`` (resolution 1, applied once). Proteins with no
positive pair are singleton clusters. Positive edges whose endpoints fall into different clusters
("cross-community edges") are counted and reported; they are never evaluated across a split boundary.
Cluster ids are deterministic: clusters sorted by (component, sorted member list).

IFT assignment (§3.2, stratified per R2 review + PI addenda)
-------------------------------------------------------------
Clusters are put in strata before randomisation so that a single random draw cannot swallow the giant component
or several of the largest complexes:
  * ``giant``    : the Louvain communities of components that were split (here: the 144-protein component);
  * ``top10``    : the 10 non-giant clusters with most within-cluster positives;
  * ``pos_q1..3``: remaining clusters with >=1 within-cluster positive, tertiles of within-cluster positives;
  * ``no_pos``   : clusters without within-cluster positives (singletons).
Within each positive stratum, clusters are visited in a seeded random order (``SEEDS[0]``) and moved to IFT until
the stratum's IFT within-cluster positives reach IFT_FRACTION (25%) of the stratum's within-cluster positives
(a cluster is taken only if it brings the stratum closer to the target, the first one always).
``no_pos`` clusters go to IFT with probability equal to the protein fraction of IFT among positive strata.
A draw is rejected (and redrawn from the same generator) if the Kish effective number of clusters of the IFT
(weights = within-cluster positives) is < 0.2 x that of all clusters; rejections are recorded.
IFT pairs = both proteins IFT; dev pairs = both dev; cross pairs (one each) are excluded from everything.
Because every protein has exactly one side, a protein can never be on both sides; the positive edges between an
IFT and a dev cluster are dropped (counted).

Unmapped labelled proteins (no DepMap column) get a side too, only so that §3.10 (IFT reported with unmapped pairs
at the minimum score) is defined: majority side of their positive partners, else a seeded draw with the IFT
protein fraction. They never affect clusters or dev metrics.

The assignment is written to data_cache/splits/protein_assignment.csv (gitignored) and its SHA-256 committed in
results/splits/protein_assignment.sha256; ``load_assignment`` verifies the hash.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.config import CACHE, RESULTS, SEEDS

MAX_CLUSTER_SIZE = 40
IFT_FRACTION = 0.25
N_TOP = 10                 # top non-giant clusters by within-cluster positives form their own stratum
MIN_NEFF_RATIO = 0.2       # reject IFT draws with Kish n_eff(IFT) < 0.2 * n_eff(all)
ASSIGN_FILE = CACHE / "splits" / "protein_assignment.csv"
SHA_FILE = RESULTS / "splits" / "protein_assignment.sha256"
SUMMARY_FILE = RESULTS / "splits" / "clusters_summary.json"


# ------------------------------------------------------------------------------------------- clusters
def build_clusters(pairs: pd.DataFrame, max_size: int = MAX_CLUSTER_SIZE, seed: int = SEEDS[0]) -> pd.DataFrame:
    """pairs: scorable pair table (a, b, label). Returns one row per protein: protein, component, component_size,
    cluster, cluster_size, louvain_split (bool: its component was split)."""
    prots = sorted(set(pairs.a) | set(pairs.b))
    G = nx.Graph()
    G.add_nodes_from(prots)
    pos = pairs[pairs.label == 1]
    G.add_edges_from(zip(pos.a, pos.b))
    comps = sorted((sorted(c) for c in nx.connected_components(G)), key=lambda c: c[0])
    rows = []
    groups = []
    for ci, members in enumerate(comps):
        if len(members) > max_size:
            # integer relabelling: louvain aggregates nodes in sets, so string node ids made the result depend on
            # PYTHONHASHSEED (found 2026-10-27); ints hash deterministically
            sub = nx.convert_node_labels_to_integers(G.subgraph(members), ordering="sorted", label_attribute="pid")
            com = nx.community.louvain_communities(sub, seed=seed)
            com = [[sub.nodes[i]["pid"] for i in c] for c in com]
            parts = sorted((sorted(c) for c in com), key=lambda c: c[0])
            split = True
        else:
            parts, split = [members], False
        for part in parts:
            groups.append((ci, len(members), split, part))
    for cl, (ci, csize, split, part) in enumerate(groups):
        for p in part:
            rows.append(dict(protein=p, component=ci, component_size=csize, cluster=cl, cluster_size=len(part),
                             louvain_split=split))
    return pd.DataFrame(rows).sort_values("protein").reset_index(drop=True)


def cluster_edge_stats(pairs: pd.DataFrame, cl: pd.DataFrame) -> dict:
    c = dict(zip(cl.protein, cl.cluster))
    pos = pairs[pairs.label == 1]
    ca, cb = pos.a.map(c).values, pos.b.map(c).values
    cross = ca != cb
    sizes = cl.groupby("cluster").size()
    return dict(n_proteins=len(cl), n_components=int(cl.component.nunique()), n_clusters=int(cl.cluster.nunique()),
                n_singleton_clusters=int((sizes == 1).sum()), max_cluster_size=int(sizes.max()),
                components_split=int(cl.loc[cl.louvain_split, "component"].nunique()),
                split_component_sizes=sorted(cl.loc[cl.louvain_split].groupby("component").size().tolist()),
                n_positive_edges=int(len(pos)), cross_community_positive_edges=int(cross.sum()),
                frac_positive_edges_cross_community=float(cross.mean()))


# ----------------------------------------------------------------------------------------- IFT split
def _within_pos(pairs, cl):
    c = dict(zip(cl.protein, cl.cluster))
    pos = pairs[pairs.label == 1]
    ca, cb = pos.a.map(c).values, pos.b.map(c).values
    return pd.Series(ca[ca == cb]).value_counts().reindex(sorted(cl.cluster.unique())).fillna(0).astype(int)


def strata(pairs: pd.DataFrame, cl: pd.DataFrame, n_top: int = N_TOP) -> pd.Series:
    """Stratum label per cluster (index = cluster id): giant / top10 / pos_q1..3 / no_pos (see module doc)."""
    wp = _within_pos(pairs, cl)
    split = cl.groupby("cluster").louvain_split.first().reindex(wp.index).values
    st = pd.Series("no_pos", index=wp.index, dtype=object)
    st[split] = "giant"
    cand = wp[(~split) & (wp.values > 0)]
    top = cand.sort_values(ascending=False, kind="mergesort").index[:n_top]
    st[top] = "top10"
    rest = (st.values == "no_pos") & (wp.values > 0)
    edges = np.quantile(wp.values[rest], [1 / 3, 2 / 3])
    q = 1 + np.searchsorted(edges, wp.values, side="left")          # 1..3; equal values share a bin
    st[rest] = [f"pos_q{int(k)}" for k in q[rest]]
    return st


def kish_neff(w) -> float:
    w = np.asarray(w, dtype=np.float64)
    return float(w.sum() ** 2 / (w ** 2).sum()) if (w ** 2).sum() > 0 else 0.0


def _draw(st, wp, csize, rng, frac):
    side = pd.Series("dev", index=wp.index, dtype=object)
    for s in sorted(set(st) - {"no_pos"}):
        ids = np.array(sorted(st.index[st == s]))
        target = frac * wp[ids].sum()
        acc = 0
        for c in rng.permutation(ids):
            if acc >= target:
                break
            if acc == 0 or abs(acc + wp[c] - target) <= abs(acc - target):
                side[c] = "ift"
                acc += wp[c]
    posc = st.index[st != "no_pos"]
    p_ift = csize[posc][side[posc] == "ift"].sum() / csize[posc].sum()
    nop = np.array(sorted(st.index[st == "no_pos"]))
    draws = rng.random(len(nop))
    side[nop[draws < p_ift]] = "ift"
    return side


def assign_sides(pairs: pd.DataFrame, cl: pd.DataFrame, seed: int = SEEDS[0], frac: float = IFT_FRACTION,
                 min_neff_ratio: float = MIN_NEFF_RATIO, max_draws: int = 1000) -> tuple[pd.DataFrame, dict]:
    """Adds columns stratum, cluster_within_pos, side ('ift'/'dev') to the cluster table. Draws are repeated from
    the same seeded generator until Kish n_eff(IFT) >= min_neff_ratio * Kish n_eff(all), with n_eff computed over
    clusters weighted by within-cluster positives. Returns (table, draw_info)."""
    wp = _within_pos(pairs, cl)
    st = strata(pairs, cl)
    csize = cl.groupby("cluster").size().reindex(wp.index)
    rng = np.random.default_rng(seed)
    neff_all = kish_neff(wp.values)
    for k in range(max_draws):
        side = _draw(st, wp, csize, rng, frac)
        neff_ift = kish_neff(wp[side == "ift"].values)
        if neff_ift >= min_neff_ratio * neff_all:
            break
    else:
        raise RuntimeError("no acceptable IFT draw")
    out = cl.copy()
    out["stratum"] = out.cluster.map(st)
    out["cluster_within_pos"] = out.cluster.map(wp)
    out["side"] = out.cluster.map(side)
    out["mapped"] = True
    return out, dict(n_rejected_draws=k, kish_neff_all=neff_all, kish_neff_ift=neff_ift,
                     kish_neff_dev=kish_neff(wp[side == "dev"].values), min_neff_ratio=min_neff_ratio)


def assign_unmapped(all_pairs: pd.DataFrame, assign: pd.DataFrame, seed: int = SEEDS[0]) -> pd.DataFrame:
    """Rows for labelled proteins outside the scorable universe (side by majority of positive partners, else a
    seeded draw with the IFT protein fraction). component/cluster = -1."""
    side = dict(zip(assign.protein, assign.side))
    allp = sorted(set(all_pairs.a) | set(all_pairs.b))
    extra = [p for p in allp if p not in side]
    pos = all_pairs[all_pairs.label == 1]
    p_ift = (assign.side == "ift").mean()
    rng = np.random.default_rng([seed, 1])                         # stream derived from SEEDS[0]
    rows = []
    for p in extra:
        partners = list(pos.b[pos.a == p]) + list(pos.a[pos.b == p])
        votes = [side[q] for q in partners if q in side]
        u = rng.random()
        n_ift, n_dev = votes.count("ift"), votes.count("dev")
        s = "ift" if n_ift > n_dev else "dev" if n_dev > n_ift else ("ift" if u < p_ift else "dev")
        rows.append(dict(protein=p, component=-1, component_size=0, cluster=-1, cluster_size=0, louvain_split=False,
                         stratum="unmapped", cluster_within_pos=0, side=s, mapped=False))
    return pd.concat([assign, pd.DataFrame(rows)], ignore_index=True).sort_values("protein").reset_index(drop=True)


def pair_sides(pairs: pd.DataFrame, assign: pd.DataFrame) -> np.ndarray:
    """'dev' / 'ift' / 'cross' per pair."""
    s = dict(zip(assign.protein, assign.side))
    sa, sb = pairs.a.map(s).values, pairs.b.map(s).values
    return np.where(sa == sb, sa, "cross")


# --------------------------------------------------------------------------------------- persistence
def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_assignment(assign: pd.DataFrame, path: Path = ASSIGN_FILE, sha_file: Path = SHA_FILE) -> str:
    b = _csv_bytes(assign)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b)
    h = sha256_bytes(b)
    sha_file.parent.mkdir(parents=True, exist_ok=True)
    sha_file.write_text(f"{h}  data_cache/splits/{path.name}\n")
    return h


def load_assignment(path: Path = ASSIGN_FILE, sha_file: Path = SHA_FILE) -> pd.DataFrame:
    """Loads the protein assignment and verifies its SHA-256 against the committed value."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/03_make_splits.py (regenerated from SEEDS[0])")
    b = path.read_bytes()
    want = sha_file.read_text().split()[0]
    got = sha256_bytes(b)
    if got != want:
        raise RuntimeError(f"protein assignment hash mismatch: {got} != committed {want}")
    return pd.read_csv(io.BytesIO(b))


def summary(pairs: pd.DataFrame, all_pairs: pd.DataFrame, assign: pd.DataFrame) -> dict:
    """Counts only (no protein or pair lists)."""
    sc = assign[assign.mapped]
    sides = pair_sides(pairs, sc)
    y = pairs.label.values == 1
    out = dict(cluster_stats=cluster_edge_stats(pairs, sc))
    out["pairs"] = {s: dict(pos=int((y & (sides == s)).sum()), neg=int((~y & (sides == s)).sum()))
                    for s in ("dev", "ift", "cross")}
    tot = int(y.sum())
    out["ift_frac_of_positives"] = out["pairs"]["ift"]["pos"] / tot
    out["proteins"] = sc.side.value_counts().to_dict()
    out["clusters"] = sc.groupby("side").cluster.nunique().to_dict()
    giant = sc.component_size.max()
    g = sc[sc.component_size == giant]
    c = dict(zip(sc.protein, sc.cluster))
    gpos = pairs[y & pairs.a.isin(g.protein) & pairs.b.isin(g.protein)]
    gs = pair_sides(gpos, sc)
    out["giant_component"] = dict(n_proteins=int(len(g)), n_positive_edges=int(len(gpos)),
                                  n_communities=int(g.cluster.nunique()),
                                  frac_proteins_ift=float((g.side == "ift").mean()),
                                  frac_positive_edges_ift=float((gs == "ift").mean()),
                                  frac_positive_edges_dev=float((gs == "dev").mean()),
                                  frac_positive_edges_cross=float((gs == "cross").mean()))
    ift_cl = sc[sc.side == "ift"]
    ift_pos = pairs[y & (sides == "ift")]
    per_cl = pd.Series(ift_pos.a.map(c).values).value_counts()
    out["ift_max_cluster_share_of_positives"] = float(per_cl.max() / max(len(ift_pos), 1))
    out["strata"] = {s: dict(n_clusters=int(d.cluster.nunique()),
                             within_pos=int(d.drop_duplicates("cluster").cluster_within_pos.sum()),
                             within_pos_ift=int(d[d.side == "ift"].drop_duplicates("cluster").cluster_within_pos.sum()))
                     for s, d in sc.groupby("stratum")}
    um = assign[~assign.mapped]
    out["unmapped_proteins"] = um.side.value_counts().to_dict()
    asides = pair_sides(all_pairs, assign)
    ns = ~all_pairs.scorable.values
    ya = all_pairs.label.values == 1
    out["unscorable_pairs"] = {s: dict(pos=int((ns & ya & (asides == s)).sum()), neg=int((ns & ~ya & (asides == s)).sum()))
                               for s in ("dev", "ift", "cross")}
    out["ift_n_clusters"] = int(ift_cl.cluster.nunique())
    top = sc[sc.stratum == "top10"].drop_duplicates("cluster")
    out["top10_clusters_in_ift"] = int((top.side == "ift").sum())
    out["giant_communities_in_ift"] = int(g[g.side == "ift"].cluster.nunique())
    out["protein_fraction_ift"] = float((sc.side == "ift").mean())
    return out


def build_all(all_pairs: pd.DataFrame, seed: int = SEEDS[0]) -> tuple[pd.DataFrame, dict]:
    """all_pairs: canonical pair table from data.load_pairs (incl. unscorable rows).
    networkx Louvain tie-breaking still depends on PYTHONHASHSEED (2 giant-component proteins swap communities), so the
    canonical assignment is defined as the output under PYTHONHASHSEED=0; a copy is committed in results/splits/."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("build_all must run with PYTHONHASHSEED=0 (Louvain is hash-seed dependent)")
    pairs = all_pairs[all_pairs.scorable].reset_index(drop=True)
    cl = build_clusters(pairs, seed=seed)
    assign, draw = assign_sides(pairs, cl, seed=seed)
    assign = assign_unmapped(all_pairs, assign, seed=seed)
    summ = summary(pairs, all_pairs, assign)
    summ["ift_draw"] = draw
    return assign, summ


if __name__ == "__main__":
    print(json.dumps(dict(MAX_CLUSTER_SIZE=MAX_CLUSTER_SIZE, IFT_FRACTION=IFT_FRACTION)))
