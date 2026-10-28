"""Standalone, label-free scorer for arbitrary UniProt protein-pair files (DESIGN §3.10, §5; RESEARCH_TREE D3).

Pipeline (``score_pairs`` / scripts/predict_pairs.py)
-----------------------------------------------------
1. Gene-effect matrix: raw ``datasets/DepMAP/CRISPRGeneEffect.csv`` parsed exactly as ``src.data.build_matrix_cache``
   (float32). Its ``np.save`` serialisation is hashed and asserted equal to the ``input.sha256`` of the layer
   manifests, i.e. the matrix is bit-identical to the one every cached layer was built from. ``cache=True`` reads
   ``data_cache/CRISPRGeneEffect.f32.npy`` instead (same check).
2. ID mapping: ``src.data.build_uniprot_map`` (HGNC 'Approved' rows, UniProt -> Entrez -> DepMap column, matched on
   Entrez; ambiguous paralog copies dropped). The status of an accession depends only on that accession. A pair is
   *mapped* iff both proteins map, to different DepMap genes, and every requested component is finite.
3. Components (label-free, each a deterministic function of X and the two gene indices; configs are frozen here and
   asserted equal to the manifest configs):
     B  rowz[socos_q0.25[spec_k20_t0.1_std]]                       results/layers/screen_v2.manifest.json col 17
     E  emb_lap512[knn_union_lin_q0.25[spec_k20_t0.1_std]]          results/phaseA_graph/graph_v1.manifest.json col 52
     H  rowz[cos_heat1[knn_union_lin_q0.25[spec_k20_t0.1_std]]]    results/phaseA_graph/graph_v1.manifest.json col 45
     F  spec_k20_t0.1_std                                           results/layers/manifest.json col 62
     G  GLS signed t (Wainberg 2021; src/baselines.py, best dev variant gls_signed)
     P  Pearson r, pairwise complete (pearson_signed)
     S  Spearman rho, pairwise complete (spearman_signed)
   Layers B/E/H/F use src.layers / layers_ext / layers_graph unchanged (genome-wide state computed over all genes in
   fixed blocks). G/P/S re-use src.baselines (pattern grouping, ``sample_covariance``, ``whitening``) but take the
   final per-pair products as row dot products and, for GLS, warp genes in fixed genome-wide blocks of 1024 genes,
   so that a pair's value never depends on which other pairs are requested (src.baselines uses one matrix product
   per request, equal to ~1e-12).
4. z-scores: fixed genome-wide null = the 200,000 random BG gene pairs of ``src.layers.null_pairs`` (seed SEEDS[0];
   SHA-256 asserted equal to the manifest's). Null moments are recomputed here and, for B/E/H/F, asserted equal to
   the manifest values (|d mean| <= ``NULL_TOL`` * null SD, |d SD| <= ``NULL_TOL`` * null SD). G/P/S have no manifest
   moments; theirs are recomputed the same way and recorded.
5. Methods: a single component (``single`` = B = D3's C_single) returns the raw component value (z is a monotone
   transform of it); ``zavg`` returns the equal-weight mean of component z-scores; a model file (``--model``)
   returns ``model.predict(z)`` (see ``load_model``).
6. Unmapped pairs get ``min(score over mapped pairs in the file) - 1`` (DESIGN §3.10) and mapped = 0. That floor is
   the only file-dependent quantity; every mapped pair's score is independent of the other pairs and of (a, b) order
   (pairs are canonicalised to (min gene index, max gene index) and de-duplicated before scoring).

Nothing here reads pair labels, the HuMAP pair table or which genes are labelled.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import pickle
import platform
import resource
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import baselines, data, layers, layers_ext, layers_graph
from src.config import DEPMAP, HGNC_FILE, RESULTS, ROOT
from src.layers import LayerEngine, SecondOrder, Spectral
from src.layers_ext import RowZSO
from src.layers_graph import GraphEngine, GraphLayer, KNNGraph

log = logging.getLogger(__name__)

MATRIX = "CRISPRGeneEffect"
RAW_CSV = DEPMAP / f"{MATRIX}.csv"
CACHED_NPY = data.CACHE / f"{MATRIX}.f32.npy"
NULL_TOL = 1e-4                  # null-moment agreement with the manifests, in units of the null SD
WHITE_CACHE = 16                 # GLS whitening operators kept in memory (recomputed deterministically if evicted)
GENE_BLOCK = 1024                # fixed genome-wide gene blocks for GLS warping (batch invariance)

# ------------------------------------------------------------------------------------------ components
F_CFG = Spectral(20, 0.1, standardize=True)
UNION = KNNGraph(F_CFG, 0.25, "union", "lin")
CONFIGS = {
    "B": RowZSO(SecondOrder(F_CFG, 0.25, "cos", True)),
    "E": GraphLayer(UNION, "lap512", "emb", False),
    "H": GraphLayer(UNION, "heat1", "cos", True),
    "F": F_CFG,
}
MANIFESTS = {
    "B": RESULTS / "layers" / "screen_v2.manifest.json",
    "E": RESULTS / "phaseA_graph" / "graph_v1.manifest.json",
    "H": RESULTS / "phaseA_graph" / "graph_v1.manifest.json",
    "F": RESULTS / "layers" / "manifest.json",
}
BASELINES = {"G": "gls", "P": "pearson", "S": "spearman"}
COMPONENT_NAMES = {**{k: c.name for k, c in CONFIGS.items()},
                   "G": "gls_signed", "P": "pearson_signed", "S": "spearman_signed"}
LAYER_ORDER = ("F", "B", "E", "H", "G", "P", "S")       # computation order (shared prep, memory)
_ALIAS = {"gls": "G", "pearson": "P", "spearman": "S", "single": "B"}


def _code(tok: str) -> str:
    t = tok.strip()
    c = _ALIAS.get(t.lower(), t.upper() if len(t) == 1 else t)
    if c not in COMPONENT_NAMES:
        raise ValueError(f"unknown component {tok!r}; known: {sorted(COMPONENT_NAMES)} + {sorted(_ALIAS)}")
    return c


# ------------------------------------------------------------------------------------------ methods / models
class ZAvg:
    """Equal-weight mean of component z-scores (frozen genome-wide null)."""

    def __init__(self, components):
        self.components = list(components)

    def predict(self, Z: dict) -> np.ndarray:
        out = np.zeros(len(Z[self.components[0]]))
        for c in self.components:
            out = out + Z[c]
        return out / len(self.components)

    def to_dict(self):
        return dict(kind="zavg", components=self.components)


class LinearZ:
    """intercept + sum_c w_c z_c (e.g. a logistic regression fitted on component z-scores)."""

    def __init__(self, components, weights, intercept=0.0):
        if len(components) != len(weights):
            raise ValueError("components / weights length mismatch")
        self.components, self.weights, self.intercept = list(components), [float(w) for w in weights], float(intercept)

    def predict(self, Z: dict) -> np.ndarray:
        out = np.full(len(Z[self.components[0]]), self.intercept)
        for c, w in zip(self.components, self.weights):
            out = out + w * Z[c]
        return out

    def to_dict(self):
        return dict(kind="linear", components=self.components, weights=self.weights, intercept=self.intercept)


def load_model(path) -> object:
    """Plug-in point for a fitted (e.g. supervised) combination.

    * ``.json``: {"kind": "zavg", "components": [...]} or {"kind": "linear", "components": [...], "weights": [...],
      "intercept": x}; components are codes (B, E, H, F, G, P, S) or names (gls, pearson, spearman).
    * ``.pkl`` / ``.pickle``: any object with ``components`` (list of codes) and ``predict(Z) -> scores`` where Z maps
      code -> z-score array of the mapped pairs. Only load model files you produced yourself (pickle executes code).
    """
    path = Path(path)
    if path.suffix == ".json":
        spec = json.loads(path.read_text())
        comps = [_code(c) for c in spec["components"]]
        if spec["kind"] == "zavg":
            return ZAvg(comps)
        if spec["kind"] == "linear":
            return LinearZ(comps, spec["weights"], spec.get("intercept", 0.0))
        raise ValueError(f"unknown model kind {spec['kind']!r}")
    with open(path, "rb") as fh:
        m = pickle.load(fh)
    m.components = [_code(c) for c in m.components]
    if not callable(getattr(m, "predict", None)):
        raise TypeError("model object needs predict(Z)")
    return m


@dataclass
class Method:
    name: str
    kind: str                     # 'raw' | 'zavg' | 'model'
    components: list
    model: object = None
    spec: dict = field(default_factory=dict)

    @property
    def needs_z(self) -> bool:
        return self.kind != "raw"


def parse_method(method: str | None, model_path=None) -> Method:
    """'single' | B/E/H/F/G/P/S | gls/pearson/spearman -> raw component; 'zavg:B,E,F' or 'comb_zavg_BEF' -> z-average;
    model_path -> fitted model (method ignored)."""
    if model_path is not None:
        m = load_model(model_path)
        return Method(f"model:{Path(model_path).name}", "model", list(m.components), m,
                      dict(path=str(model_path), sha256=sha256_file(model_path),
                           **(m.to_dict() if hasattr(m, "to_dict") else {})))
    if method is None:
        raise ValueError("--method or --model required")
    if method.startswith("zavg:"):
        comps = [_code(t) for t in method[5:].split(",") if t.strip()]
    elif method.startswith("comb_zavg_"):
        comps = [_code(t) for t in method[len("comb_zavg_"):]]
    else:
        return Method(method, "raw", [_code(method)])
    if len(comps) < 1 or len(set(comps)) != len(comps):
        raise ValueError(f"bad component list in {method!r}")
    return Method(method, "zavg", comps, ZAvg(comps), ZAvg(comps).to_dict())


# ------------------------------------------------------------------------------------------ hashing / provenance
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def npy_sha256(X: np.ndarray) -> str:
    """SHA-256 of ``np.save(X)`` (the manifests hash data_cache/CRISPRGeneEffect.f32.npy)."""
    buf = io.BytesIO()
    np.save(buf, X)
    return hashlib.sha256(buf.getbuffer()).hexdigest()


def git_state() -> dict:
    def g(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = g("status", "--porcelain", "--", "src", "scripts/predict_pairs.py")
    return dict(commit=g("rev-parse", "HEAD"), dirty_files=[l[3:] for l in dirty.splitlines()])


# ------------------------------------------------------------------------------------------ inputs
def load_matrix(cache: bool = False) -> tuple[np.ndarray, pd.Index, dict]:
    """(X float32 lines x genes, DepMap gene columns, provenance). Raw CSV by default."""
    t0 = time.perf_counter()
    if cache:
        X, _, genes = data.load_matrix(MATRIX)
        src = dict(source=str(CACHED_NPY.relative_to(ROOT)), file_sha256=sha256_file(CACHED_NPY))
    else:
        df = pd.read_csv(RAW_CSV, index_col=0)          # identical parse to data.build_matrix_cache
        genes = df.columns
        X = df.values.astype(np.float32)
        del df
        src = dict(source=str(RAW_CSV.relative_to(ROOT)), file_sha256=sha256_file(RAW_CSV))
    ref = json.loads(MANIFESTS["F"].read_text())["input"]
    h = npy_sha256(X)
    if h != ref["sha256"] or list(X.shape) != ref["shape"]:
        raise RuntimeError(f"gene-effect matrix differs from the one the layers were built from ({h} != {ref['sha256']})")
    src.update(shape=list(X.shape), npy_sha256=h, npy_sha256_matches_manifest=True,
               load_s=round(time.perf_counter() - t0, 2))
    return X, genes, src


def read_pairs(path) -> pd.DataFrame:
    """Two tab-separated UniProt columns, no header (extra columns ignored, blank lines skipped)."""
    a, b = [], []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            f = line.rstrip("\r\n").split("\t")
            if len(f) < 2 or not f[0].strip() or not f[1].strip():
                raise ValueError(f"{path}:{n}: expected two tab-separated UniProt IDs")
            a.append(f[0].strip())
            b.append(f[1].strip())
    return pd.DataFrame(dict(uniprot_a=a, uniprot_b=b), dtype=object)


# ------------------------------------------------------------------------------------------ baselines
def _rowdot(Z, la, lb, chunk: int = layers.PAIR_CHUNK):
    """Per-pair dot products of rows of Z (chunked for memory; each value depends only on its two rows)."""
    out = np.empty(len(la))
    for s in range(0, len(la), chunk):
        out[s:s + chunk] = np.einsum("ij,ij->i", Z[la[s:s + chunk]], Z[lb[s:s + chunk]])
    return out


class BaselineEngine:
    """Batch-invariant pairwise-complete Pearson / Spearman / GLS signed t (same estimators as src/baselines.py)."""

    def __init__(self, X: np.ndarray):
        self.X = X
        self.XT = np.ascontiguousarray(np.asarray(X, dtype=np.float64).T)       # genes x lines, NaN kept
        valid = ~np.isnan(X)
        keys = [valid[:, j].tobytes() for j in range(X.shape[1])]
        uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        self.pid = np.array([uniq[k] for k in keys])
        self.pmask = [np.frombuffer(k, dtype=bool) for k in uniq]
        self._sigma = None
        self._white: dict = {}
        self.info: dict = {}

    def groups(self, ia, ib):
        """Pairs grouped by the sample set observed in both genes (as baselines._pattern_blocks)."""
        n = len(self.pmask)
        p1, p2 = np.minimum(self.pid[ia], self.pid[ib]), np.maximum(self.pid[ia], self.pid[ib])
        code = p1 * n + p2
        for c in np.unique(code):
            idx = np.flatnonzero(code == c)
            P, Q = divmod(int(c), n)
            yield idx, self.pmask[P] & self.pmask[Q]

    def compute(self, method: str, ia, ib) -> np.ndarray:
        ia, ib = np.asarray(ia, np.int64), np.asarray(ib, np.int64)
        out = np.full(len(ia), np.nan)
        for idx, S in self.groups(ia, ib):
            genes = np.unique(np.r_[ia[idx], ib[idx]])
            la, lb = np.searchsorted(genes, ia[idx]), np.searchsorted(genes, ib[idx])
            if method == "gls":
                out[idx] = self._gls(S, genes, la, lb)
                continue
            Y = self.XT[genes][:, S]
            if method == "spearman":
                Y = stats.rankdata(Y, axis=1)
            Y = Y - Y.mean(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                Z = Y / np.sqrt((Y * Y).sum(1, keepdims=True))
            out[idx] = np.clip(_rowdot(Z, la, lb), -1.0, 1.0)
        return out

    def _whiten(self, S):
        key = S.tobytes()
        if key not in self._white:
            if self._sigma is None:
                self._sigma, ng = baselines.sample_covariance(self.X)
                self.info["gls_n_genes_for_covariance"] = ng
            A, keep = baselines.whitening(self._sigma[np.ix_(S, S)], "drop")
            w0 = A.sum(1)
            self._white[key] = (A, np.flatnonzero(S)[keep], w0 / np.linalg.norm(w0))
            while len(self._white) > WHITE_CACHE:                  # bound memory (each A ~ 15 MB)
                self._white.pop(next(iter(self._white)))
        return self._white[key]

    def _gls(self, S, genes, la, lb):
        A, Sk, w0 = self._whiten(S)
        df = A.shape[0] - 2
        Z = np.empty((len(genes), A.shape[0]))
        G = self.XT.shape[0]
        for b in np.unique(genes // GENE_BLOCK):            # warp whole fixed blocks: same matmul whatever requested
            rows = np.arange(b * GENE_BLOCK, min((b + 1) * GENE_BLOCK, G))
            Y = np.nan_to_num(self.XT[rows][:, Sk])         # NaN only in genes not observed on S (never used)
            W = Y @ A.T
            R = W - np.outer(W @ w0, w0)                    # project out the warped intercept
            with np.errstate(invalid="ignore", divide="ignore"):
                R /= np.linalg.norm(R, axis=1, keepdims=True)
            sel = (genes >= rows[0]) & (genes <= rows[-1])
            Z[sel] = R[genes[sel] - rows[0]]
        r = np.clip(_rowdot(Z, la, lb), -1.0, 1.0)
        if df < 1:
            return np.full(len(la), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            return r * np.sqrt(df / (1.0 - r * r))


# ------------------------------------------------------------------------------------------ scorer
class PairScorer:
    """Genome-wide state for all components; ``component(code, ia, ib)`` returns raw scores (float64)."""

    def __init__(self, X: np.ndarray, genes: pd.Index, graph_cache: int = 2, keep_graph: bool = False):
        t0 = time.perf_counter()
        self.keep_graph = keep_graph            # keep full graph score matrices after scoring (tests, repeated calls)
        self.genes = genes
        self.eng = LayerEngine(X)
        self.ge = GraphEngine(self.eng, cache_size=graph_cache)
        self.base = BaselineEngine(X)
        self.null = layers.null_pairs(self.eng)
        self.moments: dict = {}
        self.timing: dict = dict(engine_init_s=round(time.perf_counter() - t0, 2))
        self._check_null_pairs()

    def _check_null_pairs(self):
        ref = json.loads(MANIFESTS["F"].read_text())["null"]
        h = sha256_arrays(*self.null)
        if h != ref["sha256"] or len(self.null[0]) != ref["n"]:
            raise RuntimeError("genome-wide null pairs differ from the manifest")

    def component(self, code: str, ia, ib) -> np.ndarray:
        ia, ib = np.asarray(ia, np.int64), np.asarray(ib, np.int64)
        if (ia == ib).any():
            raise ValueError("self pairs are undefined")
        lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)       # canonical orientation
        if code in BASELINES:
            return self.base.compute(BASELINES[code], lo, hi)
        cfg = CONFIGS[code]
        if isinstance(cfg, GraphLayer):
            return self.ge.compute(cfg, lo, hi)
        if isinstance(cfg, RowZSO):
            return layers_ext.compute(cfg, lo, hi, self.eng)
        return self.eng.compute(cfg, lo, hi)

    def null_moments(self, code: str, check: bool = True) -> dict:
        if code not in self.moments:
            t0 = time.perf_counter()
            mo = layers.moments(self.component(code, *self.null))
            rec = dict(name=COMPONENT_NAMES[code], **mo, runtime_s=round(time.perf_counter() - t0, 2))
            if code in MANIFESTS:
                cols = {c["name"]: c for c in json.loads(MANIFESTS[code].read_text())["columns"]}
                ref = cols[COMPONENT_NAMES[code]]
                parse = layers_graph.as_config if isinstance(CONFIGS[code], GraphLayer) else layers_ext.as_config
                if parse(ref["config"]) != CONFIGS[code]:
                    raise RuntimeError(f"manifest config of {code} differs from the frozen config")
                d_mean = abs(mo["null_mean"] - ref["null_mean"]) / ref["null_sd"]
                d_sd = abs(mo["null_sd"] - ref["null_sd"]) / ref["null_sd"]
                rec.update(manifest=str(MANIFESTS[code].relative_to(ROOT)), manifest_col=ref["col"],
                           manifest_null_mean=ref["null_mean"], manifest_null_sd=ref["null_sd"],
                           rel_diff_mean=d_mean, rel_diff_sd=d_sd)
                if check and (d_mean > NULL_TOL or d_sd > NULL_TOL or mo["null_n"] != ref["null_n"]):
                    raise RuntimeError(f"null moments of {code} differ from the manifest: {rec}")
            else:
                rec["manifest"] = None
            self.moments[code] = rec
        return self.moments[code]

    def z(self, code, s):
        m = self.null_moments(code)
        return (np.asarray(s, np.float64) - m["null_mean"]) / m["null_sd"]

    def release_graph(self):
        if not self.keep_graph:
            self.ge.release()


def map_ids(pairs: pd.DataFrame, genes: pd.Index) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    ids = pd.unique(pairs[["uniprot_a", "uniprot_b"]].values.ravel())
    umap = data.build_uniprot_map(ids, genes)
    ia = umap.gene_idx.reindex(pairs.uniprot_a).values.astype(np.int64)
    ib = umap.gene_idx.reindex(pairs.uniprot_b).values.astype(np.int64)
    return ia, ib, umap


def score_genes(scorer: PairScorer, method: Method, ia, ib, components: bool = False,
                check_null: bool = True) -> tuple[np.ndarray, dict, dict]:
    """Scores for mapped gene-index pairs (ia != ib, all >= 0). Returns (score, {code: z}, timing). Unique canonical
    pairs are scored once; each pair's value depends only on its two genes."""
    lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
    key = lo * len(scorer.genes) + hi
    uk, inv = np.unique(key, return_inverse=True)
    ua, ub = uk // len(scorer.genes), uk % len(scorer.genes)
    raw, Z, timing = {}, {}, {}
    order = [c for c in LAYER_ORDER if c in method.components]
    graph_codes = [c for c in order if isinstance(CONFIGS.get(c), GraphLayer)]
    for c in order:
        t0 = time.perf_counter()
        raw[c] = scorer.component(c, ua, ub)
        if c in MANIFESTS:                       # reproducibility assertion for every frozen layer
            scorer.null_moments(c, check=check_null)
        if method.needs_z or components:
            Z[c] = scorer.z(c, raw[c])
        if graph_codes and c == graph_codes[-1]:
            scorer.release_graph()
        timing[c] = round(time.perf_counter() - t0, 2)
    if method.kind == "raw":
        s = raw[method.components[0]]
    else:
        s = np.asarray(method.model.predict(Z), dtype=np.float64)
    return s[inv], {c: z[inv] for c, z in Z.items()}, timing


def score_pairs(pairs: pd.DataFrame, method: Method, scorer: PairScorer, components: bool = False,
                check_null: bool = True) -> tuple[pd.DataFrame, dict]:
    """Output frame (input order and orientation): uniprot_a, uniprot_b, score, mapped [, z_<code>...]."""
    t0 = time.perf_counter()
    ia, ib, umap = map_ids(pairs, scorer.genes)
    t_map = time.perf_counter() - t0
    ok = (ia >= 0) & (ib >= 0) & (ia != ib)
    s = np.full(len(pairs), np.nan)
    Zc = {c: np.full(len(pairs), np.nan) for c in method.components} if components else {}
    timing = {}
    if ok.any():
        sm, Zm, timing = score_genes(scorer, method, ia[ok], ib[ok], components, check_null)
        s[ok] = sm
        for c in Zc:
            Zc[c][ok] = Zm[c]
    undefined = ok & ~np.isfinite(s)
    mapped = ok & np.isfinite(s)
    floor = (float(s[mapped].min()) if mapped.any() else 0.0) - 1.0
    s[~mapped] = floor
    out = pairs[["uniprot_a", "uniprot_b"]].copy()
    out["score"] = s
    out["mapped"] = mapped.astype(np.int8)
    for c in Zc:
        Zc[c][~mapped] = np.nan
        out[f"z_{c}"] = Zc[c]
    st_a = umap.status.reindex(pairs.uniprot_a).values
    st_b = umap.status.reindex(pairs.uniprot_b).values
    both = (ia >= 0) & (ib >= 0)
    counts = dict(n_pairs=len(pairs), n_mapped=int(mapped.sum()), n_unmapped=int((~mapped).sum()),
                  n_unique_canonical_mapped=int(len(np.unique(np.minimum(ia, ib)[ok] * 10**6 + np.maximum(ia, ib)[ok]))),
                  unmapped_reasons=dict(
                      protein_not_mapped=int((~both).sum()),
                      self_or_same_gene=int((both & (ia == ib)).sum()),
                      undefined_score=int(undefined.sum())),
                  protein_status_counts=umap.status.value_counts().to_dict(),
                  n_proteins=len(umap), unmapped_floor=floor,
                  pairs_with_protein_status={st: int(((st_a == st) | (st_b == st)).sum())
                                        for st in ("not_in_hgnc", "not_in_depmap", "ambiguous")})
    timing = dict(map_s=round(t_map, 2), components_s=timing, total_s=round(time.perf_counter() - t0, 2))
    return out, dict(counts=counts, timing=timing)


def run(pairs_path, method: Method, out_path, components: bool = False, cache: bool = False,
        record_path=None) -> dict:
    """End-to-end: load matrix (raw by default), score, write TSV + sidecar run record. Returns the record."""
    t_all = time.perf_counter()
    git = git_state()
    pairs = read_pairs(pairs_path)
    X, genes, msrc = load_matrix(cache)
    scorer = PairScorer(X, genes)
    del X
    out, rep = score_pairs(pairs, method, scorer, components)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False, na_rep="NA")
    rec = dict(
        tool="scripts/predict_pairs.py", created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        label_free="scores use only CRISPRGeneEffect + HGNC ID mapping; no pair label is read",
        method=dict(name=method.name, kind=method.kind, components=method.components,
                    component_names={c: COMPONENT_NAMES[c] for c in method.components},
                    score="raw component value" if method.kind == "raw" else "model.predict(component z)",
                    spec=method.spec),
        git=git,
        inputs=dict(pairs=dict(path=str(pairs_path), sha256=sha256_file(pairs_path)),
                    matrix=msrc, hgnc=dict(path=str(HGNC_FILE.relative_to(ROOT)), sha256=sha256_file(HGNC_FILE))),
        output=dict(path=str(out_path), sha256=sha256_file(out_path), columns=list(out.columns)),
        null=dict(n=len(scorer.null[0]), seed="SEEDS[0]", sha256=sha256_arrays(*scorer.null),
                  moments={c: scorer.moments[c] for c in scorer.moments}, tolerance_in_null_sd=NULL_TOL),
        unmapped_rule="score = min(mapped scores in this file) - 1, mapped = 0 (DESIGN §3.10)",
        **rep,
        runtime=dict(total_s=round(time.perf_counter() - t_all, 2), matrix_load_s=msrc["load_s"],
                     engine_init_s=scorer.timing["engine_init_s"], scoring=rep["timing"],
                     peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)),
        env=dict(python=platform.python_version(), numpy=np.__version__, machine=platform.machine()))
    rec.pop("timing")
    record_path = Path(record_path) if record_path else out_path.with_name(out_path.name + ".run.json")
    record_path.write_text(json.dumps(rec, indent=1, default=float))
    return rec
