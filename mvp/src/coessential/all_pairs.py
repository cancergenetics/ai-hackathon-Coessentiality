"""Score every unordered pair of protein-coding DepMap genes with zavg_BEF and its components (frozen method).

Definitions
-----------
* Gene universe: the DepMap ``CRISPRGeneEffect`` gene columns.
* Protein gene: a DepMap gene whose Entrez ID (from the column name ``SYMBOL (ENTREZ)``) is the ``entrez_id`` of an
  HGNC ``Approved`` record with at least one ``uniprot_ids`` accession. Other DepMap genes are excluded (counted).
* Low coverage: NaN fraction >= ``Config.max_nan`` (10%). These genes are not background genes of the method (they do
  not enter the covariance, the row moments, the neighbour sets or the null), but the method scores them exactly as
  ``predict_pairs`` does (mean imputation, top-K background neighbours, Nystrom embedding). They are scored and
  flagged (``low_coverage``).
* Pairs: all unordered pairs i < j of protein genes, i and j indexing the protein-gene list (DepMap column order).

Computation
-----------
The genome-wide state is the one of ``CoessentialityModel`` (unchanged). Scores come from ``CoessentialityModel.z_block``
over the model's fixed gene blocks (the blocks ``predict_genome_wide`` ranks on): row block x all genes, restricted to
protein rows / columns. The upper triangle (row index < column index) is the computed value; the lower triangle is its
exact mirror; the diagonal is NaN. z-scores use the fixed null moments (200,000 random background gene pairs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from coessential import data
from coessential.method import COMPONENTS, Config, CoessentialityModel

log = logging.getLogger("coessential")

MATRIX_DTYPES = {"zavg_BEF": np.float32, "z_B": np.float16, "z_E": np.float16, "z_F": np.float16}
THRESHOLDS = (2.0, 3.0, 4.0, 5.0)
QUANTILES = (0.0, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999,
             1.0)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ gene set
def protein_genes(genes: pd.Index, hgnc_path, nan_frac: np.ndarray, max_nan: float) -> tuple[pd.DataFrame, dict]:
    """Protein-gene table (index, symbol, entrez, uniprot_ids, hgnc_id, low_coverage, nan_fraction, depmap_column,
    depmap_index) and the counts behind the definition."""
    h = pd.read_csv(hgnc_path, sep="\t", dtype=str,
                    usecols=lambda c: c in ("hgnc_id", "symbol", "status", "entrez_id", "uniprot_ids"))
    h = h[h.status == "Approved"].dropna(subset=["entrez_id"])
    if h.entrez_id.duplicated().any():
        raise ValueError("Entrez ID on several Approved HGNC records")
    h = h.set_index("entrez_id")
    parsed = [data.parse_depmap_column(c) for c in genes]
    ent = pd.Index([p[1] for p in parsed])
    in_hgnc = ent.isin(h.index)
    uni = h.uniprot_ids.reindex(ent)
    has_uni = uni.notna().values & (uni.fillna("").str.len().values > 0)
    sel = np.flatnonzero(has_uni)
    low = nan_frac[sel] >= max_nan
    tab = pd.DataFrame(dict(index=np.arange(len(sel)),
                            symbol=h.symbol.reindex(ent[sel]).values,
                            entrez=ent[sel].astype(np.int64),
                            uniprot_ids=uni.values[sel],
                            hgnc_id=h.hgnc_id.reindex(ent[sel]).values,
                            low_coverage=low.astype(np.int8),
                            nan_fraction=nan_frac[sel],
                            depmap_column=np.asarray(genes)[sel],
                            depmap_index=sel))
    excl = np.flatnonzero(~has_uni)
    n_acc = tab.uniprot_ids.str.count(r"\|") + 1
    counts = dict(
        n_depmap_genes=len(genes),
        n_depmap_entrez_in_hgnc_approved=int(in_hgnc.sum()),
        n_depmap_not_in_hgnc_approved=int((~in_hgnc).sum()),
        n_excluded_no_uniprot=len(excl),
        excluded_genes=[str(genes[i]) for i in excl],
        n_protein_genes=len(tab),
        n_protein_genes_low_coverage=int(low.sum()),
        n_protein_genes_background=int((~low).sum()),
        protein_genes_by_n_uniprot_accessions={str(k): int(v) for k, v in n_acc.value_counts().sort_index().items()},
        n_pairs=len(tab) * (len(tab) - 1) // 2,
        n_pairs_background_only=int((~low).sum()) * (int((~low).sum()) - 1) // 2,
    )
    return tab, counts


# ------------------------------------------------------------------------------------------ scoring
def score_all_pairs(model: CoessentialityModel, gidx: np.ndarray, threshold: float = 3.0,
                    max_rows: int = 20_000_000, top_rows: int = 10_000_000):
    """Dense symmetric matrices over the genes ``gidx`` (model gene indices, ascending) and the pairs (i < j, indices
    into gidx) with zavg_BEF >= threshold. Returns (matrices, pair dict, stats)."""
    n = len(gidx)
    if not (np.diff(gidx) > 0).all():
        raise ValueError("gidx must be strictly increasing")
    pos = np.full(model.G, -1)
    pos[gidx] = np.arange(n)
    mats = {k: np.empty((n, n), dt) for k, dt in MATRIX_DTYPES.items()}
    cand = {k: [] for k in ("a", "b", "zavg_BEF", *[f"z_{c}" for c in COMPONENTS])}
    n_above = {t: 0 for t in THRESHOLDS}
    n_nonfinite = 0
    col = np.arange(n)
    t0 = time.perf_counter()
    for bi, rows in enumerate(model._blocks()):
        local = np.flatnonzero(pos[rows] >= 0)
        if not len(local):
            continue
        ip = pos[rows[local]]
        Zb = model.z_block(rows)                              # fixed genome-wide block (as predict_genome_wide)
        Z = {c: Zb[c][local][:, gidx] for c in COMPONENTS}
        del Zb
        za = (Z["B"] + Z["E"] + Z["F"]) / 3.0                 # identical arithmetic to zavg_block
        mats["zavg_BEF"][ip] = za
        for c in COMPONENTS:
            mats[f"z_{c}"][ip] = Z[c]
        upper = col[None, :] > ip[:, None]
        n_nonfinite += int((upper & ~np.isfinite(za)).sum())
        for t in THRESHOLDS:
            n_above[t] += int((upper & (za >= t)).sum())
        r, c = np.nonzero(upper & (za >= threshold))
        cand["a"].append(ip[r])
        cand["b"].append(c)
        cand["zavg_BEF"].append(za[r, c])
        for comp in COMPONENTS:
            cand[f"z_{comp}"].append(Z[comp][r, c].astype(np.float32))
        log.info("block %d: %d protein rows, %d pairs >= %g so far (%.0f s)", bi, len(ip),
                 sum(map(len, cand["a"])), threshold, time.perf_counter() - t0)
    for M in mats.values():                                   # mirror the upper triangle, NaN diagonal
        _mirror_upper(M)
    pairs = {k: np.concatenate(v) for k, v in cand.items()}
    n_cand = len(pairs["a"])
    rule = f"zavg_BEF >= {threshold:g}"
    if n_cand > max_rows:
        keep = np.argpartition(-pairs["zavg_BEF"], top_rows - 1)[:top_rows]
        pairs = {k: v[keep] for k, v in pairs.items()}
        rule = f"top {top_rows} by zavg_BEF ({n_cand} pairs have zavg_BEF >= {threshold:g})"
    o = np.lexsort((pairs["b"], pairs["a"], -pairs["zavg_BEF"]))
    pairs = {k: v[o] for k, v in pairs.items()}
    stats = dict(n_above={f">={t:g}": n_above[t] for t in THRESHOLDS}, n_nonfinite_upper=n_nonfinite,
                 pair_table_rule=rule, pair_table_rows=len(pairs["a"]), n_pairs_above_threshold=n_cand,
                 block_scoring_s=round(time.perf_counter() - t0, 1))
    return mats, pairs, stats


def _mirror_upper(M: np.ndarray, step: int = 2048):
    n = len(M)
    for i0 in range(0, n, step):
        i1 = min(i0 + step, n)
        M[i0:i1, :i0] = M[:i0, i0:i1].T
        D = M[i0:i1, i0:i1]
        lo = np.tril_indices(i1 - i0, -1)
        D[lo] = D.T[lo]
    np.fill_diagonal(M, np.nan)


def upper_values(M: np.ndarray) -> np.ndarray:
    """Upper-triangle values (i < j), row-major."""
    return np.concatenate([M[i, i + 1:] for i in range(len(M) - 1)])


def distribution(M: np.ndarray) -> dict:
    v = upper_values(M).astype(np.float64)
    q = np.quantile(v, QUANTILES)
    return dict(n=len(v), mean=float(v.mean()), sd=float(v.std()),
                quantiles={f"{p:.6g}": float(x) for p, x in zip(QUANTILES, q)},
                count_ge={f"{t:g}": int((v >= t).sum()) for t in THRESHOLDS})


# ------------------------------------------------------------------------------------------ writing
def pair_table(pairs: dict, tab: pd.DataFrame) -> pd.DataFrame:
    a, b = pairs["a"], pairs["b"]
    low = tab.low_coverage.values.astype(bool)
    return pd.DataFrame(dict(
        symbol_a=tab.symbol.values[a], symbol_b=tab.symbol.values[b],
        entrez_a=tab.entrez.values[a], entrez_b=tab.entrez.values[b],
        uniprot_a=tab.uniprot_ids.values[a], uniprot_b=tab.uniprot_ids.values[b],
        zavg_BEF=pairs["zavg_BEF"].astype(np.float32),
        **{f"z_{c}": pairs[f"z_{c}"].astype(np.float32) for c in COMPONENTS},
        rank=np.arange(1, len(a) + 1, dtype=np.int64),
        low_coverage_any=low[a] | low[b]))


def git_state(root: Path) -> dict:
    def g(*a):
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()
    return dict(commit=g("rev-parse", "HEAD"), dirty_files=[l[3:] for l in g("status", "--porcelain", "--", "mvp").splitlines()])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="predict_all_pairs", description=(
        "Dense zavg_BEF / z_B / z_E / z_F matrices over all protein-coding DepMap genes, plus a Parquet table of the "
        "strongest pairs and a manifest."))
    ap.add_argument("--depmap", default="datasets/DepMAP/CRISPRGeneEffect.csv", help="CRISPRGeneEffect.csv")
    ap.add_argument("--hgnc", default="datasets/HGNC/hgnc_complete_set.txt", help="hgnc_complete_set.txt")
    ap.add_argument("--out-dir", default="outputs/all_pairs", help="output directory (default %(default)s)")
    ap.add_argument("--threshold", type=float, default=3.0, help="pair table: zavg_BEF >= this (default %(default)s)")
    ap.add_argument("--max-rows", type=int, default=20_000_000,
                    help="if more pairs pass the threshold, keep the top --top-rows instead (default %(default)s)")
    ap.add_argument("--top-rows", type=int, default=10_000_000, help="default %(default)s")
    ap.add_argument("--manifest-copy", help="also write the manifest here (small, committable)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t_all = time.perf_counter()
    root = Path(__file__).resolve().parents[3]
    git = git_state(root)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dep = data.ensure_exists(a.depmap, "DepMap CRISPRGeneEffect.csv")
    hg = data.ensure_exists(a.hgnc, "HGNC hgnc_complete_set.txt")
    inputs = {"depmap": dict(path=str(a.depmap), sha256=sha256_file(dep)),
              "hgnc": dict(path=str(a.hgnc), sha256=sha256_file(hg))}
    t = time.perf_counter()
    X, genes = data.load_depmap(dep)
    cfg = Config()
    model = CoessentialityModel(X, cfg)
    del X
    tab, counts = protein_genes(genes, hg, model.nan_frac, cfg.max_nan)
    gidx = tab.depmap_index.values.astype(np.int64)
    log.info("%d protein genes (%d low coverage), %d excluded", counts["n_protein_genes"],
             counts["n_protein_genes_low_coverage"], counts["n_excluded_no_uniprot"])
    t_load = time.perf_counter() - t
    t = time.perf_counter()
    moments = {c: model.null_moments(c) for c in COMPONENTS}
    t_state = time.perf_counter() - t
    mats, pairs, st = score_all_pairs(model, gidx, a.threshold, a.max_rows, a.top_rows)
    del model
    t = time.perf_counter()
    dist = distribution(mats["zavg_BEF"])
    comp_dist = {k: dict(zip(("mean", "sd"), (float(v.mean()), float(v.std())))) for k, v in
                 ((k, upper_values(mats[k]).astype(np.float64)) for k in ("z_B", "z_E", "z_F"))}
    t_dist = time.perf_counter() - t
    t = time.perf_counter()
    files = {}
    for k, M in mats.items():
        p = out / f"{k}.npy"
        np.save(p, M)
        files[p.name] = dict(dtype=str(M.dtype), shape=list(M.shape))
    tab.to_csv(out / "genes.tsv", sep="\t", index=False, float_format="%.6g")
    files["genes.tsv"] = dict(rows=len(tab), columns=list(tab.columns))
    pt = pair_table(pairs, tab)
    pt.to_parquet(out / "top_pairs.parquet", index=False, compression="zstd", row_group_size=1_000_000)
    files["top_pairs.parquet"] = dict(rows=len(pt), columns={c: str(d) for c, d in pt.dtypes.items()},
                                      rule=st["pair_table_rule"])
    for name, rec in files.items():
        rec.update(bytes=(out / name).stat().st_size, sha256=sha256_file(out / name))
    t_write = time.perf_counter() - t
    man = dict(
        tool="mvp/scripts/predict_all_pairs.py",
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        method=dict(name="zavg_BEF", definition="(z_B + z_E + z_F) / 3, fixed-null z-scores (mvp/src/coessential/method.py)",
                    config={k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                    null_moments={c: dict(mean=m, sd=s) for c, (m, s) in moments.items()},
                    label_free="no pair labels (hu.MAP or other) are read"),
        definitions=dict(
            gene_universe="DepMap CRISPRGeneEffect gene columns",
            protein_gene="Entrez ID of the DepMap column 'SYMBOL (ENTREZ)' is the entrez_id of an HGNC Approved record "
                         "with >= 1 uniprot_ids accession",
            low_coverage=f"NaN fraction >= {cfg.max_nan} (non-background genes: scored as predict_pairs does, flagged)",
            pairs="all unordered pairs i < j of protein genes (genes.tsv index = DepMap column order)",
            matrices="dense n x n; upper triangle (row < col) = computed value, lower triangle = its exact mirror, "
                     "diagonal NaN",
            pair_table=st["pair_table_rule"] + "; ranked by zavg_BEF (float64 before storage) descending, ties by "
                       "index_a, index_b; scores stored as float32",
            low_coverage_any="either gene has low_coverage = 1"),
        inputs=inputs, code=git, counts=counts,
        files=files,
        scores=dict(zavg_BEF_over_all_pairs=dist, components_over_all_pairs=comp_dist,
                    n_ge_from_float64={k: v for k, v in st["n_above"].items()},
                    n_nonfinite_upper=st["n_nonfinite_upper"],
                    note="quantiles / count_ge from the stored float32 matrix; n_ge_from_float64 from the unrounded "
                         "block values (the pair-table threshold is applied to these)"),
        pair_table=dict(rule=st["pair_table_rule"], rows=st["pair_table_rows"],
                        n_pairs_above_threshold=st["n_pairs_above_threshold"], threshold=a.threshold),
        runtime=dict(total_s=round(time.perf_counter() - t_all, 1), load_and_model_init_s=round(t_load, 1),
                     genome_wide_state_and_null_s=round(t_state, 1), block_scoring_s=st["block_scoring_s"],
                     distribution_s=round(t_dist, 1), write_and_hash_s=round(t_write, 1),
                     peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                       / (1e9 if platform.system() == "Darwin" else 1e6), 2)),
        env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                 machine=platform.machine(), platform=platform.platform()))
    txt = json.dumps(man, indent=1)
    (out / "manifest.json").write_text(txt + "\n")
    if a.manifest_copy:
        Path(a.manifest_copy).write_text(txt + "\n")
    print(json.dumps(dict(out_dir=str(out), n_protein_genes=counts["n_protein_genes"], n_pairs=counts["n_pairs"],
                          pair_table_rows=len(pt), runtime_s=man["runtime"]["total_s"],
                          peak_rss_gb=man["runtime"]["peak_rss_gb"])))
    return man
