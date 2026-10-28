"""Command-line entry points: score a pair file, rank gene pairs genome-wide, evaluate scores against labels."""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from coessential import data
from coessential.baselines import METHODS as BASELINE_METHODS
from coessential.baselines import Baselines
from coessential.method import COMPONENTS, Config, CoessentialityModel

METHODS = ("zavg_BEF", *COMPONENTS, *BASELINE_METHODS)
log = logging.getLogger("coessential")


def _config_args(ap: argparse.ArgumentParser):
    g = ap.add_argument_group("advanced (changing these changes the published method)")
    d = Config()
    g.add_argument("--k", type=int, default=d.k, help="top eigencomponents removed (default %(default)s)")
    g.add_argument("--tau", type=float, default=d.tau, help="ridge, fraction of mean eigenvalue (default %(default)s)")
    g.add_argument("--q", type=float, default=d.q, help="neighbours kept, percent of genes (default %(default)s)")
    g.add_argument("--dim", type=int, default=d.d, help="embedding dimension (default %(default)s)")
    g.add_argument("--null-pairs", type=int, default=d.n_null, help="null gene pairs (default %(default)s)")


def _config(a) -> Config:
    return Config(k=a.k, tau=a.tau, q=a.q, d=a.dim, n_null=a.null_pairs)


# ------------------------------------------------------------------------------------------ pair scoring
def score_pairs(pairs: pd.DataFrame, genes: pd.Index, hgnc: pd.DataFrame, X: np.ndarray, method: str,
                config: Config = Config(), components: bool = False, add_gls: bool = False,
                model: CoessentialityModel | None = None, baselines: Baselines | None = None) -> pd.DataFrame:
    """Score UniProt pairs (input order and orientation kept).

    Columns: uniprot_a, uniprot_b, score, mapped [, z_B, z_E, z_F] [, gls_t]. ``score`` is the zavg_BEF value, the
    raw component value (B, E, F) or the baseline value (gls: t, pearson: r, spearman: rho). Pairs with an unmapped
    protein, two proteins on the same gene, or an undefined score have mapped = 0 and score NaN (callers choose how
    to fill it)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    ia, ib, _ = data.map_pairs(pairs, genes, hgnc)
    ok = (ia >= 0) & (ib >= 0) & (ia != ib)
    lo, hi = np.minimum(ia[ok], ib[ok]), np.maximum(ia[ok], ib[ok])
    key = lo * len(genes) + hi
    uk, inv = np.unique(key, return_inverse=True)            # score each unique pair once
    ua, ub = uk // len(genes), uk % len(genes)
    cols: dict[str, np.ndarray] = {}
    needs_model = method in ("zavg_BEF", *COMPONENTS) or components
    if needs_model and model is None:
        model = CoessentialityModel(X, config)
    if (method in BASELINE_METHODS or add_gls) and baselines is None:
        baselines = Baselines(X)
    t0 = time.perf_counter()
    if method == "zavg_BEF" or components:
        s, Z = model.zavg(ua, ub)
        if method == "zavg_BEF":
            cols["score"] = s
        if components:
            cols.update({f"z_{c}": Z[c] for c in COMPONENTS})
    if method in COMPONENTS:
        cols["score"] = model.component(method, ua, ub)
    if method in BASELINE_METHODS:
        cols["score"] = baselines.score(method, ua, ub)
    if add_gls:
        cols["gls_t"] = cols["score"] if method == "gls" else baselines.score("gls", ua, ub)
    log.info("scored %d unique pairs in %.1f s", len(uk), time.perf_counter() - t0)
    out = pairs[["uniprot_a", "uniprot_b"]].copy().reset_index(drop=True)
    full = {}
    for c, v in cols.items():
        f = np.full(len(pairs), np.nan)
        f[ok] = v[inv]
        full[c] = f
    mapped = ok & np.isfinite(full["score"])
    out["score"] = np.where(mapped, full.pop("score"), np.nan)
    out["mapped"] = mapped.astype(np.int8)
    for c in ([f"z_{c}" for c in COMPONENTS] if components else []) + (["gls_t"] if add_gls else []):
        out[c] = np.where(mapped, full[c], np.nan)
    return out


def predict_pairs_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="predict_pairs", description="Score a UniProt pair file (two tab-separated columns, no header).")
    ap.add_argument("--pairs", required=True, help="tab-separated UniProt pair file, no header")
    ap.add_argument("--method", default="zavg_BEF", choices=METHODS, help="default %(default)s")
    ap.add_argument("--depmap", default="datasets/DepMAP/CRISPRGeneEffect.csv", help="CRISPRGeneEffect.csv")
    ap.add_argument("--hgnc", default="datasets/HGNC/hgnc_complete_set.txt", help="hgnc_complete_set.txt")
    ap.add_argument("--out", required=True, help="output TSV (.gz for compressed)")
    ap.add_argument("--components", action="store_true", help="also write the z-scores z_B, z_E, z_F")
    ap.add_argument("--add-gls", action="store_true", help="also write the GLS t-statistic (gls_t)")
    ap.add_argument("--score-name", default="score", help="name of the score column (default %(default)s)")
    ap.add_argument("--unmapped", choices=("floor", "na"), default="floor",
                    help="score of unscorable pairs: 'floor' = min(mapped scores) - 1 (default), 'na' = empty")
    ap.add_argument("--float-format", default="%.6g", help="output float format (default %(default)s)")
    _config_args(ap)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.perf_counter()
    pairs = data.read_pairs(data.ensure_exists(a.pairs, "pair file"))
    X, genes = data.load_depmap(data.ensure_exists(a.depmap, "DepMap CRISPRGeneEffect.csv"))
    hgnc = data.load_hgnc(data.ensure_exists(a.hgnc, "HGNC hgnc_complete_set.txt"))
    out = score_pairs(pairs, genes, hgnc, X, a.method, _config(a), a.components, a.add_gls)
    m = out.mapped.values == 1
    if a.unmapped == "floor":
        out.loc[~m, "score"] = (out.score[m].min() if m.any() else 0.0) - 1.0
    out = out.rename(columns={"score": a.score_name})
    out.to_csv(a.out, sep="\t", index=False, na_rep="", float_format=a.float_format)
    print(json.dumps(dict(method=a.method, out=str(a.out), n_pairs=len(out), n_mapped=int(m.sum()),
                          runtime_s=round(time.perf_counter() - t0, 1))))
    return out


# ------------------------------------------------------------------------------------------ genome-wide
def top_pairs(model: CoessentialityModel, n_top: int) -> tuple[np.ndarray, np.ndarray]:
    """Gene-index pairs (a < b, both BG genes) with the n_top highest zavg_BEF, best first."""
    best_s = np.empty(0)
    best_a = best_b = np.empty(0, np.int64)
    bg = model.bg
    for rows in model._blocks():
        S = model.zavg_block(rows)
        mask = bg[rows, None] & bg[None, :] & (np.arange(model.G)[None, :] > rows[:, None])
        r, c = np.nonzero(mask & (S > (best_s.min() if len(best_s) >= n_top else -np.inf)))
        s = np.r_[best_s, S[r, c]]
        a, b = np.r_[best_a, rows[r]], np.r_[best_b, c]
        if len(s) > n_top:
            keep = np.argpartition(-s, n_top - 1)[:n_top]
            s, a, b = s[keep], a[keep], b[keep]
        best_s, best_a, best_b = s, a, b
    o = np.lexsort((best_b, best_a, -best_s))
    return best_a[o], best_b[o]


def genome_wide_main(argv=None):
    ap = argparse.ArgumentParser(prog="predict_genome_wide",
                                 description="Top-N gene pairs genome-wide by zavg_BEF (genes with <10%% NaN).")
    ap.add_argument("--depmap", default="datasets/DepMAP/CRISPRGeneEffect.csv", help="CRISPRGeneEffect.csv")
    ap.add_argument("--hgnc", default="datasets/HGNC/hgnc_complete_set.txt", help="hgnc_complete_set.txt (symbols and UniProt IDs)")
    ap.add_argument("--top", type=int, default=200_000, help="number of pairs (default %(default)s)")
    ap.add_argument("--out", required=True, help="output TSV (.gz for compressed)")
    ap.add_argument("--float-format", default="%.6g", help="output float format (default %(default)s)")
    _config_args(ap)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.perf_counter()
    X, genes = data.load_depmap(data.ensure_exists(a.depmap, "DepMap CRISPRGeneEffect.csv"))
    hgnc = data.load_hgnc(data.ensure_exists(a.hgnc, "HGNC hgnc_complete_set.txt"))
    model = CoessentialityModel(X, _config(a))
    del X
    ga, gb = top_pairs(model, a.top)
    s, Z = model.zavg(ga, gb)                                 # exact per-pair values
    o = np.lexsort((gb, ga, -s))
    ga, gb, s, Z = ga[o], gb[o], s[o], {c: v[o] for c, v in Z.items()}
    ann = data.gene_annotation(genes, hgnc)
    out = pd.DataFrame(dict(rank=np.arange(1, len(s) + 1),
                            symbol_a=ann.symbol.values[ga], symbol_b=ann.symbol.values[gb],
                            entrez_a=ann.entrez.values[ga], entrez_b=ann.entrez.values[gb],
                            uniprot_a=ann.uniprot.values[ga], uniprot_b=ann.uniprot.values[gb],
                            zavg_BEF=s, **{f"z_{c}": Z[c] for c in COMPONENTS}))
    out.to_csv(a.out, sep="\t", index=False, float_format=a.float_format)
    print(json.dumps(dict(out=str(a.out), n_pairs=len(out), n_genes=int(model.n_bg),
                          runtime_s=round(time.perf_counter() - t0, 1))))
    return out


# ------------------------------------------------------------------------------------------ evaluation
def ht_weights(ca, cb, f: float) -> np.ndarray:
    """Horvitz-Thompson design weights: 1 for within-cluster pairs, 1/f for between-cluster pairs."""
    return np.where(np.asarray(ca) == np.asarray(cb), 1.0, 1.0 / f)


def macro_cluster_ap(y, s, ca, cb, w) -> float:
    """Mean over clusters (with >= 1 positive and >= 1 negative among the pairs touching them) of weighted AP."""
    from sklearn.metrics import average_precision_score
    df = pd.DataFrame(dict(c=np.r_[ca, cb], i=np.r_[np.arange(len(ca)), np.arange(len(cb))])).drop_duplicates()
    aps = []
    for _, g in df.groupby("c"):
        ix = g.i.values
        if y[ix].any() and (~y[ix]).any():
            aps.append(average_precision_score(y[ix], s[ix], sample_weight=w[ix]))
    return float(np.mean(aps)) if aps else float("nan")


def _canon(a, b):
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    lo = np.where(a < b, a, b)
    hi = np.where(a < b, b, a)
    return pd.Series(lo).str.cat(pd.Series(hi), sep="\t").values


def evaluate(scores: pd.DataFrame, labels: pd.DataFrame, score_col: str = "score",
             assignment: pd.DataFrame | None = None, side: str | None = None, f: float | None = None) -> dict:
    """AP / AUROC of ``scores`` (uniprot_a, uniprot_b, score_col, mapped) on ``labels`` (uniprot_a, uniprot_b,
    label). Primary set: pairs with mapped == 1. With an assignment (protein, cluster[, side, mapped]) also the
    HT-weighted AP (between-cluster pairs weighted 1/f) and macro per-cluster AP; ``side`` restricts to pairs with
    both proteins on that side and sets f = fraction of mapped proteins on that side."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    sk = pd.Series(np.arange(len(scores)), index=_canon(scores.uniprot_a, scores.uniprot_b))
    if sk.index.duplicated().any():
        sk = sk[~sk.index.duplicated()]
    lk = _canon(labels.uniprot_a, labels.uniprot_b)
    pos = sk.reindex(lk).values
    missing = int(np.isnan(pos).sum())
    lab = labels.assign(_i=pos)[~np.isnan(pos)]
    sc = scores.iloc[lab._i.astype(int).values].reset_index(drop=True)
    y = lab.label.values.astype(bool)
    s = sc[score_col].values.astype(np.float64)
    mapped = sc["mapped"].values == 1 if "mapped" in sc else np.isfinite(s)
    keep = np.ones(len(y), bool)
    res: dict = dict(n_label_pairs=len(labels), n_missing_from_scores=missing)
    if assignment is not None:
        asg = assignment.set_index("protein")
        ca = asg.cluster.reindex(sc.uniprot_a).values
        cb = asg.cluster.reindex(sc.uniprot_b).values
        if side is not None:
            sa = asg.side.reindex(sc.uniprot_a).values
            sb = asg.side.reindex(sc.uniprot_b).values
            keep &= (sa == side) & (sb == side)
            pm = asg[asg.mapped.astype(str) == "True"] if "mapped" in asg else asg
            f = float((pm.side == side).mean())
        keep &= pd.notna(ca) & pd.notna(cb)
    sel = keep & mapped
    yy, ss = y[sel], s[sel]
    res.update(n_pos=int(yy.sum()), n_neg=int((~yy).sum()), n_unmapped_pos=int((y & keep & ~mapped).sum()),
               n_unmapped_neg=int((~y & keep & ~mapped).sum()), prevalence=float(yy.mean()),
               ap=float(average_precision_score(yy, ss)), auroc=float(roc_auc_score(yy, ss)))
    smin = np.where(keep & ~mapped, ss.min() - 1.0, s)[keep]       # unmapped pairs at the minimum score
    res["ap_incl_unmapped_at_min"] = float(average_precision_score(y[keep], smin))
    if assignment is not None and f is not None:
        w = ht_weights(ca[sel], cb[sel], f)
        res.update(ht_f=f, ht_prevalence=float(w[yy].sum() / w.sum()),
                   ap_ht=float(average_precision_score(yy, ss, sample_weight=w)),
                   macro_cluster_ap=macro_cluster_ap(yy, ss, ca[sel], cb[sel], w))
    return res


def _read_label_files(pos, neg, labels):
    if labels:
        d = pd.read_csv(labels, sep="\t", header=None, names=["uniprot_a", "uniprot_b", "label"], dtype=str)
        if not d.label.iloc[0].lstrip("-").isdigit():         # header row
            d = d.iloc[1:]
        return d.assign(label=d.label.astype(int))
    parts = []
    for path, lab in ((pos, 1), (neg, 0)):
        d = data.read_pairs(path)
        parts.append(d.assign(label=lab))
    return pd.concat(parts, ignore_index=True)


def evaluate_main(argv=None):
    ap = argparse.ArgumentParser(prog="evaluate", description="AP / HT-weighted AP of a score file on labelled pairs.")
    ap.add_argument("--scores", required=True, help="TSV from predict_pairs (uniprot_a, uniprot_b, <score>, mapped)")
    ap.add_argument("--score-col", default="score", help="score column (default %(default)s)")
    ap.add_argument("--pos", help="positive pair file (two tab-separated UniProt columns)")
    ap.add_argument("--neg", help="negative pair file")
    ap.add_argument("--labels", help="alternative: TSV uniprot_a, uniprot_b, label (0/1)")
    ap.add_argument("--assignment", help="CSV with columns protein, cluster [, side, mapped] (HT-AP, macro AP)")
    ap.add_argument("--side", help="restrict to pairs with both proteins on this side (e.g. ift); sets f")
    ap.add_argument("--ht-f", type=float, help="sampling fraction f for HT weights (if --side is not given)")
    ap.add_argument("--out", help="write the result JSON here")
    a = ap.parse_args(argv)
    if not a.labels and not (a.pos and a.neg):
        ap.error("give --labels or both --pos and --neg")
    scores = pd.read_csv(a.scores, sep="\t", dtype={"uniprot_a": str, "uniprot_b": str})
    labels = _read_label_files(a.pos, a.neg, a.labels)
    asg = pd.read_csv(a.assignment, dtype={"protein": str}) if a.assignment else None
    res = evaluate(scores, labels, a.score_col, asg, a.side, a.ht_f)
    txt = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt + "\n")
    print(txt)
    return res
