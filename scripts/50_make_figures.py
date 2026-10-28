"""Manuscript figures (PLOS Computational Biology): Fig1-Fig5 as TIFF (LZW, 400 dpi, RGB), PDF and PNG preview.

Inputs are committed result files only, plus (Fig4) the cached seed-0 pooled out-of-fold dev predictions of Phase B v2
(data_cache/phaseB_v2_oof/, label-free scores) and the dev universe (cluster ids, labels):
  Fig2  results/baselines_v2/<baseline>_cluster_seed{0..4}.json, results/phaseB_v2/runs/<entry>_cluster_seed{0..4}.json,
        results/phaseB_v2/compare.json
  Fig5  results/ift/look01_2026-10-27.json and results/ift/look01_2026-10-27_curves.json (written by
        scripts/final_eval.py --export-curves; this script never loads IFT pairs)
  Fig4  data_cache/phaseB_v2_oof/{zavg_BEF,ref_*}_cluster_seed0.npy + dev universe -> per-cluster AP; derived table
        cached in manuscript/figures/source_data/fig4_*.csv (first computation is logged in results/trials.jsonl;
        pass --recompute to redo it). Checked against results/phaseB_v2/compare.json (seed 0).
  Fig3  results/phaseA/screen.json (seed-0 layers + GLS reference), results/phaseB_v2/runs/zavg_BEF_cluster_seed0.json
Every panel's numbers are also written to manuscript/figures/source_data/*.csv. Deterministic (no randomness; PDF/PNG
metadata dates stripped).

Usage: ./.env/bin/python scripts/50_make_figures.py [--recompute] [--only Fig2 Fig4]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import CACHE, N_FOLDS, RESULTS, SEEDS  # noqa: E402

OUT = ROOT / "manuscript" / "figures"
SRC = OUT / "source_data"
TIFF_DPI = 400
PNG_DPI = 150
WIDTH = 7.5                                   # inches (PLOS max 7.5)
LOOK = RESULTS / "ift" / "look01_2026-10-27.json"
CURVES = RESULTS / "ift" / "look01_2026-10-27_curves.json"

# ------------------------------------------------------------------------------------------------ style
# dataviz reference palette (light), categorical slots in fixed order, one colour per method everywhere.
C = dict(zavg_BEF="#2a78d6", B="#eb6834", GLS="#1baf7a", Spearman="#eda100", Pearson="#e87ba4", F="#008300",
         E="#6250d6")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#f0efec"
LABEL = dict(zavg_BEF="zavg_BEF", B="B", GLS="GLS", Spearman="Spearman", Pearson="Pearson", F="F", E="E")
DEV_ORDER = ["Pearson", "Spearman", "GLS", "F", "B", "E", "zavg_BEF"]
IFT_ORDER = ["zavg_BEF", "B", "GLS", "Spearman", "Pearson"]
DEV_FILE = dict(Pearson=("baselines_v2", "pearson_signed"), Spearman=("baselines_v2", "spearman_signed"),
                GLS=("baselines_v2", "gls_signed"), F=("phaseB_v2/runs", "ref_F"), B=("phaseB_v2/runs", "ref_B"),
                E=("phaseB_v2/runs", "ref_E"), zavg_BEF=("phaseB_v2/runs", "zavg_BEF"))
PB2 = dict(Pearson="ref_PE", Spearman="ref_SP", GLS="ref_G", F="ref_F", B="ref_B", E="ref_E", zavg_BEF="zavg_BEF")
IFT_NAME = dict(zavg_BEF="zavg_BEF", B="B", GLS="gls_signed", Spearman="spearman_signed", Pearson="pearson_signed")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "fig", "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "lines.solid_capstyle": "round",
})


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left", color=INK)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    assert w <= 7.5 + 1e-9 and h <= 8.75 + 1e-9, (name, w, h)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None,
                                                "Creator": None})
    fig.savefig(OUT / f"{name}.png", dpi=PNG_DPI, metadata={"Software": None})
    buf = io.BytesIO()
    fig.savefig(buf, dpi=TIFF_DPI, format="png", metadata={"Software": None})
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    im.save(OUT / f"{name}.tif", format="TIFF", compression="tiff_lzw", dpi=(TIFF_DPI, TIFF_DPI))
    plt.close(fig)
    print(f"{name}: {im.size[0]}x{im.size[1]} px at {TIFF_DPI} dpi ({w:.2f} x {h:.2f} in)")


def write_src(df: pd.DataFrame, name: str):
    SRC.mkdir(parents=True, exist_ok=True)
    df.to_csv(SRC / f"{name}.csv", index=False, float_format="%.6f")


def jload(p):
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------------------------------------ Fig1
def _box(ax, x, y, w, h, text, fc=WASH, ec=AXIS, title=None, tc=INK, lw=0.7, fs=8, accent=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
    if accent is not None:                                   # identity swatch: thin coloured bar on the left edge
        ax.add_patch(FancyBboxPatch((x, y), 1.1, h, boxstyle="round,pad=0,rounding_size=0.5", fc=accent, ec="none"))
    cx, cy = x + w / 2 + (0.5 if accent is not None else 0), y + h / 2
    lh = fs * 1.3                                            # line height in points
    if title:
        nb = text.count("\n") + 1 if text else 0
        ax.annotate(title, (cx, cy), xytext=(0, nb * lh / 2), textcoords="offset points", ha="center",
                    va="center", fontsize=fs + 0.5, fontweight="bold", color=tc)
        if text:
            ax.annotate(text, (cx, cy), xytext=(0, -lh / 2), textcoords="offset points", ha="center", va="center",
                        fontsize=fs, color=tc, linespacing=1.3)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.3)


def _arrow(ax, p1, p2, color=INK2, lw=0.8, style="-|>", ls="-", conn="arc3"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=7, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0, connectionstyle=conn))


def fig1():
    fig = plt.figure(figsize=(WIDTH, 6.9))
    axA = fig.add_axes([0.01, 0.43, 0.98, 0.53])
    axB = fig.add_axes([0.01, 0.01, 0.98, 0.36])
    for ax in (axA, axB):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")
    axA.text(0, 103, "A", fontsize=12, fontweight="bold", va="bottom", color=INK)
    axA.text(4, 103, "Label-free scoring pipeline (fixed configuration; no labels used)", fontsize=9, va="bottom",
             color=INK)
    axB.text(0, 104, "B", fontsize=12, fontweight="bold", va="bottom", color=INK)
    axB.text(4, 104, "Evaluation design (hu.MAP 3.0 training pairs)", fontsize=9, va="bottom", color=INK)

    # ---- A: preprocessing column (top -> bottom)
    a = axA
    x0, w0, h0 = 0, 21, 19
    ys = [79, 54, 29, 2]
    pre = [("Input", "CRISPRGeneEffect\n1,361 cell lines × 18,434 genes"),
           ("Standardise", "z-score each gene\nacross cell lines"),
           ("Covariance", "cell-line covariance Σ\nestimated genome-wide\n(17.5k low-NaN genes)"),
           ("Filter + whiten", "remove top-20 PCs of Σ;\nridge whitening\n(λ + 0.1·λ̄)$^{-1/2}$")]
    for (t, s), y in zip(pre, ys):
        _box(a, x0, y, w0, h0, s, title=t)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        _arrow(a, (x0 + w0 / 2, y1), (x0 + w0 / 2, y2 + h0))
    # ---- views
    xF, wv = 27, 21
    _box(a, xF, 2, wv, 19, "Pearson correlation of\nfiltered, whitened profiles", fc="white", ec=AXIS,
         title="F (first order)", accent=C["F"])
    _arrow(a, (x0 + w0, 11.5), (xF, 11.5))
    _box(a, xF, 40, wv, 40, "each gene keeps its top\n0.25% genome-wide partners\n(≈44 genes), ranked by\nrow-z-balanced F"
         , title="Neighbour lists")
    _arrow(a, (xF + wv / 2, 21), (xF + wv / 2, 40))
    xV = 54
    _box(a, xV, 63, 22, 26, "cosine of sparsified\nneighbour vectors (pair\nitself excluded), then\ngenome-wide row-z",
         fc="white", title="B (second order)", accent=C["B"])
    _box(a, xV, 31, 22, 26, "512-d Laplacian eigenmap\nof the union neighbour\ngraph; cosine similarity",
         fc="white", title="E (graph embedding)", accent=C["E"])
    _arrow(a, (xF + wv, 70), (xV, 76))
    _arrow(a, (xF + wv, 50), (xV, 44))
    # ---- z-scoring and average
    xz, wz = 81, 19
    _box(a, xz, 40, wz, 49, "each view z-scored\nwith fixed moments\nfrom 200k random\ngenome-wide gene\npairs (null)",
         title="z-score")
    _arrow(a, (xV + 22, 76), (xz, 78))
    _arrow(a, (xV + 22, 44), (xz, 62))
    a.plot([xF + wv, 78.5, 78.5], [11.5, 11.5, 46], color=INK2, lw=0.8)      # F joins the z-score step
    _arrow(a, (78.5, 46), (xz, 46))
    a.text(63.5, 15.5, "F is used directly and also\ndefines the neighbour lists", ha="center", va="bottom", fontsize=8,
           color=INK2, linespacing=1.2)
    _box(a, xz, 2, wz, 25, "", fc=C["zavg_BEF"], ec=C["zavg_BEF"])
    a.text(xz + wz / 2, 14.5, "zavg_BEF\n= mean(z$_B$, z$_E$, z$_F$)", ha="center", va="center", fontsize=8,
           color="white", fontweight="bold", linespacing=1.3)
    _arrow(a, (xz + wz / 2, 40), (xz + wz / 2, 27))

    # ---- B: toy positive-pair graph
    b = axB
    b.add_patch(FancyBboxPatch((0, 2), 33, 94, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=AXIS,
                               lw=0.7))
    b.text(16.5, 90, "Positive-pair graph → clusters", ha="center", va="center", fontsize=8, fontweight="bold",
           color=INK)
    rng_free_layout = [  # (cx, cy, n_nodes, radius, side)
        (6.5, 66, 4, 3.2, "dev"), (15, 70, 3, 2.6, "ift"), (25.5, 68, 5, 3.6, "dev"),
        (6.5, 45, 3, 2.6, "ift"), (27, 46, 4, 3.0, "dev")]
    def nodes_at(cx, cy, n, r, rot=0.3):
        ang = rot + 2 * np.pi * np.arange(n) / n
        return np.c_[cx + r * np.cos(ang) * 0.62, cy + r * np.sin(ang) * 1.45]   # axes are wider than tall
    def draw_cluster(pts, side, ec=INK2):
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                b.plot(*zip(pts[i], pts[j]), color=AXIS, lw=0.7, zorder=1)
        fc = INK2 if side == "ift" else "white"
        b.scatter(pts[:, 0], pts[:, 1], s=16, facecolor=fc, edgecolor=INK2, linewidth=0.7, zorder=3)
    for cx, cy, n, r, side in rng_free_layout:
        draw_cluster(nodes_at(cx, cy, n, r), side)
    # giant component split by Louvain into two communities
    g1, g2 = nodes_at(11.5, 22, 5, 3.6, 0.1), nodes_at(21.5, 22, 5, 3.6, 0.5)
    draw_cluster(g1, "dev")
    draw_cluster(g2, "ift")
    b.plot(*zip(g1[0], g2[3]), color=AXIS, lw=0.7, zorder=1)
    b.plot(*zip(g1[1], g2[2]), color=AXIS, lw=0.7, zorder=1)
    b.plot([16.5, 16.5], [10, 34], color=MUTED, lw=0.7, ls=(0, (2, 1.5)))
    b.text(16.5, 6.5, "giant component (144 proteins) split into\n7 Louvain communities → 247 clusters", ha="center", va="center",
           fontsize=8, color=INK2)
    b.scatter([2.5], [82.5], s=16, facecolor="white", edgecolor=INK2, linewidth=0.7)
    b.text(3.8, 82.5, "dev protein", va="center", fontsize=8, color=INK2)
    b.scatter([18.5], [82.5], s=16, facecolor=INK2, edgecolor=INK2, linewidth=0.7)
    b.text(19.8, 82.5, "IFT protein", va="center", fontsize=8, color=INK2)
    b.text(16.5, 56.5, "241 connected components", ha="center", va="center", fontsize=8, color=INK2)

    # ---- B: split boxes
    _arrow(b, (33, 70), (38, 74))
    _arrow(b, (33, 30), (38, 26))
    _box(b, 38, 56, 30, 38, "≈25% of positives by stratified\ncluster assignment; 57 clusters,\n847 pos / 40,786 neg"
         "\nsingle pre-specified look", fc="white", title="Internal final test (IFT)")
    _box(b, 38, 6, 30, 38, "190 clusters, 2,299 pos / 320,229 neg\n5-fold cluster-grouped CV,\n5 seed repeats;"
         "\nall method selection here", fc="white", title="Development (dev)")
    b.text(53, 50, "cross pairs (dev × IFT protein) dropped", ha="center", va="center", fontsize=8, color=INK2)
    # ---- B: HT weighting note
    _box(b, 72, 6, 28, 88, "AP over held-out pairs, each\npair weighted by 1 / its\ninclusion probability:\n\n"
         "within-cluster pair: 1\nbetween-cluster pair: 1/f\n(dev fold f = 1/5; IFT f = 0.265)\n\n"
         "→ evaluated pair mix matches\nthe full universe (natural\nprevalence 0.53%)", fc=WASH,
         title="HT-weighted AP (primary)")
    _arrow(b, (68, 75), (72, 70))
    _arrow(b, (68, 25), (72, 30))
    save(fig, "Fig1")


# ------------------------------------------------------------------------------------------------ Fig2
def dev_seed_values():
    rows = []
    for m in DEV_ORDER:
        d, stem = DEV_FILE[m]
        for s in SEEDS:
            r = jload(RESULTS / d / f"{stem}_cluster_seed{s}.json")
            pb = jload(RESULTS / "phaseB_v2" / "runs" / f"{PB2[m]}_cluster_seed{s}.json")
            assert abs(pb["summary"]["ap_ht"]["mean"] - r["summary"]["ap_ht"]["mean"]) < 1e-12, (m, s)
            rows.append(dict(method=m, seed=s, ap_ht=r["summary"]["ap_ht"]["mean"],
                             macro_cluster_ap=pb["summary"]["macro_cluster_ap"]["mean"],
                             source=f"results/{d}/{stem}_cluster_seed{s}.json"))
    df = pd.DataFrame(rows)
    cmp = jload(RESULTS / "phaseB_v2" / "compare.json")
    df["ap_ht_no_top10_pooled_oof"] = [cmp["seeds"][str(s)]["no_top10"]["ap"][PB2[m]]["ap"]
                                       for m, s in zip(df.method, df.seed)]
    return df, cmp


def _dots_and_mean(ax, df, col, order, jitter=0.09):
    off = (np.arange(len(SEEDS)) - 2) * jitter
    for i, m in enumerate(order):
        v = df[df.method == m].sort_values("seed")[col].values
        mu = v.mean()
        ax.plot([i - 0.32, i + 0.32], [mu, mu], color=C[m], lw=2.0, solid_capstyle="round", zorder=3)
        ax.scatter(i + off, v, s=14, color=C[m], edgecolor="white", linewidth=0.6, zorder=4)
        yield i, m, max(mu, v.max())


def fig2():
    df, cmp = dev_seed_values()
    write_src(df, "Fig2A_C_dev_per_seed")
    fig = plt.figure(figsize=(WIDTH, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 0.95], hspace=0.5, wspace=0.32,
                          left=0.08, right=0.985, top=0.95, bottom=0.1)
    # A
    ax = fig.add_subplot(gs[0, 0])
    for i, m, top in _dots_and_mean(ax, df, "ap_ht", DEV_ORDER):
        mu = df[df.method == m].ap_ht.mean()
        ax.text(i, top + 0.012, f"{mu:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(DEV_ORDER)), [LABEL[m] for m in DEV_ORDER], rotation=30, ha="right",
                  rotation_mode="anchor")
    ax.set_ylabel("HT-AP (dev, fold mean)")
    ax.set_ylim(0.15, 0.56)
    ax.set_xlim(-0.6, len(DEV_ORDER) - 0.4)
    ygrid(ax)
    ax.axvline(2.5, color=GRID, lw=0.8)
    ax.text(1.0, 0.535, "baselines", ha="center", va="center", fontsize=8, color=MUTED)
    ax.text(4.5, 0.535, "this work (label-free)", ha="center", va="center", fontsize=8, color=MUTED)
    panel(ax, "A", x=-0.13)
    # B
    ax = fig.add_subplot(gs[0, 1])
    rows = []
    for k, (m, key) in enumerate((("zavg_BEF", "zavg_BEF-ref_G"), ("B", "ref_B-ref_G"))):
        for s in SEEDS:
            d = cmp["seeds"][str(s)]["all"]["diff"][key]
            rows.append(dict(method=m, seed=s, delta=d["diff"], ci_low=d["ci_low"], ci_high=d["ci_high"]))
            x = s + (-0.14 if k == 0 else 0.14)
            ax.plot([x, x], [d["ci_low"], d["ci_high"]], color=C[m], lw=1.2, zorder=2)
            ax.scatter([x], [d["diff"]], s=18, color=C[m], edgecolor="white", linewidth=0.6, zorder=3)
    write_src(pd.DataFrame(rows), "Fig2B_delta_vs_GLS")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(range(5), [str(s) for s in SEEDS])
    ax.set_xlabel("seed (split repeat)")
    ax.set_ylabel("Δ HT-AP vs GLS\n(pooled OOF, 95% CI)")
    ax.set_ylim(-0.02, 0.33)
    ygrid(ax)
    ax.legend(handles=[Line2D([], [], color=C[m], marker="o", lw=1.2, ms=4, mec="white", label=f"{LABEL[m]} − GLS")
                       for m in ("zavg_BEF", "B")], loc="upper left", ncol=2, handlelength=1.4, columnspacing=1.0,
              borderaxespad=0.1)
    panel(ax, "B", x=-0.2)
    # C (two sub-panels)
    sub = gs[1, :].subgridspec(1, 2, wspace=0.22)
    for j, (col, lab, letter) in enumerate((("macro_cluster_ap", "macro per-cluster AP (fold mean)", "C"),
                                            ("ap_ht_no_top10_pooled_oof", "HT-AP without top-10 clusters\n(pooled OOF)",
                                             None))):
        ax = fig.add_subplot(sub[0, j])
        for i, m, top in _dots_and_mean(ax, df, col, DEV_ORDER, jitter=0.08):
            mu = df[df.method == m][col].mean()
            ax.text(i, top + 0.01, f"{mu:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(DEV_ORDER)), [LABEL[m] for m in DEV_ORDER], rotation=30, ha="right",
                      rotation_mode="anchor")
        ax.set_ylabel(lab)
        ax.set_ylim(0.15, 0.47)
        ax.set_xlim(-0.6, len(DEV_ORDER) - 0.4)
        ax.axvline(2.5, color=GRID, lw=0.8)
        ygrid(ax)
        if letter:
            panel(ax, letter, x=-0.17)
    save(fig, "Fig2")
    return df


# ------------------------------------------------------------------------------------------------ Fig3
def fig3():
    look, cur = jload(LOOK), jload(CURVES)
    assert cur["look_sha256"] == __import__("hashlib").sha256(LOOK.read_bytes()).hexdigest(), "curves from other look"
    M = look["metrics"]
    for m in IFT_ORDER:
        c = cur["curves"][IFT_NAME[m]]
        assert abs(c["ap_from_curve"] - M[IFT_NAME[m]]["ap_ht"]) < 1e-9, m
    fig = plt.figure(figsize=(WIDTH, 3.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.1, 0.9], wspace=0.42, left=0.07, right=0.99, top=0.83,
                          bottom=0.19)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C[m], label=LABEL[m]) for m in IFT_ORDER],
               loc="upper left", bbox_to_anchor=(0.46, 0.995), ncol=5, handlelength=1.0, handleheight=0.9,
               columnspacing=1.2, borderaxespad=0.2)
    # A: PR curves
    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for m in reversed(IFT_ORDER):                                     # primary drawn last (on top)
        c = cur["curves"][IFT_NAME[m]]
        r, p = np.r_[0.0, c["recall"]], np.r_[c["precision"][0], c["precision"]]
        ax.plot(r, p, drawstyle="steps-pre", color=C[m], lw=1.3 if m == "zavg_BEF" else 1.0, zorder=3)
        rows += [dict(method=m, recall=a, precision=b) for a, b in zip(c["recall"], c["precision"])]
    write_src(pd.DataFrame(rows), "Fig5A_IFT_PR_curves")
    ax.axhline(cur["ht_prevalence"], color=MUTED, lw=0.7, zorder=1)
    ax.text(0.99, 0.065, f"HT prevalence {cur['ht_prevalence']:.4f}", ha="right", va="bottom", fontsize=8,
            color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("HT-weighted precision (IFT)")
    ygrid(ax)
    ax.legend(handles=[Line2D([], [], color=C[m], lw=1.5, label=f"{LABEL[m]}  {M[IFT_NAME[m]]['ap_ht']:.3f}")
                       for m in IFT_ORDER], title="HT-AP", title_fontsize=8, loc="upper right", handlelength=1.4,
              borderaxespad=0.1, labelspacing=0.3)
    panel(ax, "A", x=-0.17)
    # B: strata
    ax = fig.add_subplot(gs[0, 1])
    strata = [("both_nonessential", "both non-\nessential"), ("mixed", "mixed"), ("both_essential", "both\nessential")]
    bw = 0.15
    rows = []
    for k, m in enumerate(IFT_ORDER):
        v = [M[IFT_NAME[m]][f"ap_ht_{s}"] for s, _ in strata]
        x = np.arange(3) + (k - 2) * bw
        ax.bar(x, v, width=bw * 0.86, color=C[m], zorder=3)
        rows += [dict(method=m, stratum=s, ap_ht=a, n_pos=M[IFT_NAME[m]][f"n_pos_{s}"],
                      n_neg=M[IFT_NAME[m]][f"n_neg_{s}"]) for (s, _), a in zip(strata, v)]
    write_src(pd.DataFrame(rows), "Fig5B_IFT_strata")
    z = M["zavg_BEF"]
    ax.set_xticks(range(3), [f"{lab}\n({z[f'n_pos_{s}']} pos)" for s, lab in strata])
    ax.set_ylabel("HT-AP (IFT)")
    ax.set_ylim(0, 0.45)
    ygrid(ax)
    panel(ax, "B", x=-0.2)
    # C: macro / no largest component
    ax = fig.add_subplot(gs[0, 2])
    mets = [("macro_cluster_ap", "macro per-\ncluster AP"), ("ap_ht_no_largest_component", "HT-AP w/o\nlargest comp.")]
    rows = []
    for k, m in enumerate(IFT_ORDER):
        v = [M[IFT_NAME[m]][q] for q, _ in mets]
        x = np.arange(2) + (k - 2) * bw
        ax.bar(x, v, width=bw * 0.86, color=C[m], zorder=3)
        rows += [dict(method=m, metric=q, value=a) for (q, _), a in zip(mets, v)]
    write_src(pd.DataFrame(rows), "Fig5C_IFT_macro_nolargest")
    ax.set_xticks(range(2), [f"{lab}" for _, lab in mets])
    ax.set_ylabel("AP (IFT)")
    ax.set_ylim(0, 0.52)
    ygrid(ax)
    panel(ax, "C", x=-0.28)
    save(fig, "Fig5")


# ------------------------------------------------------------------------------------------------ Fig4
FIG4_METHODS = ["Pearson", "Spearman", "GLS", "F", "B", "E", "zavg_BEF"]


def fig4_data(recompute=False):
    f_cl, f_sub = SRC / "fig4_per_cluster_seed0.csv", SRC / "fig4_top10_split_seed0.csv"
    if f_cl.exists() and f_sub.exists() and not recompute:
        return pd.read_csv(f_cl), pd.read_csv(f_sub)
    from src import evaluate, metrics
    from src.universe import load_dev_universe
    tid = uuid.uuid4().hex[:12]
    evaluate.log_trial("start", tid, tag="manuscript_fig4_source_data", universe="dev", seed=SEEDS[0],
                       git=evaluate.git_state(), scorer=dict(name="figure: per-cluster AP of committed Phase B v2 "
                                                                   "seed-0 pooled-OOF scores (no selection)",
                                                              label_free=True))
    try:
        u = load_dev_universe()
        S = {m: np.load(CACHE / "phaseB_v2_oof" / f"{PB2[m]}_cluster_seed0.npy") for m in FIG4_METHODS}
        idx = np.flatnonzero(np.isfinite(S["zavg_BEF"]))
        for v in S.values():
            assert np.array_equal(np.flatnonzero(np.isfinite(v)), idx)
        y = u.y[idx].astype(bool)
        ca, cb = u.col("cluster_a")[idx], u.col("cluster_b")[idx]
        w = metrics.ht_weights(ca, cb, 1.0 / N_FOLDS)
        cmp = jload(RESULTS / "phaseB_v2" / "compare.json")["seeds"]["0"]
        rows = []
        for c, ix in metrics._groups_touching(ca, cb).items():
            if not (y[ix].any() and (~y[ix]).any()):
                continue
            r = dict(cluster=int(c), n_pos=int(y[ix].sum()), n_pos_within=int((y[ix] & (ca[ix] == cb[ix])).sum()),
                     n_pairs=int(len(ix)), top10=int(c) in cmp["top10_clusters"])
            for m in FIG4_METHODS:
                r[f"ap_{m}"] = metrics.weighted_ap(y[ix], S[m][idx][ix], w[ix])
            rows.append(r)
        cl = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
        for m in FIG4_METHODS:                 # consistency with committed compare.json (seed 0)
            assert abs(cl[f"ap_{m}"].mean() - cmp["macro_cluster_ap_pooled"][PB2[m]]) < 1e-9, m
        top = cmp["top10_clusters"]
        touch = np.isin(ca, top) | np.isin(cb, top)
        sub = []
        for name, msk in (("all pairs", np.ones(len(idx), bool)), ("top-10 clusters only", touch),
                          ("without top-10 clusters", ~touch)):
            for m in FIG4_METHODS:
                ap = metrics.weighted_ap(y[msk], S[m][idx][msk], w[msk])
                ref = {"all pairs": cmp["all"], "without top-10 clusters": cmp["no_top10"]}.get(name)
                if ref is not None:
                    assert abs(ap - ref["ap"][PB2[m]]["ap"]) < 1e-9, (name, m)
                sub.append(dict(subset=name, method=m, ap_ht=ap, n_pos=int(y[msk].sum()), n_pairs=int(msk.sum())))
        sub = pd.DataFrame(sub)
        write_src(cl, "fig4_per_cluster_seed0")
        write_src(sub, "fig4_top10_split_seed0")
    except BaseException as e:
        evaluate.log_trial("end", tid, tag="manuscript_fig4_source_data", status="error", error=repr(e)[:300])
        raise
    evaluate.log_trial("end", tid, tag="manuscript_fig4_source_data", status="ok",
                       result="manuscript/figures/source_data/fig4_*.csv")
    return cl, sub


def _kernel_trend(x, y, grid, bw=0.3):
    lx = np.log10(x)
    out = []
    for g in np.log10(grid):
        k = np.exp(-0.5 * ((lx - g) / bw) ** 2)
        out.append(np.sum(k * y) / np.sum(k))
    return np.array(out)


def fig4(recompute=False):
    cl, sub = fig4_data(recompute)
    fig = plt.figure(figsize=(WIDTH, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.5, wspace=0.12, left=0.085, right=0.985, top=0.89,
                          bottom=0.09)
    grid = np.logspace(np.log10(cl.n_pos.min()), np.log10(cl.n_pos.max()), 80)
    ylim = 0.05 + max(np.abs(cl[f"ap_{m}"] - cl["ap_GLS"]).max() for m in ("zavg_BEF", "B"))
    fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=4, color=MUTED, mec="white", label="cluster"),
                        Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", mec=MUTED, mew=1.1,
                               label="one of the 10 largest clusters"),
                        Line2D([], [], color=INK, lw=1.3, label="Gaussian-kernel trend (log size)")],
               loc="upper right", bbox_to_anchor=(0.985, 1.0), ncol=3, handlelength=1.3, columnspacing=1.2,
               borderaxespad=0.2)
    axs = []
    for j, m in enumerate(("zavg_BEF", "B")):
        ax = fig.add_subplot(gs[0, j], sharey=axs[0] if axs else None)
        axs.append(ax)
        d = cl[f"ap_{m}"] - cl["ap_GLS"]
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        nt = ~cl.top10
        ax.scatter(cl.n_pos[nt], d[nt], s=13, color=C[m], alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
        ax.scatter(cl.n_pos[cl.top10], d[cl.top10], s=22, facecolor="white", edgecolor=C[m], linewidth=1.1,
                   zorder=4)
        ax.plot(grid, _kernel_trend(cl.n_pos.values, d.values, grid), color=INK, lw=1.3, zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("positive pairs touching the cluster (dev, seed 0)")
        ax.set_title(f"{LABEL[m]} − GLS", fontsize=9, color=INK, loc="left")
        ax.set_ylim(-ylim, ylim)
        ygrid(ax)
        ax.text(0.98, 0.03, f"{len(cl)} clusters: Δ > 0 in {(d > 1e-12).sum()}, Δ < 0 in {(d < -1e-12).sum()};"
                f"\nmean Δ {d.mean():+.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK2)
        if j == 0:
            ax.set_ylabel("Δ per-cluster AP vs GLS")
            panel(ax, "A", x=-0.13)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
            panel(ax, "B", x=-0.06)
    # C: dumbbell, top-10 clusters only vs without them
    ax = fig.add_subplot(gs[1, :])
    order = FIG4_METHODS
    for i, m in enumerate(order):
        a = sub[(sub.subset == "top-10 clusters only") & (sub.method == m)].ap_ht.item()
        b = sub[(sub.subset == "without top-10 clusters") & (sub.method == m)].ap_ht.item()
        ax.plot([b, a], [i, i], color=AXIS, lw=1.2, zorder=1)
        ax.scatter([b], [i], s=30, color=C[m], edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter([a], [i], s=30, facecolor="white", edgecolor=C[m], linewidth=1.3, zorder=3)
        lo, hi = min(a, b), max(a, b)
        ax.text(lo - 0.012, i, f"{lo:.3f}", ha="right", va="center", fontsize=8, color=INK)
        ax.text(hi + 0.012, i, f"{hi:.3f}", ha="left", va="center", fontsize=8, color=INK)
    n_top = sub[sub.subset == "top-10 clusters only"].n_pos.iloc[0]
    n_rest = sub[sub.subset == "without top-10 clusters"].n_pos.iloc[0]
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order])
    ax.set_xlim(0.1, 0.72)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("HT-AP (dev pooled OOF, seed 0)")
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=5, color=MUTED, mec="white",
                              label=f"pairs not touching the 10 largest clusters ({n_rest:,} pos)"),
                       Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=MUTED, mew=1.3,
                              label=f"pairs touching the 10 largest clusters ({n_top:,} pos)")],
              loc="lower right", handlelength=1.0, borderaxespad=0.1, labelspacing=0.3)
    panel(ax, "C", x=-0.085, y=1.02)
    save(fig, "Fig4")
    return cl, sub


# ------------------------------------------------------------------------------------------------ Fig5
FIG5_STEPS = [
    ("spec_k0_t0", "GLS-like whitening (k = 0 PCs removed, τ = 0)"),
    ("spec_k0_t0.1", "+ ridge regularisation (τ = 0.1)"),
    ("spec_k0_t0.1_std", "+ gene standardisation before Σ"),
    ("spec_k20_t0.1_std", "+ remove top-20 PCs  = F"),
    ("socos_q0.25[spec_k20_t0.1_std]", "second order: neighbour cosine (q = 0.25%)"),
    ("rowz[socos_q0.25[spec_k20_t0.1_std]]", "+ genome-wide row-z  = B"),
    ("zavg_BEF", "+ z-average with E and F  = zavg_BEF"),
]


def fig5():
    scr = jload(RESULTS / "phaseA" / "screen.json")
    lay = {L["name"]: L for L in scr["layers"]}
    gls = {b["name"]: b for b in scr["baselines"]}["BASELINE gls_signed"]
    z = jload(RESULTS / "phaseB_v2" / "runs" / "zavg_BEF_cluster_seed0.json")
    assert z["split"]["fingerprint"] == scr["split_fingerprint"]
    rows = []
    for key, lab in FIG5_STEPS:
        if key == "zavg_BEF":
            v, sd, src = z["summary"]["ap_ht"]["mean"], z["summary"]["ap_ht"]["sd"], \
                "results/phaseB_v2/runs/zavg_BEF_cluster_seed0.json"
        else:
            v, sd, src = lay[key]["ap_ht"], lay[key]["ap_ht_sd"], lay[key]["result"]
        rows.append(dict(step=lab, layer=key, ap_ht=v, sd_over_folds=sd, source=src))
    df = pd.DataFrame(rows)
    df["delta_vs_previous"] = df.ap_ht.diff()
    df.loc[len(df)] = dict(step="reference: GLS signed baseline", layer="gls_signed", ap_ht=gls["ap_ht"],
                           sd_over_folds=gls["ap_ht_sd"], source=gls["result"], delta_vs_previous=np.nan)
    write_src(df, "Fig3_ablation_seed0")
    steps = df.iloc[:-1]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
    fig.subplots_adjust(left=0.41, right=0.97, top=0.9, bottom=0.15)
    n = len(steps)
    ys = np.arange(n)[::-1]
    x0 = 0.26
    for i, (y, r) in enumerate(zip(ys, steps.itertuples())):
        col = C["zavg_BEF"] if r.layer == "zavg_BEF" else C["B"] if r.layer.startswith("rowz[socos") else \
            C["F"] if r.layer == "spec_k20_t0.1_std" else "#86b6ef"
        if i > 0:                                          # step: segment from the previous value to this one
            prev = steps.ap_ht.iloc[i - 1]
            ax.plot([prev, prev], [y + 1, y], color=MUTED, lw=0.7, ls=(0, (2, 1.5)), zorder=2)
            ax.plot([prev, r.ap_ht], [y, y], color=col, lw=4.0, solid_capstyle="round", zorder=3)
        ax.scatter([r.ap_ht], [y], s=34, color=col, edgecolor="white", linewidth=0.9, zorder=4)
        txt = f"{r.ap_ht:.3f}" + ("" if i == 0 else f"  ({r.ap_ht - steps.ap_ht.iloc[i - 1]:+.3f})")
        ax.text(r.ap_ht + 0.006, y, txt, va="center", ha="left", fontsize=8, color=INK,
                bbox=dict(fc="white", ec="none", pad=0.4))
    ax.axvline(gls["ap_ht"], color=INK2, lw=0.8, zorder=1)
    ax.text(gls["ap_ht"], n - 0.35, f"GLS baseline {gls['ap_ht']:.3f}", ha="center", va="bottom", fontsize=8,
            color=INK2)
    ax.set_yticks(ys, steps.step.tolist())
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(x0, 0.52)
    ax.set_ylim(-0.6, n - 0.1)
    ax.set_xlabel("HT-AP (dev, seed 0, mean over 5 folds) — selection-stage results, optimistic")
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "Fig3")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="recompute Fig4 source data (logged trial)")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    todo = {"Fig1": fig1, "Fig2": fig2, "Fig3": fig3, "Fig4": lambda: fig4(a.recompute), "Fig5": fig5}
    for k, f in todo.items():
        if a.only is None or k in a.only:
            f()


if __name__ == "__main__":
    main()
