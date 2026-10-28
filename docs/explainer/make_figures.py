"""Static figures for the explainer Markdown (docs/explainer/figures/*.png).

Numbers are read from committed result files (manuscript/figures/source_data/, results/ift/) or from
data/worked_example.json (written by worked_example.py). The journey milestones are listed in JOURNEY below with
the file each value comes from.
Usage: ./.env/bin/python docs/explainer/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "manuscript" / "figures" / "source_data"
FIG = ROOT / "docs" / "explainer" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GREY, LIGHT, INK, MUTED = "#8a8984", "#d6d5cf", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 160, "savefig.bbox": "tight",
                     "savefig.facecolor": "white"})
LABEL = {"Pearson": "Pearson", "Spearman": "Spearman", "GLS": "GLS", "F": "F (direct)",
         "B": "B (shared neighbours)", "E": "E (graph embedding)", "zavg_BEF": "Final score"}


def color(m):
    return BLUE if m == "zavg_BEF" else (AQUA if m in ("F", "B", "E") else GREY)


def fig_results():
    dev = pd.read_csv(SRC / "Fig2A_C_dev_per_seed.csv")
    ift = json.loads((ROOT / "results/ift/look01_2026-10-27.json").read_text())
    ift_ap = {"Pearson": ift["metrics"]["pearson_signed"]["ap_ht"], "Spearman": ift["metrics"]["spearman_signed"]["ap_ht"],
              "GLS": ift["metrics"]["gls_signed"]["ap_ht"], "B": ift["metrics"]["B"]["ap_ht"],
              "zavg_BEF": ift["metrics"]["zavg_BEF"]["ap_ht"]}
    order = ["Pearson", "Spearman", "GLS", "F", "B", "E", "zavg_BEF"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(order))[::-1]
    ax = axes[0]
    for yi, m in zip(y, order):
        v = dev[dev.method == m].ap_ht
        ax.barh(yi, v.mean(), color=color(m), height=0.62)
        ax.scatter(v, np.full(len(v), yi), s=9, color=INK, zorder=3, linewidths=0)
        ax.text(max(v.max(), v.mean()) + 0.012, yi, f"{v.mean():.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [LABEL[m] for m in order])
    ax.set_title("Development set (5 cross-validation repeats)", fontsize=10, loc="left", color=INK)
    ax = axes[1]
    for yi, m in zip(y, order):
        if m in ift_ap:
            ax.barh(yi, ift_ap[m], color=color(m), height=0.62)
            ax.text(ift_ap[m] + 0.012, yi, f"{ift_ap[m]:.3f}", va="center", fontsize=9, color=INK)
        else:
            ax.text(0.01, yi, "not tested (not a candidate)", va="center", fontsize=8, color=MUTED)
    d = ift["bootstrap"]["diff"]["zavg_BEF-gls_signed"]
    ax.set_title("Internal final test (one look, held-out complexes)", fontsize=10, loc="left", color=INK)
    ax.text(0.99, 0.97, f"Final − GLS = +{d['diff']:.3f}\n95% CI [+{d['ci_low']:.3f}, +{d['ci_high']:.3f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK)
    for ax in axes:
        ax.set_xlim(0, 0.56)
        ax.set_xlabel("weighted average precision (HT-AP)")
        ax.axvline(0.0053, color=MUTED, lw=0.8, ls=":")
    fig.savefig(FIG / "results.png")
    plt.close(fig)


def fig_ablation():
    a = pd.read_csv(SRC / "Fig3_ablation_seed0.csv")
    steps = a[~a.step.str.startswith("reference")].reset_index(drop=True)
    gls = float(a[a.step.str.startswith("reference")].ap_ht.iloc[0])
    names = ["GLS-like whitening", "+ ridge term", "+ standardise genes first", "+ remove top 20 components  → F",
             "second order: shared neighbours", "+ per-gene background (row-z)  → B", "+ average with E and F  → final"]
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    prev = None
    for i, (n, v) in enumerate(zip(names, steps.ap_ht)):
        yi = len(names) - 1 - i
        start = steps.ap_ht.iloc[0] if prev is None else prev
        if prev is None:
            ax.barh(yi, v, color=GREY, height=0.6)
        else:
            ax.barh(yi, start, color=LIGHT, height=0.6)
            ax.barh(yi, v - start, left=start, color=BLUE if v > start else ORANGE, height=0.6)
        ax.text(v + 0.006, yi, f"{v:.3f}" + ("" if prev is None else f"  (+{v - prev:.3f})"), va="center", fontsize=8.5)
        prev = v
    ax.axvline(gls, color=INK, lw=0.9, ls="--")
    ax.text(gls, len(names) - 0.35, f" GLS baseline {gls:.3f}", fontsize=8, color=INK, va="bottom")
    ax.set_yticks(range(len(names))[::-1], names)
    ax.set_xlim(0.0, 0.53)
    ax.set_xlabel("development HT-AP (split seed 0, mean of 5 folds)")
    fig.savefig(FIG / "ablation.png")
    plt.close(fig)


def fig_size():
    c = pd.read_csv(SRC / "fig4_per_cluster_seed0.csv")
    c = c[c.n_pos > 0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, m, title in [(axes[0], "ap_B", "B alone − GLS"), (axes[1], "ap_zavg_BEF", "Final score − GLS")]:
        dlt = c[m] - c.ap_GLS
        x = np.log10(c.n_pos)
        ax.scatter(x, dlt, s=12, color=np.where(c.top10, ORANGE, BLUE), alpha=0.75, linewidths=0)
        xs = np.linspace(x.min(), x.max(), 80)
        w = np.exp(-0.5 * ((xs[:, None] - x.values[None, :]) / 0.3) ** 2)
        ax.plot(xs, (w * dlt.values).sum(1) / w.sum(1), color=INK, lw=1.6)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{title}: better in {(dlt > 0).sum()}, worse in {(dlt < 0).sum()} clusters", fontsize=9.5, loc="left")
        ax.set_xticks([0, 1, 2], ["1", "10", "100"])
        ax.set_xlabel("positive pairs in the complex cluster (log scale)")
    axes[0].set_ylabel("per-cluster AP difference")
    axes[1].scatter([], [], color=ORANGE, s=12, label="10 largest clusters")
    axes[1].scatter([], [], color=BLUE, s=12, label="other clusters")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(FIG / "gain_by_size.png")
    plt.close(fig)


def fig_worked():
    w = json.loads((ROOT / "docs/explainer/data/worked_example.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"wspace": 0.35})
    for ax, key, title in [(axes[0], "smc56_pairs", "SMC5/6 complex (28 pairs)"), (axes[1], "bbsome_pairs", "BBSome (28 pairs)")]:
        recs = w[key]
        meths = ["pearson", "gls", "F", "B", "E", "zavg"]
        labels = ["Pearson", "GLS", "F", "B", "E", "Final"]
        for i, m in enumerate(meths):
            r = np.array([x["best_rank_in_row"][m] for x in recs])
            jitter = (np.arange(len(r)) % 7 - 3) * 0.05
            ax.scatter(np.full(len(r), i) + jitter, r, s=14, color=BLUE if m == "zavg" else (AQUA if m in "FBE" else GREY),
                       alpha=0.8, linewidths=0)
            ax.text(i, 0.42, f"{int((r <= 44).sum())}/28", ha="center", fontsize=8, color=INK)
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.axhline(44, color=MUTED, lw=0.8, ls=":")
        ax.set_ylim(3e4, 0.3)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_title(title + "  (numbers: pairs in the top 44)", fontsize=10, loc="left")
    axes[0].set_ylabel("partner's rank among ~18,400 genes\n(better of the two directions; 1 = top)")
    axes[0].text(5.6, 44, "top 44", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    fig.savefig(FIG / "worked_example_ranks.png")
    plt.close(fig)


def fig_journey():
    j = json.loads((ROOT / "docs/explainer/data/journey.json").read_text())
    h = lambda t: int(t[:2]) + int(t[3:]) / 60
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    for i, p in enumerate(j["phases"]):
        if i % 2 == 0:
            ax.axvspan(h(p["start"]), h(p["end"]), color="#eef2f8", zorder=0)
        ax.text(h(p["start"]) + 0.04, 0.515, p["label"], fontsize=7.5, color=MUTED, va="bottom")
    ax.axhline(j["gls"], color=GREY, ls="--", lw=1)
    ax.text(17.95, j["gls"] + 0.004, f"GLS {j['gls']}", ha="right", fontsize=8, color=MUTED)
    xs = [h(b["t"]) for b in j["best"]] + [h(j["range"][1])]
    ys = [b["v"] for b in j["best"]] + [j["best"][-1]["v"]]
    ax.step(xs, ys, where="post", color=BLUE, lw=2)
    for b in j["best"]:
        ax.plot(h(b["t"]), b["v"], "o", color=BLUE, ms=4)
        if b["label"]:
            ax.text(h(b["t"]) + 0.04, b["v"] + 0.006, f"{b['label']} {b['v']:.3f}", fontsize=8.5, color=INK)
    for kind, y, c in [("tp", 0.17, BLUE), ("dead", 0.12, ORANGE)]:
        for e in [e for e in j["events"] if e["kind"] == kind]:
            ax.plot(h(e["t"]), y, "o", ms=11, mfc=c if kind == "tp" else "white", mec=c, mew=1.6, clip_on=False)
            if e.get("n"):
                ax.text(h(e["t"]), y, e["n"], ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    ax.text(11.97, 0.17, "turning points", ha="right", va="center", fontsize=8, color=MUTED)
    ax.text(11.97, 0.12, "dead ends", ha="right", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(12, 18)
    ax.set_ylim(0.1, 0.53)
    ax.set_yticks([0.3, 0.4, 0.5])
    ax.set_xticks(range(12, 19), [f"{t}:00" for t in range(12, 19)])
    ax.set_ylabel("best dev HT-AP")
    fig.savefig(FIG / "journey.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_results()
    fig_ablation()
    fig_size()
    fig_journey()
    if (ROOT / "docs/explainer/data/worked_example.json").exists():
        fig_worked()
    print("figures written to", FIG.relative_to(ROOT))
