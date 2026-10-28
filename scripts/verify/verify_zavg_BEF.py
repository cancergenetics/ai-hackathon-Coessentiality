"""Independent verification of the Phase-B v2 primary candidate zavg_BEF.

zavg_BEF = equal-weight mean of genome-wide-null z-scores of
  B = rowz[socos_q0.25[spec_k20_t0.1_std]]               (re-implemented in scripts/verify/verify_phaseA_best.py)
  E = emb_lap512[knn_union_lin_q0.25[spec_k20_t0.1_std]]  (re-implemented HERE from the src/layers_graph.py docstrings)
  F = spec_k20_t0.1_std                                    (verify_phaseA_best.spectral_profiles)

Does NOT import src/layers*.py, src/predict.py, src/metrics.py, src/evaluate.py or src/supervised.py. Project imports:
src.splits.make_split (fold assignment) and the previous verifier's independent helpers. Dev pairs are re-derived
from the committed protein assignment (SHA-256 checked); rows with any IFT-side protein are dropped on load and no
metric is computed on them. The producer's predictor (scripts/predict_pairs.py) is run as a black-box subprocess on
the dev pair IDs only, for the score comparison.

E (from the docstrings): BG = genes with < 10% NaN. Base = unit-norm spectral profiles F (k = 20, tau = 0.1, std).
Each gene keeps its top K = round(q/100 |BG|) BG neighbours (self excluded) by the balanced z
B_aj = (z_aj + z_ja)/2 (row moments over BG \\ self), weight max(B, 0) ('lin'). Union symmetrisation
max(W, W') with W' restricted to BG columns; non-BG rows keep their directed edges. N = D^-1/2 A D^-1/2.
Laplacian eigenmap: top d+1 eigenpairs (largest algebraic) of N_BG,BG, leading one dropped; non-BG genes by the
Nystrom extension N_g,BG U / lambda. Score = cosine of embedding rows (no partner exclusion).
Own deviations from the producer (on purpose, to test robustness of the construction): N kept in float64 (not
float32), and a different ARPACK start vector (seed V0_SEED instead of SEEDS[0]).

Usage: ./.env/bin/python scripts/verify/verify_zavg_BEF.py [--skip-predict]
Output: results/verify/zavg_BEF.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "verify"))
from src.splits import make_split  # noqa: E402
import verify_phaseA_best as V  # noqa: E402  (independent verifier helpers: matrix, dev pairs, F, B, AP, bootstrap)

OUT = ROOT / "results/verify/zavg_BEF.json"
PRED_DIR = ROOT / "data_cache/verify"
SEEDS = [0, 1, 2, 3, 4]
N_NULL = 200_000
V0_SEED = 12345
T0 = time.time()
MAN = {"B": ("screen_v2", "rowz[socos_q0.25[spec_k20_t0.1_std]]"),
       "E": ("graph_v1", "emb_lap512[knn_union_lin_q0.25[spec_k20_t0.1_std]]"),
       "F": ("screen_v1", "spec_k20_t0.1_std")}
CACHED_VARIANTS = {"E_d1024": ("graph_v2", "emb_lap1024[knn_union_lin_q0.25[spec_k20_t0.1_std]]"),
                   "E_q0.5": ("graph_v2", "emb_lap512[knn_union_lin_q0.5[spec_k20_t0.1_std]]")}


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def manifest_col(tag, name):
    m = json.loads((ROOT / f"data_cache/layers/{tag}.manifest.json").read_text())
    return m, [c for c in m["columns"] if c["name"] == name][0]


# ------------------------------------------------------------------------------------------- E (own)
def balanced_neighbours(F, bg, Kmax, block=1024):
    """Top-Kmax BG neighbours (self excluded) by balanced z; sorted by decreasing B. Returns gene-index cols, B."""
    G = F.shape[0]
    bgi = np.flatnonzero(bg)
    nb = len(bgi)
    col_of = -np.ones(G, np.int64)
    col_of[bgi] = np.arange(nb)
    F32 = F.astype(np.float32)
    Fbg = np.ascontiguousarray(F32[bgi])
    mean, sd = np.zeros(G), np.zeros(G)
    for s0 in range(0, G, block):
        rows = np.arange(s0, min(G, s0 + block))
        S = (F32[rows] @ Fbg.T).astype(np.float64)
        selfc = col_of[rows]
        hs = selfc >= 0
        sv = np.zeros(len(rows))
        sv[hs] = S[np.flatnonzero(hs), selfc[hs]]
        n = nb - hs.astype(int)
        m = (S.sum(1) - sv) / n
        mean[rows], sd[rows] = m, np.sqrt(np.maximum(((S * S).sum(1) - sv * sv) / n - m * m, 0))
    idx = np.zeros((G, Kmax), np.int64)
    val = np.zeros((G, Kmax))
    for s0 in range(0, G, block):
        rows = np.arange(s0, min(G, s0 + block))
        S = (F32[rows] @ Fbg.T).astype(np.float64)
        Bz = 0.5 * ((S - mean[rows, None]) / sd[rows, None] + (S - mean[bgi][None, :]) / sd[bgi][None, :])
        hs = np.flatnonzero(col_of[rows] >= 0)
        Bz[hs, col_of[rows][hs]] = -np.inf
        part = np.argpartition(-Bz, Kmax - 1, axis=1)[:, :Kmax]
        v = np.take_along_axis(Bz, part, 1)
        o = np.argsort(-v, axis=1, kind="stable")
        idx[rows] = bgi[np.take_along_axis(part, o, 1)]
        val[rows] = np.take_along_axis(v, o, 1)
    return idx, val


def union_lin_graph(idx, val, K, bg):
    G = idx.shape[0]
    w = np.clip(val[:, :K], 0.0, None)
    W = sps.csr_matrix((w.ravel(), idx[:, :K].ravel(), np.arange(0, G * K + 1, K)), shape=(G, G))
    WT = (W.T.tocsr() @ sps.diags(bg.astype(np.float64))).tocsr()      # reverse edges, BG columns only
    A = W.maximum(WT).tocsr()
    nbm = ~bg
    if nbm.any():                                                       # non-BG rows: directed edges only
        A = (sps.diags(bg.astype(np.float64)) @ A + sps.diags(nbm.astype(np.float64)) @ W).tocsr()
    A.eliminate_zeros()
    return A


def lap_embedding(A, bg, d_max, v0_seed=V0_SEED):
    d = np.asarray(A.sum(1)).ravel()
    r = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    N = (sps.diags(r) @ A @ sps.diags(r)).tocsr()
    bgi = np.flatnonzero(bg)
    Nbb = N[bgi][:, bgi]
    asym = abs(Nbb - Nbb.T).max()
    v0 = np.random.default_rng(v0_seed).standard_normal(len(bgi))
    t = time.time()
    lam, U = eigsh(Nbb, k=d_max + 1, which="LA", v0=v0, tol=1e-8)
    o = np.argsort(lam)[::-1]
    lam, U = lam[o], U[:, o]
    E = np.zeros((A.shape[0], d_max + 1))
    E[bgi] = U
    nbg = np.flatnonzero(~bg)
    if len(nbg):
        E[nbg] = (N[nbg][:, bgi] @ U) / np.where(np.abs(lam) > 1e-12, lam, np.inf)
    diag = dict(eigsh_s=round(time.time() - t, 1), n_eig_ge_1m1e6=int((lam > 1 - 1e-6).sum()), lam_top3=lam[:3].tolist(),
                lam_last3=lam[-3:].tolist(), nbb_max_asym=float(asym), n_zero_degree=int((d == 0).sum()),
                n_edges=int(A.nnz))
    return lam, E, diag


def emb_cos(E, d, ia, ib):
    P = E[:, 1:d + 1]
    nrm = np.linalg.norm(P, axis=1)
    P = np.nan_to_num(P / nrm[:, None])
    return np.einsum("ij,ij->i", P[ia], P[ib])


# ------------------------------------------------------------------------------------------- helpers
def null_pairs(bg, seed, n=N_NULL):
    bgi = np.flatnonzero(bg)
    rng = np.random.default_rng(seed)
    a, b = rng.choice(bgi, n), rng.choice(bgi, n)
    while (bad := a == b).any():
        b[bad] = rng.choice(bgi, int(bad.sum()))
    return np.minimum(a, b), np.maximum(a, b)


def mom(s):
    s = np.asarray(s, np.float64)
    ok = np.isfinite(s)
    return dict(null_mean=float(s[ok].mean()), null_sd=float(s[ok].std()), n=int(ok.sum()))


def agree(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return dict(pearson_r=float(np.corrcoef(a, b)[0, 1]), spearman_r=float(pd.Series(a).rank().corr(pd.Series(b).rank())),
                max_abs_diff=float(np.max(np.abs(a - b))), mean_abs_diff=float(np.mean(np.abs(a - b))))


def all_layers(X, ia, ib, na, nb, extra_null=None, variants=()):
    """B, E (q0.25 d512) [+ variants], F on the concatenation of eval pairs and null pairs."""
    F, bg, diagF = V.spectral_profiles(X, V.K_TOP, V.TAU, True)
    n_eval = len(ia)
    nulls = [(na, nb)] + ([extra_null] if extra_null is not None else [])
    PA = np.concatenate([ia] + [p[0] for p in nulls])
    PB = np.concatenate([ib] + [p[1] for p in nulls])
    so = V.second_order_rowz(F, bg, PA, PB)
    so.pop("_diag")
    out = dict(B=so["rowz_socos"], F=so["spec"])
    nbg = int(bg.sum())
    K25, K50 = int(round(0.25 / 100 * nbg)), int(round(0.5 / 100 * nbg))
    idx, val = balanced_neighbours(F, bg, K50)
    log(f"  neighbours K={K25}/{K50}")
    diag = dict(F=diagF, K_q025=K25, K_q05=K50)
    A = union_lin_graph(idx, val, K25, bg)
    lam, Emat, diag["E_q0.25_eig513"] = lap_embedding(A, bg, 512)
    log(f"  eigsh 513: {diag['E_q0.25_eig513']}")
    out["E"] = emb_cos(Emat, 512, PA, PB)
    if "E_d256" in variants:
        out["E_d256"] = emb_cos(Emat, 256, PA, PB)
    del Emat
    if "E_d1024" in variants:
        _, E2, diag["E_q0.25_eig1025"] = lap_embedding(A, bg, 1024)
        log(f"  eigsh 1025: {diag['E_q0.25_eig1025']}")
        out["E_d1024"] = emb_cos(E2, 1024, PA, PB)
        del E2
    if "E_q0.5" in variants:
        A5 = union_lin_graph(idx, val, K50, bg)
        _, E3, diag["E_q0.5_eig513"] = lap_embedding(A5, bg, 512)
        log(f"  eigsh q0.5: {diag['E_q0.5_eig513']}")
        out["E_q0.5"] = emb_cos(E3, 512, PA, PB)
        del E3
    ev = {k: v[:n_eval] for k, v in out.items()}
    nm = []
    off = n_eval
    for p in nulls:
        nm.append({k: mom(v[off:off + len(p[0])]) for k, v in out.items()})
        off += len(p[0])
    return ev, nm, bg, diag


def fold_aps(dev, split, s):
    return V.fold_aps(dev, split, s)


def boot(dev, split, s1, s2, seed):
    y = dev.label.values
    oof = split.test_fold >= 0
    ca, cb = dev.ca.values[oof], dev.cb.values[oof]
    w = V.ht_w(ca, cb, V.K_FOLDS)
    ap1, ap2 = V.weighted_ap(y[oof], s1[oof], w), V.weighted_ap(y[oof], s2[oof], w)
    b = V.boot_ap_diff(y[oof], s1[oof], s2[oof], ca, cb, w, seed=seed)
    return dict(ap1=ap1, ap2=ap2, delta=ap1 - ap2, ci=b["diff_ci"])


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-predict", action="store_true")
    args = ap.parse_args()
    res = dict(script="scripts/verify/verify_zavg_BEF.py", candidate="zavg_BEF = mean(z_B, z_E, z_F)",
               v0_seed_mine=V0_SEED)
    X, cols = V.load_matrix()
    dev, sc, f_dev, sha = V.dev_pairs()
    res.update(assignment_sha256=sha, n_dev_pairs=int(len(dev)), n_dev_pos=int(dev.label.sum()))
    clus = dict(zip(np.r_[dev.a.values, dev.b.values], np.r_[dev.ca.values, dev.cb.values]))
    splits = {s: make_split(dev[["a", "b", "label"]], "cluster", V.K_FOLDS, s, clusters=clus) for s in SEEDS}
    ia, ib, pid, y = dev.ia.values, dev.ib.values, dev.pair_id.values, dev.label.values
    log(f"dev pairs {len(dev)} pos {y.sum()}")

    # ---------------- null pairs (own draw) vs stored
    bg0 = np.isnan(X).mean(0) < V.MAX_NAN
    na, nb = null_pairs(bg0, 0)
    na1, nb1 = null_pairs(bg0, 1)
    pz = np.load(ROOT / "data_cache/layers/graph_v1.pairs.npz")
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(na).tobytes()); h.update(np.ascontiguousarray(nb).tobytes())
    mE, _ = manifest_col("graph_v1", MAN["E"][1])
    res["null_pairs"] = dict(seed=0, n=N_NULL, equal_to_stored_graph_v1=bool(np.array_equal(na, pz["null_a"]) and
                                                                             np.array_equal(nb, pz["null_b"])),
                             sha256_mine=h.hexdigest(), sha256_manifest=mE["null"]["sha256"],
                             sha_match=h.hexdigest() == mE["null"]["sha256"])
    log(f"null pairs: {res['null_pairs']}")

    # ---------------- own layers (real data) on dev + null(seed 0) + null(seed 1)
    variants = ("E_d256", "E_d1024", "E_q0.5")
    ev, nm, bg, diag = all_layers(X, ia, ib, na, nb, extra_null=(na1, nb1), variants=variants)
    res["layer_diag"] = diag
    man = {k: manifest_col(*MAN[k])[1] for k in MAN}
    res["null_moments"] = {k: dict(manifest=dict(null_mean=man[k]["null_mean"], null_sd=man[k]["null_sd"]),
                                   mine_seed0=nm[0][k], mine_seed1=nm[1][k],
                                   d_mean_in_sd=(nm[0][k]["null_mean"] - man[k]["null_mean"]) / man[k]["null_sd"],
                                   d_sd_rel=nm[0][k]["null_sd"] / man[k]["null_sd"] - 1) for k in MAN}
    for k, (tag, name) in CACHED_VARIANTS.items():
        c = manifest_col(tag, name)[1]
        res["null_moments"][k] = dict(manifest=dict(null_mean=c["null_mean"], null_sd=c["null_sd"]), mine_seed0=nm[0][k])
    res["null_moments"]["E_d256"] = dict(manifest=None, mine_seed0=nm[0]["E_d256"])
    log(f"null moments: {json.dumps(res['null_moments'])}")

    # ---------------- (1) score agreement vs caches
    cache = {}
    for k, (tag, name) in {**MAN, **CACHED_VARIANTS}.items():
        M = np.load(ROOT / f"data_cache/layers/{tag}.scores.f32.npy", mmap_mode="r")
        cache[k] = np.asarray(M[:, manifest_col(tag, name)[1]["col"]])[pid].astype(np.float64)
    res["score_agreement_vs_cache"] = {k: agree(ev[k], cache[k]) for k in cache}
    log(f"agreement: {json.dumps(res['score_agreement_vs_cache'])}")

    # ---------------- (2) zavg_BEF and HT-AP per fold
    def z(k, s, mo):
        return (s - mo["null_mean"]) / mo["null_sd"]
    zman = {k: z(k, ev[k], man[k]) for k in MAN}
    zmine = {k: z(k, ev[k], nm[0][k]) for k in ev}
    scores = {"zavg_BEF[mine, manifest moments]": (zman["B"] + zman["E"] + zman["F"]) / 3,
              "zavg_BEF[mine, own moments]": (zmine["B"] + zmine["E"] + zmine["F"]) / 3,
              "zavg_BEF[cache, manifest moments]": sum(z(k, cache[k], man[k]) for k in MAN) / 3,
              "B[mine]": ev["B"], "E[mine]": ev["E"], "F[mine]": ev["F"]}
    for v in variants:
        scores[f"zavg_B{v}F[mine, own moments]"] = (zmine["B"] + zmine[v] + zmine["F"]) / 3
        scores[f"{v}[mine]"] = ev[v]
    res["zavg_agreement"] = dict(mine_vs_cache=agree(scores["zavg_BEF[mine, manifest moments]"],
                                                     scores["zavg_BEF[cache, manifest moments]"]),
                                 own_vs_manifest_moments=agree(scores["zavg_BEF[mine, own moments]"],
                                                               scores["zavg_BEF[mine, manifest moments]"]))
    # producer OOF arrays (audit: same pair score in every fold / seed, no fold-dependent transform)
    oofchk = {}
    for s in SEEDS:
        f = ROOT / f"data_cache/phaseB_v2_oof/zavg_BEF_cluster_seed{s}.npy"
        if f.exists():
            o = np.load(f)
            m = np.isfinite(o)
            oofchk[s] = dict(n_finite=int(m.sum()), n_test=int((splits[s].test_fold >= 0).sum()),
                             max_abs_diff_vs_mine=float(np.max(np.abs(o[m] - scores["zavg_BEF[mine, manifest moments]"][m]))),
                             max_abs_diff_vs_cache_formula=float(np.max(np.abs(o[m] - scores["zavg_BEF[cache, manifest moments]"][m]))))
    res["producer_oof_vs_fixed_scores"] = oofchk
    log(f"OOF check: {oofchk}")

    rep = {}
    for name in ("zavg_BEF", "ref_B", "ref_E", "ref_F"):
        rep[name] = {}
        for s in SEEDS:
            d = json.loads((ROOT / f"results/phaseB_v2/runs/{name}_cluster_seed{s}.json").read_text())
            rep[name][s] = dict(folds=[f["ap_ht"] for f in d["folds"]], fingerprint=d["split"]["fingerprint"])
    cmp_ = json.loads((ROOT / "results/phaseB_v2/compare.json").read_text())["seeds"]
    prev = {s: float(np.mean([(lambda w, yy: (w * yy).sum() / w.sum())(
        V.ht_w(dev.ca.values[splits[s].test_fold == k], dev.cb.values[splits[s].test_fold == k], V.K_FOLDS),
        y[splits[s].test_fold == k]) for k in range(V.K_FOLDS)])) for s in SEEDS}
    ht = {}
    for name, s_ in scores.items():
        per = {s: fold_aps(dev, splits[s], s_) for s in SEEDS}
        ht[name] = dict(per_seed={s: dict(mean=float(np.mean(v)), folds=[float(x) for x in v]) for s, v in per.items()},
                        mean_5seeds=float(np.mean([np.mean(v) for v in per.values()])))
        log(f"{name}: " + " ".join(f"s{s} {np.mean(v):.4f}" for s, v in per.items()) + f" | mean {ht[name]['mean_5seeds']:.4f}")
    res["ht_ap"] = ht
    res["reported_ht_ap"] = {n: {s: dict(mean=float(np.mean(v["folds"])), folds=v["folds"]) for s, v in d.items()}
                             for n, d in rep.items()}
    res["reported_ht_ap_5seed_mean"] = {n: float(np.mean([np.mean(v["folds"]) for v in d.values()])) for n, d in rep.items()}
    res["max_abs_fold_diff_vs_reported"] = {
        n: float(max(np.max(np.abs(np.array(ht[m]["per_seed"][s]["folds"]) - np.array(rep[n][s]["folds"]))) for s in SEEDS))
        for n, m in (("zavg_BEF", "zavg_BEF[mine, manifest moments]"), ("ref_B", "B[mine]"), ("ref_E", "E[mine]"),
                     ("ref_F", "F[mine]"))}
    res["ht_prevalence_mean_per_seed"] = prev

    # pooled-OOF delta vs B, paired cluster bootstrap (bootstrap seed = split seed, as reported)
    pool = {}
    zmain = scores["zavg_BEF[mine, manifest moments]"]
    for s in SEEDS:
        b = boot(dev, splits[s], zmain, ev["B"], s)
        r = cmp_[str(s)]["all"]["diff"]["zavg_BEF-ref_B"]
        pool[s] = dict(mine=b, reported=dict(delta=r["diff"], ci=[r["ci_low"], r["ci_high"]]))
        log(f"pooled OOF seed {s}: mine {b['delta']:+.4f} {np.round(b['ci'], 4).tolist()} reported {r['diff']:+.4f} "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    res["pooled_oof_delta_vs_B"] = pool

    # ---------------- (6) robustness: E hyper-parameters
    rob = {}
    for v in ("E",) + variants:
        key = "zavg_BEF[mine, own moments]" if v == "E" else f"zavg_B{v}F[mine, own moments]"
        bs = {s: boot(dev, splits[s], scores[key], ev["B"], s) for s in SEEDS}
        rob[v] = dict(ht_ap_5seed_mean=ht[key]["mean_5seeds"], B_5seed_mean=ht["B[mine]"]["mean_5seeds"],
                      E_alone_5seed_mean=ht[f"{v}[mine]" if v != "E" else "E[mine]"]["mean_5seeds"],
                      folds_won_vs_B=int(sum(np.sum(np.array(ht[key]["per_seed"][s]["folds"]) >
                                                    np.array(ht["B[mine]"]["per_seed"][s]["folds"])) for s in SEEDS)),
                      pooled_oof_delta={s: dict(delta=b["delta"], ci=b["ci"]) for s, b in bs.items()},
                      n_seeds_ci_above_0=int(sum(b["ci"][0] > 0 for b in bs.values())))
        log(f"robust {v}: {rob[v]['ht_ap_5seed_mean']:.4f} vs B {rob[v]['B_5seed_mean']:.4f}; CI>0 in "
            f"{rob[v]['n_seeds_ci_above_0']}/5")
    res["robustness"] = rob

    # ---------------- (3) producer predictor on dev pair IDs (black box)
    if not args.skip_predict:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        fin, fout = PRED_DIR / "dev_pairs.tsv", PRED_DIR / "dev_pairs_zavg_BEF.tsv"
        dev[["a", "b"]].to_csv(fin, sep="\t", header=False, index=False)
        t = time.time()
        p = subprocess.run([sys.executable, str(ROOT / "scripts/predict_pairs.py"), "--pairs", str(fin), "--method",
                            "zavg:B,E,F", "--out", str(fout), "--components"], cwd=ROOT, capture_output=True, text=True)
        log(f"predict_pairs rc {p.returncode} ({time.time() - t:.0f}s): {p.stdout[-400:]} {p.stderr[-800:]}")
        if p.returncode == 0:
            o = pd.read_csv(fout, sep="\t")
            assert (o.uniprot_a.values == dev.a.values).all() and (o.uniprot_b.values == dev.b.values).all()
            mp = o.mapped.values.astype(bool)
            r = dict(n=int(len(o)), n_mapped=int(mp.sum()), runtime_s=round(time.time() - t, 1),
                     score=agree(o.score.values[mp], zmain[mp]))
            for k in ("B", "E", "F"):
                if f"z_{k}" in o:
                    r[f"z_{k}"] = agree(o[f"z_{k}"].values[mp], zman[k][mp])
            r["ht_ap_seed0_predictor"] = float(np.mean(fold_aps(dev, splits[0], o.score.values)))
            res["predict_pairs_vs_mine"] = r
            log(f"predictor: {json.dumps(r)}")
        else:
            res["predict_pairs_vs_mine"] = dict(error=p.stderr[-2000:])

    # ---------------- (5) negative control: shuffle gene columns, rebuild B, E, F + own null moments
    rng = np.random.default_rng(0)
    Xp = X[:, rng.permutation(X.shape[1])]
    bgp = np.isnan(Xp).mean(0) < V.MAX_NAN
    npa, npb = null_pairs(bgp, 0)
    evp, nmp, _, _ = all_layers(Xp, ia, ib, npa, npb)
    del Xp
    zp = sum((evp[k] - nmp[0][k]["null_mean"]) / nmp[0][k]["null_sd"] for k in ("B", "E", "F")) / 3
    nc = {}
    for s in (0, 1):
        fa = fold_aps(dev, splits[s], zp)
        nc[s] = dict(ht_ap=float(np.mean(fa)), folds=[float(x) for x in fa], ht_prevalence=prev[s],
                     ratio=float(np.mean(fa) / prev[s]))
    res["negative_control_gene_shuffle"] = nc
    log(f"negative control: {nc}")

    # ---------------- (4) static leakage audit (imports / label reads in producer code)
    aud = {}
    for f in ("src/layers_graph.py", "scripts/15_build_graph_layers.py", "scripts/21_phaseB_v2.py", "src/predict.py"):
        txt = (ROOT / f).read_text()
        aud[f] = dict(imports=sorted({ln.strip() for ln in txt.splitlines() if ln.startswith(("import ", "from "))}),
                      mentions_label=int(sum("label" in ln for ln in txt.splitlines())),
                      mentions_ift=int(sum("ift" in ln.lower() for ln in txt.splitlines())))
    res["static_audit"] = aud
    res["runtime_s"] = round(time.time() - T0, 1)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
