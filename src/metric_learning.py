"""Phase C (DESIGN v3 §4, time-boxed): supervised metric learning on whitened DepMap profiles.

The learned metric maps each gene's profile to an embedding with a function that depends on that profile only, so
it applies to unseen proteins; the pair score is the cosine of the two embeddings (first order), optionally followed
by the label-free genome-wide machinery of Phase A (row-z, sparse second order, row-z of second order) computed on
the learned embedding of ALL genes.

Base profiles (label-free, genome-wide): unit-norm filtered profiles of the Phase-A best base
``spec_k20_t0.1_std`` (gene-standardised Σ, top-20 PCs removed, ridge 0.1 whitening), expressed in the eigenbasis
of Σ (m = 1,340 coordinates). Identity metric => cosine = that layer exactly.

Capacity ladder (``MetricScorer(level=...)``):
  identity  no parameters (sanity: reproduces the Phase-A layers through the same code path).
  L1        diagonal weights d (init 1, penalty λ·||d-1||²). basis='line': d are cell-line weights applied to the
            whitened profiles mapped back to cell-line space (ZCA form x_line = M z, M = Q U_sel, orthonormal
            columns); score = weighted Pearson correlation with weights v = d² (weighted centring, then cosine).
            basis='eig': d multiplies the eigen-coordinates, i.e. a learned spectral filter.
  L2        low-rank residual projection e = z + (z Aᵀ) Bᵀ, A (d x m) initialised to the top-d principal axes of the
            base profiles over background genes (label-free), B (m x d) initialised to 0 (=> identity at start);
            penalty λ·(||B||² + 0.1·||A - A0||²).
Losses: 'bce' = weighted logistic loss on logit = s·cos + b (s, b learned, unpenalised) over ALL labelled training
pairs (full batch, the training fold has ~650 genes): positives weighted 1/sqrt(n_pos of their cluster) (as
Phase B), negatives weight 1, hard negatives (both genes in the same §3.1 cluster) x ``hard_neg``; the negative
total is rescaled to the positive total. 'infonce' = multi-positive InfoNCE per anchor gene over its labelled
training partners (positives vs its labelled negatives, hard negatives included), logit = s·cos.
Training: full-batch Adam; inner cluster-grouped validation split of the training pairs (src.splits.make_split,
``n_inner`` folds, inner fold 0 = validation; inner-train = pairs with both genes outside it); HT-weighted AP on
the validation pairs every ``eval_every`` steps (step 0 = identity included), early stopping with ``patience``;
λ chosen from ``lams`` by validation AP (ties -> larger λ); final model = refit on all training pairs with the
chosen λ for the chosen number of steps (0 steps => identity). Randomness: torch seed from SeedSequence(seed,
fit counter); seed must come from src.config.SEEDS. CPU float32 (deterministic; MPS gave no speed-up at this size).

Outputs (``output``): 'cos' first-order cosine; 'rowz' genome-wide row-z of it; 'so' sparse second-order
weighted cosine with top-q% balanced neighbours (q = 0.25); 'rowzso' row-z of 'so' (= the Phase-A best pipeline
on the learned embedding). Genome-wide parts reuse src.layers / src.layers_ext through ``EmbeddingEngine``, a
LayerEngine whose spectral profiles are the given embedding (background genes = the same NaN<10% BG set).

Labels: ``fit`` reads ``universe.pairs.label`` at train_idx only (NaN/non-binary raises). Fitted parameters are
memoised per (configuration, train_idx, training labels) so several outputs of one configuration share a fit.
"""
from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict

import numpy as np

from src import layers, layers_ext, metrics
from src.config import SEEDS
from src.scorers import Scorer

BEST_BASE = layers.Spectral(20, 0.1, standardize=True)
EMB = layers.Spectral(0, layers.INF, matrix="learned_embedding")      # dummy key for EmbeddingEngine
OUTPUTS = ("cos", "rowz", "so", "rowzso")
DEFAULT_LR = {"identity": 0.0, "L1": 1e-3, "L2": 2e-4}
DEFAULT_LAMS = {"L1": (1e-1, 1e-2, 1e-3), "L2": (1e-1, 1e-2, 1e-3)}


# ------------------------------------------------------------------------------------- genome-wide engine
class EmbeddingEngine(layers.LayerEngine):
    """LayerEngine over a fixed embedding E (genes x dim): every Spectral config resolves to the unit-normalised E.
    ``bg`` = background mask (genes used for row moments / neighbour sets)."""

    def __init__(self, E: np.ndarray, bg: np.ndarray, block: int = layers.BLOCK):
        E = np.asarray(E, dtype=np.float64)
        self.G, self.n = E.shape
        self.bg = np.asarray(bg, bool)
        self.bg_idx = np.flatnonzero(self.bg)
        self.n_bg = len(self.bg_idx)
        self.bg_pos = np.full(self.G, -1)
        self.bg_pos[self.bg_idx] = np.arange(self.n_bg)
        self.block, self.cache_size = block, 1
        self._eig, self._stats, self._nbr = {}, {}, {}
        self._spec = OrderedDict()
        with np.errstate(invalid="ignore", divide="ignore"):
            Z = E / np.linalg.norm(E, axis=1, keepdims=True)
        Z[~np.isfinite(Z)] = 0.0
        self._Z = (np.ascontiguousarray(Z), np.ascontiguousarray(Z, dtype=np.float32))
        self.info = {}

    def spectral(self, cfg):
        return self._Z

    def eig(self, *a, **k):
        raise NotImplementedError("EmbeddingEngine has no covariance")


def output_config(output: str, q: float = 0.25):
    so = layers.SecondOrder(EMB, q=q, measure="cos", balanced=True)
    return {"cos": EMB, "rowz": layers.RowZ(EMB), "so": so, "rowzso": layers_ext.RowZSO(so)}[output]


def engine_scores(eng: EmbeddingEngine, output: str, ia, ib, q: float = 0.25) -> np.ndarray:
    return layers_ext.compute(output_config(output, q), ia, ib, eng)


# ------------------------------------------------------------------------------------- base profiles
_BASE: dict = {}


def real_base(cfg: layers.Spectral = BEST_BASE) -> dict:
    """Genome-wide label-free base: eigen-coordinates Z (G x m, unit norm), line map M (n x m), BG mask, PCs."""
    if cfg not in _BASE:
        eng = layers.get_engine(cfg.matrix)
        Z = eng.spectral(cfg)[0].copy()
        e = eng.eig(cfg.center, cfg.standardize)
        sel = eng.filter_weights(cfg) > 0
        M = e["Q"] @ e["U"][:, sel]
        _BASE[cfg] = dict(Z=Z, M=M, bg=eng.bg.copy(), name=cfg.name)
    return _BASE[cfg]


def principal_axes(Z: np.ndarray, bg: np.ndarray, d: int) -> np.ndarray:
    """Top-d principal axes (rows, orthonormal) of the background profiles (uncentred second moment)."""
    Zb = Z[bg]
    C = Zb.T @ Zb / len(Zb)
    lam, U = np.linalg.eigh(C)
    U = U[:, np.argsort(lam)[::-1][:d]]
    sgn = np.sign(U[np.abs(U).argmax(0), np.arange(U.shape[1])])     # deterministic sign
    return (U * sgn).T.copy()


# ------------------------------------------------------------------------------------- torch model
def _torch():
    import torch
    torch.set_num_threads(4)
    return torch


class _Model:
    """Parameter container + forward map (torch). level: identity | L1 | L2."""

    def __init__(self, level, dim, d=None, A0=None, line=False):
        torch = _torch()
        self.level, self.line = level, line
        self.log_s = torch.tensor(math.log(10.0), requires_grad=True)
        self.b = torch.tensor(-5.0, requires_grad=True)
        self.params = [self.log_s, self.b]
        if level == "L1":
            self.dvec = torch.ones(dim, requires_grad=True)
            self.params.append(self.dvec)
        elif level == "L2":
            self.A0 = torch.tensor(A0, dtype=torch.float32)
            self.A = self.A0.clone().requires_grad_(True)
            self.B = torch.zeros(dim, d, requires_grad=True)
            self.params += [self.A, self.B]

    def embed(self, Z):
        torch = _torch()
        if self.level == "identity":
            E = Z
        elif self.level == "L1":
            if self.line:
                v = self.dvec ** 2
                mu = (Z * v).sum(1, keepdim=True) / v.sum()
                E = (Z - mu) * self.dvec.abs()
            else:
                E = Z * self.dvec
        else:
            E = Z + (Z @ self.A.T) @ self.B.T
        return E / torch.linalg.norm(E, dim=1, keepdim=True).clamp_min(1e-12)

    def penalty(self):
        if self.level == "L1":
            return ((self.dvec - 1) ** 2).sum()
        if self.level == "L2":
            return (self.B ** 2).sum() + 0.1 * ((self.A - self.A0) ** 2).sum()
        return 0.0

    def state(self) -> dict:
        out = dict(log_s=float(self.log_s.detach()), b=float(self.b.detach()))
        if self.level == "L1":
            out["d"] = self.dvec.detach().numpy().copy()
        elif self.level == "L2":
            out["A"], out["B"] = self.A.detach().numpy().copy(), self.B.detach().numpy().copy()
        return out


def embed_numpy(Z: np.ndarray, level: str, state: dict | None, line: bool = False, chunk: int = 4096) -> np.ndarray:
    """Apply a fitted map to profiles (rows) with numpy (unnormalised; callers normalise). Row-wise only."""
    if level == "identity" or state is None:
        return np.asarray(Z, dtype=np.float64)
    out = np.empty((len(Z), Z.shape[1]))
    for s in range(0, len(Z), chunk):
        z = np.asarray(Z[s:s + chunk], dtype=np.float64)
        if level == "L1":
            dv = state["d"].astype(np.float64)
            if line:
                v = dv ** 2
                mu = (z * v).sum(1, keepdims=True) / v.sum()
                out[s:s + chunk] = (z - mu) * np.abs(dv)
            else:
                out[s:s + chunk] = z * dv
        else:
            A, B = state["A"].astype(np.float64), state["B"].astype(np.float64)
            out[s:s + chunk] = z + (z @ A.T) @ B.T
    return out


def train_metric(Z, la, lb, y, w, hard, level, *, loss="bce", lam=1e-2, d=64, A0=None, line=False, lr=1e-2,
                 max_steps=600, eval_every=10, patience=60, val=None, fixed_steps=None, seed=0, hard_neg=3.0,
                 calib_steps=100):
    """Full-batch training on local gene profiles Z (n_g x dim) and labelled pairs (la, lb, y).

    val: None or dict(la, lb, y, w) of validation pairs (HT design weights) -> early stopping, returns best step.
    fixed_steps: train exactly this many steps (refit). Returns (state, log)."""
    torch = _torch()
    torch.manual_seed(seed)
    Zt = torch.tensor(Z, dtype=torch.float32)
    m = _Model(level, Z.shape[1], d=d, A0=A0, line=line)
    y = np.asarray(y).astype(bool)
    log = dict(level=level, loss=loss, lam=lam, n_pairs=int(len(y)), n_pos=int(y.sum()))
    if level == "identity" or fixed_steps == 0:
        log.update(steps=0)
        return m.state(), log
    la_t, lb_t = torch.tensor(la), torch.tensor(lb)
    yt = torch.tensor(y, dtype=torch.float32)
    if loss == "bce":
        ww = np.asarray(w, dtype=np.float64).copy()
        ww[~y & hard] *= hard_neg
        ww[~y] *= ww[y].sum() / max(ww[~y].sum(), 1e-12)
        wt = torch.tensor(ww / ww.sum(), dtype=torch.float32)
    else:
        n = len(Z)
        Lm = np.zeros((n, n), bool)
        Pm = np.zeros((n, n), bool)
        Lm[la, lb] = Lm[lb, la] = True
        Pm[la[y], lb[y]] = Pm[lb[y], la[y]] = True
        rows = torch.tensor(np.flatnonzero(Pm.any(1) & (Lm.sum(1) > Pm.sum(1))))
        Lt, Pt = torch.tensor(Lm), torch.tensor(Pm)
    calib = [m.log_s, m.b]
    metric_params = [p for p in m.params if all(p is not q for q in calib)]

    def objective():
        E = m.embed(Zt)
        if loss == "bce":
            c = (E @ E.T)[la_t, lb_t]            # n_g x n_g Gram, then gather (cheaper than per-pair rows)
            logit = m.log_s.exp() * c + m.b
            return (torch.nn.functional.binary_cross_entropy_with_logits(logit, yt, reduction="none") * wt).sum()
        C = m.log_s.exp() * (E @ E.T)
        neg_inf = torch.finfo(C.dtype).min
        lse_all = torch.logsumexp(C.masked_fill(~Lt, neg_inf)[rows], 1)
        lse_pos = torch.logsumexp(C.masked_fill(~Pt, neg_inf)[rows], 1)
        return (lse_all - lse_pos).mean()

    # calibration warm-up: scale/bias only, metric frozen at its initialisation (does not change the ranking)
    opt_c = torch.optim.Adam(calib, lr=0.05)
    for _ in range(calib_steps):
        opt_c.zero_grad()
        objective().backward()
        opt_c.step()
    for p in metric_params:
        p.grad = None
    opt = torch.optim.Adam([dict(params=calib, lr=0.01), dict(params=metric_params, lr=lr)])

    def val_ap():
        with torch.no_grad():
            E = m.embed(Zt)
            c = (E @ E.T).numpy()[val["la"], val["lb"]]
        return metrics.weighted_ap(val["y"], c, val["w"])

    best = dict(step=0, ap=val_ap() if val is not None else float("nan"), state=m.state())
    log["val_ap_identity"] = best["ap"]
    hist = []
    n_steps = fixed_steps if fixed_steps is not None else max_steps
    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        opt.zero_grad()
        L = objective()
        obj = L + lam * m.penalty()
        obj.backward()
        opt.step()
        if val is not None and step % eval_every == 0:
            ap = val_ap()
            hist.append((step, float(L.detach()), ap))
            if ap > best["ap"] + 1e-6:
                best = dict(step=step, ap=ap, state=m.state())
            elif step - best["step"] >= patience:
                break
    log.update(train_s=round(time.perf_counter() - t0, 2), steps_run=step, history=hist[-200:])
    if val is not None:
        log.update(best_step=best["step"], best_val_ap=best["ap"])
        return best["state"], log
    log.update(steps=n_steps, final_loss=float(L.detach()))
    return m.state(), log


# ------------------------------------------------------------------------------------- scorer
_FIT_MEMO: "OrderedDict[tuple, tuple]" = OrderedDict()
_FIT_MEMO_SIZE = 64


def _hash(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


class MetricScorer(Scorer):
    """Supervised learned metric (see module docstring). ``base``: dict(Z, M, bg) from ``real_base`` (default) or a
    toy dict(Z, bg[, M]) with Z indexed by the universe's gene column indices ia/ib."""
    label_free = False

    def __init__(self, name, level="L2", basis="eig", d=64, loss="bce", output="cos", seed=SEEDS[0], lams=None,
                 hard_neg=3.0, q=0.25, base=None, lr=None, max_steps=600, eval_every=10, patience=60, n_inner=4):
        super().__init__()
        if level not in ("identity", "L1", "L2") or basis not in ("eig", "line") or output not in OUTPUTS:
            raise ValueError((level, basis, output))
        if loss not in ("bce", "infonce"):
            raise ValueError(loss)
        if seed not in SEEDS:
            raise ValueError("seed must come from src.config.SEEDS")
        self.name, self.level, self.basis, self.d, self.loss, self.output = name, level, basis, d, loss, output
        self.seed, self.hard_neg, self.q, self._base = seed, hard_neg, q, base
        self.lams = tuple(lams) if lams is not None else DEFAULT_LAMS.get(level, (0.0,))
        self.lr = lr if lr is not None else DEFAULT_LR[level]
        self.max_steps, self.eval_every, self.patience, self.n_inner = max_steps, eval_every, patience, n_inner
        self.params = dict(level=level, basis=basis if level == "L1" else "eig", d=d if level == "L2" else None,
                           loss=loss, output=output, seed=seed, lams=list(self.lams), hard_neg=hard_neg, q=q,
                           lr=self.lr, max_steps=max_steps, eval_every=eval_every, patience=patience,
                           n_inner=n_inner, base=(base or {}).get("name", BEST_BASE.name) if base else BEST_BASE.name)
        self.fit_log, self._fit_count, self.state = [], 0, None

    # ------------------------------------------------------------------ data
    def _profiles(self):
        b = self._base if self._base is not None else real_base()
        if "hash" not in b:
            b["hash"] = _hash(np.asarray(b["Z"], dtype=np.float64), np.asarray(b["bg"], bool))
        if self.level == "L1" and self.basis == "line":
            if "Zline" not in b:
                b["Zline"] = b["Z"] @ b["M"].T
            return b["Zline"], b["bg"]
        return b["Z"], b["bg"]

    def _profiles_hash(self):
        b = self._base if self._base is not None else real_base()
        self._profiles()
        return b["hash"]

    def bind(self, universe):
        super().bind(universe)
        self._profiles()
        return self

    def _labels(self, idx):
        y = self.u.pairs["label"].to_numpy(dtype=np.float64)[np.asarray(idx)]
        if not np.isfinite(y).all() or not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("training labels must be finite 0/1")
        return y.astype(bool)

    def _key(self, train_idx, y):
        p = self.params
        return (p["level"], p["basis"], p["d"], p["loss"], p["seed"], tuple(p["lams"]), p["hard_neg"], p["lr"],
                p["max_steps"], p["eval_every"], p["patience"], p["n_inner"], p["base"], self._profiles_hash(), self.u.name,
                _hash(np.asarray(train_idx, np.int64), y.astype(np.int8)), self._fit_count_key)

    # ------------------------------------------------------------------ fit
    def fit(self, train_idx):
        from src import splits
        from src.supervised import cluster_balanced_weights
        train_idx = np.asarray(train_idx)
        y = self._labels(train_idx)
        self._fit_count_key = self._fit_count
        key = self._key(train_idx, y)
        self._fit_count += 1
        if key in _FIT_MEMO:
            self.state, log = _FIT_MEMO[key]
            self.fit_log.append(dict(log, memo=True))
            self._eng = None
            return self
        t0 = time.perf_counter()
        Z, _ = self._profiles()
        ia, ib = self._ia[train_idx], self._ib[train_idx]
        genes, inv = np.unique(np.r_[ia, ib], return_inverse=True)
        la, lb = inv[:len(ia)], inv[len(ia):]
        Zl = np.asarray(Z[genes], dtype=np.float64)
        ca, cb = self.u.col("cluster_a")[train_idx], self.u.col("cluster_b")[train_idx]
        w = cluster_balanced_weights(y, ca, cb)
        hard = ca == cb
        A0 = principal_axes(*self._profiles(), self.d) if self.level == "L2" else None
        line = self.level == "L1" and self.basis == "line"
        ss = np.random.SeedSequence([self.seed, self._fit_count_key, 11])
        tseed = int(ss.generate_state(1)[0] % (2**31))
        # inner cluster-grouped validation split
        p = self.u.pairs.iloc[train_idx][["a", "b"]].reset_index(drop=True)
        p["label"] = y.astype(np.int8)
        cmap = dict(zip(np.r_[p.a.values, p.b.values], np.r_[ca, cb]))
        sp = splits.make_split(p, "cluster", self.n_inner, int(ss.generate_state(2)[1] % (2**31)), clusters=cmap)
        itr, val = sp.train_mask(0), sp.test_mask(0)
        vw = metrics.ht_weights(ca[val], cb[val], 1.0 / self.n_inner)
        vdata = dict(Z=Zl, la=la[val], lb=lb[val], y=y[val], w=vw)
        kw = dict(loss=self.loss, d=self.d, A0=A0, line=line, lr=self.lr, max_steps=self.max_steps,
                  eval_every=self.eval_every, patience=self.patience, seed=tseed, hard_neg=self.hard_neg)
        log = dict(n_train=int(len(y)), n_pos=int(y.sum()), n_genes=int(len(genes)), n_val=int(val.sum()),
                   n_val_pos=int(y[val].sum()), lam_search=[])
        best = None
        for lam in (self.lams if self.level != "identity" else (0.0,)):
            st, lg = train_metric(Zl, la[itr], lb[itr], y[itr], w[itr], hard[itr], self.level, lam=lam, val=vdata,
                                  **kw)
            lg["history"] = lg.get("history", [])[::5]
            log["lam_search"].append(lg)
            ap = lg.get("best_val_ap", lg.get("val_ap_identity", float("nan")))
            if self.level == "identity":
                best = (lam, 0, ap)
                break
            if best is None or ap > best[2] + 1e-6:
                best = (lam, lg["best_step"], ap)
        lam, steps, ap = best
        log.update(chosen_lam=lam, chosen_steps=int(steps), chosen_val_ap=ap,
                   val_ap_identity=log["lam_search"][0].get("val_ap_identity"))
        st, lg = train_metric(Zl, la, lb, y, w, hard, self.level, lam=lam, fixed_steps=int(steps), **kw)
        log["refit"] = lg
        log["fit_s"] = round(time.perf_counter() - t0, 2)
        self.state = st if steps > 0 else None
        _FIT_MEMO[key] = (self.state, log)
        while len(_FIT_MEMO) > _FIT_MEMO_SIZE:
            _FIT_MEMO.popitem(last=False)
        self.fit_log.append(log)
        self._eng = None
        return self

    # ------------------------------------------------------------------ score
    def embedding(self) -> np.ndarray:
        Z, _ = self._profiles()
        line = self.level == "L1" and self.basis == "line"
        return embed_numpy(Z, self.level if self.state is not None else "identity", self.state, line=line)

    def engine(self) -> EmbeddingEngine:
        if getattr(self, "_eng", None) is None:
            _, bg = self._profiles()
            layers_ext.release()                 # so-row-stats memo is keyed by id(engine)
            self._eng = EmbeddingEngine(self.embedding(), bg)
        return self._eng

    def score(self, idx):
        if self.state is None and self.level != "identity" and not self.fit_log:
            raise RuntimeError("fit first")
        idx = np.asarray(idx)
        return engine_scores(self.engine(), self.output, self._ia[idx], self._ib[idx], self.q)

    def fit_summary(self) -> list[dict]:
        keys = ("n_train", "n_pos", "n_genes", "n_val", "n_val_pos", "chosen_lam", "chosen_steps", "chosen_val_ap",
                "val_ap_identity", "fit_s", "memo")
        return [{k: lg.get(k) for k in keys} for lg in self.fit_log]
