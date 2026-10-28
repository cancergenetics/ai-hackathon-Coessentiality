"""Process safeguards: IFT loader import restriction (AST), dirty-tree refusal, start/end trial records, evaluate
end-to-end on toy, held-fold label-shuffle control, look #0 present."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src import evaluate, scorers
from src.config import ROOT, SEEDS
from src.scorers import Scorer
from src.toy import make_toy, toy_dev

ALLOWED = {ROOT / "scripts" / "final_eval.py"}


def _ift_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if a.name == "src.ift" or a.name.startswith("src.ift.")]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "src.ift" or mod.startswith("src.ift."):
                hits.append(mod)
            if mod == "src" and any(a.name == "ift" for a in node.names):
                hits.append("src.ift")
            if mod.endswith("ift") and node.level > 0:
                hits.append("." + mod)
        elif isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name == "load_ift_universe":
                hits.append("load_ift_universe")
    return hits


def test_only_final_eval_imports_ift():
    files = [p for d in ("src", "scripts", "tests") for p in (ROOT / d).rglob("*.py")]
    assert any(p.name == "final_eval.py" for p in files)
    offenders = {str(p.relative_to(ROOT)): h for p in files if p not in ALLOWED and p.name != "ift.py"
                 for h in [_ift_imports(p)] if h}
    assert not offenders, offenders
    assert _ift_imports(ROOT / "scripts" / "final_eval.py")              # the scanner does see real imports


def test_look_zero_seeded():
    first = json.loads((ROOT / "results" / "ift_looks.jsonl").read_text().splitlines()[0])
    assert first["look"] == 0 and "D1" in first["what"]


@pytest.fixture(scope="module")
def toy():
    u, X, _ = make_toy(seed=SEEDS[0])
    return toy_dev(u), X


def test_dirty_refusal_and_trial_records(toy, tmp_path, monkeypatch):
    d, X = toy
    trials = tmp_path / "trials.jsonl"
    monkeypatch.setattr(evaluate, "git_state", lambda: dict(commit="x", dirty=True, dirty_src=True))
    sc = scorers.make_baseline("pearson_signed", scorers._Matrix(X))
    with pytest.raises(RuntimeError, match="uncommitted"):
        evaluate.run_dev_cv(sc, universe=d, out_dir=tmp_path, trials_path=trials)
    assert not trials.exists()
    res = evaluate.run_dev_cv(sc, universe=d, out_dir=tmp_path, trials_path=trials, allow_dirty=True)
    recs = [json.loads(l) for l in trials.read_text().splitlines()]
    assert [r["event"] for r in recs] == ["start", "end"] and recs[1]["status"] == "ok"
    assert recs[0]["trial_id"] == recs[1]["trial_id"]
    out = json.loads(Path(res["path"]).read_text())
    assert out["run_record"]["allow_dirty"] is True and out["run_record"]["git_dirty_src"] is True
    assert {"git_commit", "cache_sha256", "seed", "counts", "runtime_s"} <= set(out["run_record"])
    assert len(out["folds"]) == 5 and "pooled" in out


class _Boom(Scorer):
    name = "boom"

    def score(self, idx):
        raise RuntimeError("boom")


def test_aborted_run_leaves_start_and_error_end(toy, tmp_path, monkeypatch):
    d, _ = toy
    trials = tmp_path / "t.jsonl"
    monkeypatch.setattr(evaluate, "git_state", lambda: dict(commit="x", dirty=False, dirty_src=False))
    with pytest.raises(RuntimeError, match="boom"):
        evaluate.run_dev_cv(_Boom(), universe=d, out_dir=tmp_path, trials_path=trials)
    recs = [json.loads(l) for l in trials.read_text().splitlines()]
    assert [r["event"] for r in recs] == ["start", "end"] and recs[1]["status"] == "error"


def test_seed_must_come_from_config(toy):
    d, X = toy
    with pytest.raises(ValueError):
        evaluate.run_dev_cv(scorers.make_baseline("pearson_signed", scorers._Matrix(X)), seed=12345, universe=d,
                            write=False)


def test_label_free_fold_scores_match_direct_scoring(toy):
    d, X = toy
    sc = scorers.make_baseline("gls_signed", scorers._Matrix(X))
    res = evaluate.run_dev_cv(sc, universe=d, write=False)
    sp = evaluate.make_dev_split(d, "cluster", SEEDS[0])
    te = np.flatnonzero(sp.test_mask(2))
    s = scorers.make_baseline("gls_signed", scorers._Matrix(X)).bind(d).score(te)
    from src import metrics
    w = evaluate.design_weights(d, te, "cluster")
    assert res["folds"][2]["ap_ht"] == pytest.approx(metrics.weighted_ap(d.y[te], s, w))


class Propensity(Scorer):
    """Toy supervised scorer: number of common positive neighbours in the training pairs (triadic closure; works
    under random pair splits, uninformative when training labels are permuted)."""
    name = "common_neighbours"
    label_free = False

    def fit(self, train_idx):
        p = self.u.pairs.iloc[train_idx]
        pos = p[p.label == 1]
        nb = {}
        for x, y_ in zip(pos.a, pos.b):
            nb.setdefault(x, set()).add(y_)
            nb.setdefault(y_, set()).add(x)
        self.nb = nb
        return self

    def score(self, idx):
        p = self.u.pairs.iloc[idx]
        e = set()
        cn = np.array([len(self.nb.get(x, e) & self.nb.get(y_, e)) for x, y_ in zip(p.a, p.b)], dtype=float)
        pid = self.u.col("pair_id")[idx]
        tie = np.random.default_rng(SEEDS[0]).random(int(self.u.col("pair_id").max()) + 1)[pid]   # per-pair, fixed
        return cn + 1e-3 * tie


def test_label_shuffle_control(toy):
    d, _ = toy
    real = evaluate.run_dev_cv(Propensity(), universe=d, scheme="random", write=False)
    assert real["summary"]["ap_ht"]["mean"] > 3 * d.y.mean()
    for mode in ("train", "held"):
        c = evaluate.label_shuffle_control(Propensity, d, seed=SEEDS[0], mode=mode, scheme="random")
        assert c["mean_ap_ht"] < 1.6 * c["mean_prevalence"], (mode, c)
