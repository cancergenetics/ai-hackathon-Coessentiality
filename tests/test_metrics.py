"""Metric correctness on hand-computed examples; bootstrap coverage; negative control."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.config import SEEDS
from src.metrics import (PresortedScores, auprc, bootstrap_auprc, bootstrap_auprc_diff, evaluate,
                         precision_at_recall)


def test_ap_hand_computed():
    # ranks: + - + -  -> P@1 = 1 (R .5), P@3 = 2/3 (R 1) -> AP = .5*1 + .5*2/3
    y, s = [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]
    assert auprc(y, s) == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert PresortedScores(y, s).ap() == pytest.approx(0.5 + 0.5 * 2 / 3)
    # perfect ranking
    assert auprc([1, 1, 0, 0], [4, 3, 2, 1]) == pytest.approx(1.0)
    # worst ranking with 1 pos among 4: P = 1/4
    assert auprc([0, 0, 0, 1], [4, 3, 2, 1]) == pytest.approx(0.25)


def test_ap_ties_block_semantics():
    # tied block {-,+} at 0.5 enters together: thresholds .9 (P1,R.5), .5 (P 2/3, R 1)
    y, s = [1, 0, 1, 0], [0.9, 0.5, 0.5, 0.1]
    assert auprc(y, s) == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert PresortedScores(y, s).ap() == pytest.approx(0.5 + 0.5 * 2 / 3)
    # order of tied items does not matter
    assert auprc([0, 1, 1, 0], [0.5, 0.9, 0.5, 0.1]) == pytest.approx(auprc(y, s))
    # constant scorer -> prevalence exactly
    yy = np.r_[np.ones(7), np.zeros(93)]
    assert auprc(yy, np.zeros(100)) == pytest.approx(0.07)
    assert PresortedScores(yy, np.zeros(100)).ap() == pytest.approx(0.07)


def test_presorted_matches_sklearn_with_weights():
    rng = np.random.default_rng(SEEDS[0])
    y = rng.random(3000) < 0.05
    s = np.round(rng.normal(size=3000) + y, 1)          # heavy ties
    w = rng.integers(1, 4, size=3000)
    ps = PresortedScores(y, s)
    assert ps.ap() == pytest.approx(average_precision_score(y, s), abs=1e-12)
    yd, sd = np.repeat(y, w), np.repeat(s, w)          # weights == multiplicities
    assert ps.ap(w) == pytest.approx(average_precision_score(yd, sd), abs=1e-12)


def test_precision_at_recall_hand():
    y, s = [1, 0, 1, 0, 1, 0], [6, 5, 4, 3, 2, 1]
    out = precision_at_recall(y, s, levels=(0.3, 0.6, 1.0))
    assert out["P@R0.3"] == pytest.approx(1.0)
    assert out["P@R0.6"] == pytest.approx(2 / 3)
    assert out["P@R1"] == pytest.approx(3 / 5)


def test_bootstrap_ci_covers_truth():
    """Binormal scores; the 'true' AP is estimated on a very large sample from the same distribution.
    Over 60 independent datasets the 95% CI should cover it close to nominally."""
    rng = np.random.default_rng(SEEDS[1])
    def draw(npos, nneg):
        y = np.r_[np.ones(npos, bool), np.zeros(nneg, bool)]
        return y, rng.normal(size=npos + nneg) + 1.5 * y
    yt, st = draw(200_000, 4_000_000)
    truth = auprc(yt, st)
    cover = 0
    for i in range(60):
        y, s = draw(300, 6000)
        b = bootstrap_auprc(y, s, n_boot=300, seed=SEEDS[0] + i)
        cover += b["ci_low"] <= truth <= b["ci_high"]
    assert cover / 60 >= 0.85


def test_paired_diff():
    rng = np.random.default_rng(SEEDS[2])
    y = rng.random(5000) < 0.05
    s = rng.normal(size=5000) + y
    d0 = bootstrap_auprc_diff(y, s, s, n_boot=200)
    assert d0["diff"] == 0 and d0["ci_low"] == 0 and d0["ci_high"] == 0
    s_better = s + 1.5 * y
    d = bootstrap_auprc_diff(y, s_better, s, n_boot=300)
    assert d["diff"] > 0 and d["ci_low"] > 0 and d["p_two_sided"] < 0.01


def test_negative_control_shuffled_labels():
    rng = np.random.default_rng(SEEDS[3])
    y = rng.random(200_000) < 0.01
    s = rng.normal(size=200_000) + 2 * y
    assert auprc(y, s) > 0.2
    ysh = rng.permutation(y)
    ev = evaluate(ysh, s)
    assert ev["auprc"] == pytest.approx(ysh.mean(), abs=0.003)
    assert ev["auroc"] == pytest.approx(0.5, abs=0.02)
