"""Tests for src.metrics."""
import numpy as np
import pytest
from src.metrics import (
    best_so_far_curve,
    normalized_auc,
    simple_regret,
    trials_to_reach,
)
from src.reward_surfaces import all_surfaces

@pytest.fixture(scope="module")
def surface():
    return all_surfaces()[0]

def _toy_history(true_rewards):
    tr = list(true_rewards)
    return {
        "true_rewards": tr,
        "rewards": tr,
        "best_so_far": list(np.maximum.accumulate(np.array(tr, dtype=float))),
        "params": [np.zeros(4) for _ in tr],
    }

def test_best_so_far_curve(surface):
    h = _toy_history([0.1, 0.3, 0.2, 0.5, 0.4])
    c = best_so_far_curve(h)
    assert np.allclose(c, [0.1, 0.3, 0.3, 0.5, 0.5])

def test_simple_regret(surface):
    # Final best = 0.8, so regret = optimum_value - 0.8
    h = _toy_history([0.1, 0.5, 0.8, 0.6])
    r = simple_regret(h, surface)
    assert r == pytest.approx(surface.optimum_value - 0.8, rel=1e-6)

def test_trials_to_reach(surface):
    h = _toy_history([0.1, 0.3, 0.5, 0.6, 0.7])
    assert trials_to_reach(h, 0.4) == 3   # first index where cumulative best >= 0.4
    assert trials_to_reach(h, 0.9) is None

def test_normalized_auc_range(surface):
    h = _toy_history([0.9, 0.9, 0.9, 0.9, 0.9])
    a = normalized_auc(h, surface)
    assert 0.0 <= a <= 1.0
    # If every step is at optimum, AUC should be close to 1
    assert a >= 0.9 - 1e-9
