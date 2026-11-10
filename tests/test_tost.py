"""Tests for src.tost."""
import numpy as np
import pytest

from src.tost import best_params_from_history, tost, tost_on_optima


def test_tost_identical_samples_are_equivalent():
    """Two samples from the same distribution should be declared equivalent
    under a loose bound."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.01, size=20)
    b = rng.normal(0.5, 0.01, size=20)
    res = tost(a, b, bound=0.1)
    assert res["equivalent"] is True
    assert res["p_max"] < 0.05


def test_tost_obviously_different_samples_are_not_equivalent():
    rng = np.random.default_rng(1)
    a = rng.normal(0.9, 0.01, size=20)
    b = rng.normal(0.1, 0.01, size=20)
    res = tost(a, b, bound=0.05)
    assert res["equivalent"] is False
    assert res["p_max"] > 0.5   # cannot reject either null


def test_tost_ci_contains_mean_diff():
    rng = np.random.default_rng(2)
    a = rng.normal(0.5, 0.02, size=15)
    b = rng.normal(0.48, 0.02, size=15)
    res = tost(a, b, bound=0.1)
    assert res["ci_lo"] <= res["mean_diff"] <= res["ci_hi"]


def test_tost_bound_must_be_positive():
    a = np.random.default_rng(0).normal(0.5, 0.01, 10)
    b = np.random.default_rng(1).normal(0.5, 0.01, 10)
    with pytest.raises(ValueError):
        tost(a, b, bound=0.0)


def test_tost_needs_at_least_2_samples():
    with pytest.raises(ValueError):
        tost([1.0], [1.0], bound=0.1)


def test_tost_on_optima_all_equivalent():
    """If two methods find nearly identical optima, all 4 params should be
    equivalent."""
    rng = np.random.default_rng(0)
    base = np.array([0.2, 0.45, 0.1, 0.5])
    A = base[None, :] + rng.normal(0.0, 0.001, size=(20, 4))
    B = base[None, :] + rng.normal(0.0, 0.001, size=(20, 4))
    res = tost_on_optima(A, B)
    assert res["all_equivalent"] is True


def test_tost_on_optima_not_equivalent():
    """If one method finds a far-off optimum, that parameter should NOT be
    equivalent."""
    rng = np.random.default_rng(0)
    A = np.tile([0.2, 0.45, 0.1, 0.5], (20, 1)) + rng.normal(0.0, 0.001, size=(20, 4))
    B = np.tile([0.9, 0.45, 0.1, 0.5], (20, 1)) + rng.normal(0.0, 0.001, size=(20, 4))
    res = tost_on_optima(A, B)
    assert res["all_equivalent"] is False
    # parameter 0 (rise_time) should not be equivalent
    assert res["params"][0]["equivalent"] is False


def test_best_params_from_history():
    history = {
        "true_rewards": [0.1, 0.3, 0.9, 0.5],
        "params": [np.zeros(4), np.ones(4), 2 * np.ones(4), 3 * np.ones(4)],
    }
    p = best_params_from_history(history)
    assert np.array_equal(p, 2 * np.ones(4))
