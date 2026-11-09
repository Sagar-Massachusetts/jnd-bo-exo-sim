"""Tests for src.human_oracle."""
import numpy as np
import pytest

from src.human_oracle import (
    simulate_jnd_aware_preference,
    simulate_ordinal_rating,
    simulate_preference_response,
    simulate_scalar_response,
)
from src.reward_surfaces import all_surfaces


@pytest.fixture
def surface():
    return all_surfaces()[0]   # unimodal


@pytest.fixture
def optimum(surface):
    return surface.optimum.copy()


@pytest.fixture
def far_point(surface):
    # Pick a point in the unit cube that is far from the optimum.
    from src.reward_surfaces import _from_unit
    return _from_unit(np.array([0.05, 0.05, 0.05, 0.05]))


# --- Scalar --------------------------------------------------------------
def test_scalar_response_bounded(surface, optimum):
    for seed in range(5):
        r = simulate_scalar_response(surface, optimum, noise_std=0.02, rng=seed)
        assert 0.0 <= r <= 1.0


def test_scalar_response_is_noisy(surface, optimum):
    vals = [simulate_scalar_response(surface, optimum, noise_std=0.05, rng=s)
            for s in range(30)]
    assert np.std(vals) > 0.0


def test_scalar_response_is_reproducible(surface, optimum):
    a = simulate_scalar_response(surface, optimum, noise_std=0.05, rng=42)
    b = simulate_scalar_response(surface, optimum, noise_std=0.05, rng=42)
    assert a == b


# --- Bradley-Terry preference --------------------------------------------
def test_preference_returns_minus1_or_0_or_1(surface, optimum, far_point):
    for seed in range(10):
        p = simulate_preference_response(surface, optimum, far_point, rng=seed)
        assert p in (-1, 0, 1)


def test_preference_prefers_better_point(surface, optimum, far_point):
    # At the optimum vs a far point, the optimizer should very often prefer
    # the optimum.
    votes = [simulate_preference_response(surface, optimum, far_point,
                                          noise_beta=10.0, rng=s)
             for s in range(50)]
    assert votes.count(1) > 0.9 * len(votes)


def test_preference_tie_when_identical(surface, optimum):
    p = simulate_preference_response(surface, optimum, optimum, rng=0)
    assert p == 0


# --- JND-aware preference ------------------------------------------------
def test_jnd_returns_tie_when_within_thresholds(surface, optimum):
    # A tiny perturbation smaller than half the JND in each coordinate
    from src.torque_profile import PARAM_BOUNDS, PARAM_JND
    span = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]
    delta = 0.3 * PARAM_JND * span   # 30% of JND → definitely imperceptible
    params_B = optimum + delta
    p = simulate_jnd_aware_preference(surface, optimum, params_B, rng=0)
    assert p == 0


def test_jnd_returns_preference_when_above_thresholds(surface, optimum, far_point):
    # far_point is far from the optimum → definitely perceptible
    p = simulate_jnd_aware_preference(surface, optimum, far_point, rng=0)
    assert p in (-1, 1)


def test_jnd_reproducible(surface, optimum, far_point):
    for s in range(5):
        a = simulate_jnd_aware_preference(surface, optimum, far_point, rng=s)
        b = simulate_jnd_aware_preference(surface, optimum, far_point, rng=s)
        assert a == b


# --- Ordinal rating ------------------------------------------------------
def test_ordinal_rating_in_range(surface, optimum):
    for seed in range(10):
        r = simulate_ordinal_rating(surface, optimum, n_levels=5, rng=seed)
        assert 1 <= r <= 5


def test_ordinal_rating_prefers_higher_reward(surface, optimum, far_point):
    # At optimum, ratings should average higher than at a far point.
    ratings_opt = [simulate_ordinal_rating(surface, optimum, n_levels=5, rng=s)
                   for s in range(30)]
    ratings_far = [simulate_ordinal_rating(surface, far_point, n_levels=5, rng=s)
                   for s in range(30)]
    assert np.mean(ratings_opt) > np.mean(ratings_far)
"""Tests for src.human_oracle."""
import numpy as np
import pytest

from src.human_oracle import (
    simulate_jnd_aware_preference,
    simulate_ordinal_rating,
    simulate_preference_response,
    simulate_scalar_response,
)
from src.reward_surfaces import all_surfaces


@pytest.fixture
def surface():
    return all_surfaces()[0]   # unimodal


@pytest.fixture
def optimum(surface):
    return surface.optimum.copy()


@pytest.fixture
def far_point(surface):
    # Pick a point in the unit cube that is far from the optimum.
    from src.reward_surfaces import _from_unit
    return _from_unit(np.array([0.05, 0.05, 0.05, 0.05]))


# --- Scalar --------------------------------------------------------------
def test_scalar_response_bounded(surface, optimum):
    for seed in range(5):
        r = simulate_scalar_response(surface, optimum, noise_std=0.02, rng=seed)
        assert 0.0 <= r <= 1.0


def test_scalar_response_is_noisy(surface, optimum):
    vals = [simulate_scalar_response(surface, optimum, noise_std=0.05, rng=s)
            for s in range(30)]
    assert np.std(vals) > 0.0


def test_scalar_response_is_reproducible(surface, optimum):
    a = simulate_scalar_response(surface, optimum, noise_std=0.05, rng=42)
    b = simulate_scalar_response(surface, optimum, noise_std=0.05, rng=42)
    assert a == b


# --- Bradley-Terry preference --------------------------------------------
def test_preference_returns_minus1_or_0_or_1(surface, optimum, far_point):
    for seed in range(10):
        p = simulate_preference_response(surface, optimum, far_point, rng=seed)
        assert p in (-1, 0, 1)


def test_preference_prefers_better_point(surface, optimum, far_point):
    # At the optimum vs a far point, the optimizer should very often prefer
    # the optimum.
    votes = [simulate_preference_response(surface, optimum, far_point,
                                          noise_beta=10.0, rng=s)
             for s in range(50)]
    assert votes.count(1) > 0.9 * len(votes)


def test_preference_tie_when_identical(surface, optimum):
    p = simulate_preference_response(surface, optimum, optimum, rng=0)
    assert p == 0


# --- JND-aware preference ------------------------------------------------
def test_jnd_returns_tie_when_within_thresholds(surface, optimum):
    # A tiny perturbation smaller than half the JND in each coordinate
    from src.torque_profile import PARAM_BOUNDS, PARAM_JND
    span = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]
    delta = 0.3 * PARAM_JND * span   # 30% of JND → definitely imperceptible
    params_B = optimum + delta
    p = simulate_jnd_aware_preference(surface, optimum, params_B, rng=0)
    assert p == 0


def test_jnd_returns_preference_when_above_thresholds(surface, optimum, far_point):
    # far_point is far from the optimum → definitely perceptible
    p = simulate_jnd_aware_preference(surface, optimum, far_point, rng=0)
    assert p in (-1, 1)


def test_jnd_reproducible(surface, optimum, far_point):
    for s in range(5):
        a = simulate_jnd_aware_preference(surface, optimum, far_point, rng=s)
        b = simulate_jnd_aware_preference(surface, optimum, far_point, rng=s)
        assert a == b


# --- Ordinal rating ------------------------------------------------------
def test_ordinal_rating_in_range(surface, optimum):
    for seed in range(10):
        r = simulate_ordinal_rating(surface, optimum, n_levels=5, rng=seed)
        assert 1 <= r <= 5


def test_ordinal_rating_prefers_higher_reward(surface, optimum, far_point):
    # At optimum, ratings should average higher than at a far point.
    ratings_opt = [simulate_ordinal_rating(surface, optimum, n_levels=5, rng=s)
                   for s in range(30)]
    ratings_far = [simulate_ordinal_rating(surface, far_point, n_levels=5, rng=s)
                   for s in range(30)]
    assert np.mean(ratings_opt) > np.mean(ratings_far)
