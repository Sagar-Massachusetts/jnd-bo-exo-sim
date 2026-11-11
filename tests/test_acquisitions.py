"""Tests for src.acquisitions and src.experiment (integration)."""
import numpy as np
import pytest
from src.acquisitions import (
    expected_improvement,
    grid_acquire,
    jnd_aware_expected_improvement,
    random_acquire,
    thompson_sampling,
)
from src.experiment import run_optimizer
from src.gp_model import GPModel
from src.reward_surfaces import all_surfaces
from src.torque_profile import PARAM_BOUNDS

def _small_train(surface, n=30, seed=0):
    rng = np.random.default_rng(seed)
    span = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]
    P = PARAM_BOUNDS[:, 0] + rng.uniform(0, 1, size=(n, 4)) * span
    y = np.array([surface.f(p) for p in P])
    return P, y

@pytest.fixture(scope="module")
def surface():
    return all_surfaces()[0]   # unimodal

def test_random_acquire_returns_valid_bounds(surface):
    p = random_acquire(surface, None, None, rng=0)
    assert p.shape == (4,)
    for i, v in enumerate(p):
        lo, hi = PARAM_BOUNDS[i]
        assert lo - 1e-9 <= v <= hi + 1e-9

def test_grid_acquire_skips_seen_points(surface):
    p1 = grid_acquire(surface, None, None, grid_resolution=3, rng=0)
    p2 = grid_acquire(surface, None, p1[None, :], grid_resolution=3, rng=0)
    assert not np.allclose(p1, p2), "grid_acquire returned the same point twice"

def test_ei_returns_valid_point(surface):
    X, y = _small_train(surface, n=20, seed=0)
    gp = GPModel(seed=0).fit(X, y)
    p = expected_improvement(surface, gp, X, y_seen=y, rng=0)
    assert p.shape == (4,)
    for i, v in enumerate(p):
        lo, hi = PARAM_BOUNDS[i]
        assert lo - 1e-9 <= v <= hi + 1e-9

def test_ts_returns_valid_point(surface):
    X, y = _small_train(surface, n=20, seed=0)
    gp = GPModel(seed=0).fit(X, y)
    p = thompson_sampling(surface, gp, X, rng=0)
    assert p.shape == (4,)

def test_jnd_ei_returns_valid_point(surface):
    X, y = _small_train(surface, n=20, seed=0)
    gp = GPModel(seed=0).fit(X, y)
    idx = int(np.argmax(y))
    p = jnd_aware_expected_improvement(surface, gp, X,
                                       y_seen=y, best_params=X[idx], rng=0)
    assert p.shape == (4,)

def test_jnd_ei_avoids_same_best_point(surface):
    """
    So, I think if the current best is already the true optimum and the reward is
    deterministic, JND-EI should NOT immediately re-query a point within JND
    of that optimum; it should explore.
    """
    from src.torque_profile import PARAM_JND, PARAM_BOUNDS
    X = surface.optimum[None, :].copy()
    y = np.array([surface.f(X[0])])
    gp = GPModel(seed=0).fit(X, y)
    p = jnd_aware_expected_improvement(
        surface, gp, X, y_seen=y, best_params=X[0], jnd=PARAM_JND, rng=0,
    )
    span = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]
    dist = np.abs(p - X[0]) / span
    # The returned point should be MORE than JND away in at least one coordinate
    assert np.any(dist >= PARAM_JND), \
        f"JND-EI returned a point perceptually indistinguishable from best: dist={dist}"

def test_run_optimizer_ei_beats_random_on_unimodal(surface):
    """
    Again smoke test: on a unimodal surface, EI should achieve a higher best_true
    than random, given the same budget and seed. Loose threshold — we only
    require EI to be at least as good as random within noise.
    """
    h_ei = run_optimizer(surface, "ei", budget=15, n_initial=5, seed=0,
                         noise_std=0.02, gp_noise_std=0.03)
    h_rand = run_optimizer(surface, "random", budget=15, n_initial=5, seed=0,
                           noise_std=0.02, gp_noise_std=0.03)
    best_ei = max(h_ei["true_rewards"])
    best_rand = max(h_rand["true_rewards"])
    assert best_ei > best_rand - 0.20, \
        f"EI best {best_ei} way below random best {best_rand}"
