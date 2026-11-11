"""Tests for src.gp_model."""
import numpy as np
import pytest
from src.gp_model import GPModel
from src.reward_surfaces import all_surfaces
from src.torque_profile import PARAM_BOUNDS

def _sample_grid(n_per_dim=8):
    """8^4 = 4096 training points — dense enough to resolve the Gaussian peak."""
    grids = [np.linspace(PARAM_BOUNDS[i, 0], PARAM_BOUNDS[i, 1], n_per_dim)
             for i in range(4)]
    from itertools import product
    return np.array(list(product(*grids)))

@pytest.fixture(scope="module")
def surface():
    return all_surfaces()[0]   # unimodal

@pytest.fixture(scope="module")
def training_data(surface):
    X = _sample_grid(n_per_dim=8)          # 4096 points
    y = np.array([surface.f(p) for p in X])
    return X, y

def test_fit_and_predict_shapes(surface, training_data):
    X, y = training_data
    gp = GPModel(noise_std=0.02, seed=0).fit(X, y)
    mu, sigma = gp.predict(X[:5])
    assert mu.shape == (5,)
    assert sigma.shape == (5,)
    assert np.all(sigma >= 0)

def test_posterior_mean_approximates_true_reward_on_training_points(surface, training_data):
    X, y = training_data
    gp = GPModel(noise_std=0.02, seed=0).fit(X, y)
    mu, _ = gp.predict(X)
    err = np.abs(mu - y)
    assert np.mean(err) < 0.10, f"mean abs err too high: {np.mean(err)}"

def test_posterior_mean_is_reasonable_at_optimum(surface, training_data):
    X, y = training_data
    gp = GPModel(noise_std=0.02, seed=0).fit(X, y)
    mu, _ = gp.predict(surface.optimum[None, :])
    # With 4096 training points the GP should resolve the Gaussian peak closely I believe.
    assert abs(float(mu[0]) - surface.optimum_value) < 0.15, \
        f"GP optimum prediction too far: {mu[0]} vs {surface.optimum_value}"

def test_posterior_std_decreases_with_more_data(surface):
    """
    Fair test: same test point, same data distribution, MORE data → smaller std.
    Both training sets are drawn from the SAME random generator with the SAME
    distribution — the only difference is N.
    """
    rng = np.random.default_rng(123)
    span = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]

    # Test point in the middle of the domain.
    test_pt = PARAM_BOUNDS[:, 0] + 0.5 * span

    # Both sets are drawn uniformly from the same distribution.
    def _sample(N):
        P = PARAM_BOUNDS[:, 0] + rng.uniform(0, 1, size=(N, 4)) * span
        y = np.array([surface.f(p) for p in P])
        return P, y

    P_small, y_small = _sample(8)
    P_large, y_large = _sample(400)

    gp_small = GPModel(noise_std=0.02, seed=0).fit(P_small, y_small)
    gp_large = GPModel(noise_std=0.02, seed=0).fit(P_large, y_large)

    _, s_small = gp_small.predict(test_pt[None, :])
    _, s_large = gp_large.predict(test_pt[None, :])
    assert float(s_small[0]) > float(s_large[0]), \
        f"std did not decrease with more data: {s_small[0]} vs {s_large[0]}"

def test_predict_best_returns_a_valid_point(surface, training_data):
    X, y = training_data
    gp = GPModel(noise_std=0.02, seed=0).fit(X, y)
    best_mu, best_sigma, best_params = gp.predict_best(X)
    assert best_params.shape == (4,)
    for i, p in enumerate(best_params):
        lo, hi = PARAM_BOUNDS[i]
        assert lo - 1e-6 <= p <= hi + 1e-6
    mu_all, _ = gp.predict(X)
    assert best_mu >= float(np.max(mu_all)) - 1e-6

def test_sample_posterior_shapes(surface, training_data):
    X, y = training_data
    gp = GPModel(noise_std=0.02, seed=0).fit(X, y)
    samples = gp.sample_posterior(X[:10], n_samples=4, rng=0)
    # Shape is (n_samples, M)
    assert samples.shape == (4, 10)

def test_seed_reproducibility(surface, training_data):
    X, y = training_data
    gp1 = GPModel(noise_std=0.02, seed=42).fit(X, y)
    gp2 = GPModel(noise_std=0.02, seed=42).fit(X, y)
    m1, s1 = gp1.predict(X[:5])
    m2, s2 = gp2.predict(X[:5])
    assert np.allclose(m1, m2)
    assert np.allclose(s1, s2)

def test_predict_without_fit_raises():
    gp = GPModel()
    with pytest.raises(RuntimeError):
        gp.predict(np.zeros((1, 4)))
