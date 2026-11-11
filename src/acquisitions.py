"""
Md Mainuddin Sagar-
-
acquisitions.py acquisition functions for the HIL optimizer.
"""
from __future__ import annotations
import math
import numpy as np
from src.gp_model import GPModel
from src.reward_surfaces import RewardSurface
from src.torque_profile import PARAM_BOUNDS, PARAM_JND

_LO = PARAM_BOUNDS[:, 0]
_HI = PARAM_BOUNDS[:, 1]
_SPAN = _HI - _LO

# Helpers
def _as_rng(rng):
    """Coerce rng (None, int, Generator, RandomState) into a modern Generator."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        return np.random.default_rng(rng.randint(0, 2**31 - 1))
    raise TypeError(f"Unsupported rng type: {type(rng)}")

def _normal_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

def _normal_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)

def _ei_terms(mu, sigma, best, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - best - xi) / sigma
    return (mu - best - xi) * _normal_cdf(z) + sigma * _normal_pdf(z)

def _candidate_pool(n=2048, rng=None):
    rng = _as_rng(rng)
    u = rng.uniform(0.0, 1.0, size=(n, 4))
    return _LO + u * _SPAN

# Baselines
def random_acquire(surface, gp, X_seen, rng=None):
    rng = _as_rng(rng)
    u = rng.uniform(0.0, 1.0, size=4)
    return _LO + u * _SPAN

def grid_acquire(surface, gp, X_seen, grid_resolution=6, rng=None):
    from itertools import product
    grids = [np.linspace(_LO[i], _HI[i], grid_resolution) for i in range(4)]
    candidates = np.array(list(product(*grids)))

    seen = np.atleast_2d(X_seen) if X_seen is not None else np.zeros((0, 4))
    if seen.shape[0] > 0:
        keep = np.ones(candidates.shape[0], dtype=bool)
        for idx, pt in enumerate(candidates):
            if np.any(np.all(np.isclose(seen, pt, atol=1e-9), axis=1)):
                keep[idx] = False
        candidates = candidates[keep]

    if candidates.shape[0] == 0:
        return random_acquire(surface, gp, X_seen, rng=rng)
    return candidates[0]

# Standard BO — Expected Improvement
def expected_improvement(surface, gp, X_seen, y_seen=None,
                         best_so_far=None, n_candidates=2048, xi=0.01, rng=None):
    rng = _as_rng(rng)
    if best_so_far is None:
        if y_seen is None:
            raise ValueError("Either best_so_far or y_seen must be provided.")
        best_so_far = float(np.max(y_seen))
    pool = _candidate_pool(n=n_candidates, rng=rng)
    mu, sigma = gp.predict(pool)
    ei = _ei_terms(mu, sigma, best_so_far, xi=xi)
    return pool[int(np.argmax(ei))]

# Standard BO — Thompson Sampling
def thompson_sampling(surface, gp, X_seen, n_candidates=1024,
                      n_samples=1, rng=None):
    rng = _as_rng(rng)
    pool = _candidate_pool(n=n_candidates, rng=rng)
    samples = gp.sample_posterior(pool, n_samples=n_samples, rng=rng)
    return pool[int(np.argmax(samples[0]))]

# Proposed — JND-aware EI
def jnd_aware_expected_improvement(surface, gp, X_seen,
                                   y_seen=None, best_so_far=None,
                                   best_params=None, jnd=None,
                                   n_candidates=2048, xi=0.01, rng=None):
    rng = _as_rng(rng)
    if jnd is None:
        jnd = PARAM_JND
    if best_so_far is None:
        if y_seen is None:
            raise ValueError("Either best_so_far or y_seen must be provided.")
        best_so_far = float(np.max(y_seen))

    pool = _candidate_pool(n=n_candidates, rng=rng)
    mu, sigma = gp.predict(pool)
    ei = _ei_terms(mu, sigma, best_so_far, xi=xi)

    if best_params is not None:
        dist = np.abs(pool - best_params[None, :]) / _SPAN[None, :]
        within_jnd = np.all(dist < jnd[None, :], axis=1)
        if np.any(~within_jnd) and np.any(ei[~within_jnd] > 0):
            ei_effective = ei.copy()
            ei_effective[within_jnd] *= 0.05
        else:
            ei_effective = ei
    else:
        ei_effective = ei

    return pool[int(np.argmax(ei_effective))]
