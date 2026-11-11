"""
Md Mainuddin Sagar-
-
human_oracle.py simulated human feedback used by the HIL optimizer.

Four feedback modes are supported:
1. simulate_scalar_response(surface, params, noise_std, rng)
     -> float in [0, 1] with additive Gaussian noise, clipped to [0, 1].
     Models a metabolic-cost-like continuous measurement.

2. simulate_preference_response(surface, params_A, params_B, noise_beta, rng)
     -> {-1, 0, +1}
     Bradley-Terry model: P(A > B) = sigmoid(beta * (f(A) - f(B))).
     +1 means A is preferred, -1 means B is preferred, 0 means tie.
     Ties occur when the two rewards are exactly equal after noise.

3. simulate_jnd_aware_preference(surface, params_A, params_B, jnd, noise_beta, rng)
     -> {-1, 0, +1}
     Same as (2), but returns 0 (tie / too close to call) when the L-infinity
     distance in normalized parameter space is BELOW the JND vector.
     This is the perception-aware comparator that motivates the project.

4. simulate_ordinal_rating(surface, params, n_levels, noise_std, rng)
     -> int in {1, ..., n_levels}
     1 = worst, n_levels = best. A noisy 5-point Likert-style survey response.

All functions accept an optional numpy Generator (rng) so experiments
are reproducible by seeding.
"""
from __future__ import annotations
import numpy as np
from src.reward_surfaces import RewardSurface
from src.torque_profile import PARAM_BOUNDS, PARAM_JND, clamp_params

# Helpers
def _default_rng(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _normalized_distance(params_A: np.ndarray, params_B: np.ndarray) -> np.ndarray:
    """Element-wise |Δ| / range, in physical units → unit cube."""
    a = clamp_params(np.asarray(params_A, dtype=float))
    b = clamp_params(np.asarray(params_B, dtype=float))
    span = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]
    return np.abs(a - b) / span

# 1. Scalar response
def simulate_scalar_response(surface: RewardSurface,
                             params: np.ndarray,
                             noise_std: float = 0.02,
                             rng=None) -> float:
    """Return a noisy scalar reward in [0, 1]."""
    rng = _default_rng(rng)
    truth = surface.f(params)
    noisy = truth + rng.normal(0.0, noise_std)
    return float(np.clip(noisy, 0.0, 1.0))

# 2. Bradley-Terry preference
def simulate_preference_response(surface: RewardSurface,
                                 params_A: np.ndarray,
                                 params_B: np.ndarray,
                                 noise_beta: float = 5.0,
                                 rng=None) -> int:
    """
    Return +1 if A is preferred, -1 if B is preferred, 0 if tie.

    The probability that the simulated user prefers A over B is
    sigmoid(beta * (f(A) - f(B))). If the two rewards are identical,
    a tie is returned deterministically.
    """
    rng = _default_rng(rng)
    fA = surface.f(params_A)
    fB = surface.f(params_B)
    if np.isclose(fA, fB, atol=1e-12):
        return 0
    pA = _sigmoid(noise_beta * (fA - fB))
    return 1 if rng.random() < pA else -1

# 3. JND-aware preference (the project's novel comparator)
def simulate_jnd_aware_preference(surface: RewardSurface,
                                  params_A: np.ndarray,
                                  params_B: np.ndarray,
                                  jnd: np.ndarray | None = None,
                                  noise_beta: float = 5.0,
                                  rng=None) -> int:
    """
    Return +1 if A is preferred, -1 if B is preferred,
    0 if the two parameter sets are perceptually indistinguishable
    (i.e. every coordinate's normalized difference is below the JND).

    Perceptual indistinguishability is checked FIRST: if the user cannot
    tell the two settings apart, no preference is reported regardless of
    the underlying reward difference. This is what makes the optimizer
    perception-aware.
    """
    rng = _default_rng(rng)
    if jnd is None:
        jnd = PARAM_JND

    dist = _normalized_distance(params_A, params_B)
    if np.all(dist < jnd):
        return 0

    return simulate_preference_response(surface, params_A, params_B,
                                        noise_beta=noise_beta, rng=rng)

# 4. Ordinal rating
def simulate_ordinal_rating(surface: RewardSurface,
                            params: np.ndarray,
                            n_levels: int = 5,
                            noise_std: float = 0.08,
                            rng=None) -> int:
    """
    Return an integer in [1, n_levels].

    We treat the noisy continuous reward r in [0, 1] as a Likert score by
    binning it uniformly into n_levels bins.
    """
    rng = _default_rng(rng)
    r = simulate_scalar_response(surface, params, noise_std=noise_std, rng=rng)
    # bin [0,1] into n_levels equal bins; 1 is worst, n_levels is best
    level = int(np.floor(r * n_levels)) + 1
    return int(np.clip(level, 1, n_levels))
