"""
Md Mainuddin Sagar-
-
reward_surfaces.py four synthetic reward surfaces with known optima, used to compare
optimization strategies fairly. Each surface maps a 4-parameter
torque profile to a scalar reward in [0, 1].
Parameter order (matches src.torque_profile):
    0: rise_time      [fraction of stride]
    1: peak_timing    [fraction of stride]
    2: fall_time      [fraction of stride]
    3: peak_torque    [N m / kg]

Surfaces
UnimodalSurface       : single smooth peak, ideal case.
MultimodalSurface     : two distinct peaks, tests local-optimum trapping.
AsymmetricSurface     : sharp rise / slow fall in the reward surface.
TimingSensitiveSurface: reward sensitive to peak_timing, insensitive to peak_torque.

Each surface exposes:
    .name            : str
    .optimum         : np.ndarray (4,)
    .optimum_value   : float
    .f(params)       : float  (scalar reward in [0, 1])
    .f_batch(P)      : np.ndarray (N,)  where P is (N, 4)
"""
from __future__ import annotations

import numpy as np

from src.torque_profile import PARAM_BOUNDS, clamp_params

# Normalized-in-unit-cube helpers
# We map parameters to [0, 1] using PARAM_BOUNDS so that all surfaces are
# expressed in the same geometry. This makes comparisons between surfaces fair I believe.

_LO = PARAM_BOUNDS[:, 0]
_HI = PARAM_BOUNDS[:, 1]

def _to_unit(params: np.ndarray) -> np.ndarray:
    """Map params from PARAM_BOUNDS to [0, 1]^4 (clamped)."""
    p = clamp_params(np.asarray(params, dtype=float))
    return (p - _LO) / (_HI - _LO)

def _from_unit(u: np.ndarray) -> np.ndarray:
    """Map [0, 1]^4 back to PARAM_BOUNDS (clamped)."""
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    return _LO + u * (_HI - _LO)

def _gauss(u: np.ndarray, center: np.ndarray, sigma: float) -> float:
    """Isotropic Gaussian bump in unit cube."""
    d2 = np.sum((u - center) ** 2)
    return float(np.exp(-d2 / (2.0 * sigma ** 2)))

# Base class
class RewardSurface:
    name: str = "base"
    optimum: np.ndarray = None          # (4,) in physical units
    optimum_value: float = None         # float

    def f(self, params: np.ndarray) -> float:
        raise NotImplementedError

    def f_batch(self, P: np.ndarray) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, dtype=float))
        return np.array([self.f(row) for row in P], dtype=float)

# 1. Unimodal
class UnimodalSurface(RewardSurface):
    """
    A single smooth Gaussian peak. This is the easy case: any reasonable
    optimizer should find it. Serves as a sanity baseline for BO.
    """
    name = "unimodal"

    _CENTER = np.array([0.45, 0.50, 0.40, 0.55])   # in unit cube
    _SIGMA  = 0.18

    def __init__(self):
        self.optimum = _from_unit(self._CENTER)
        self.optimum_value = self.f(self.optimum)

    def f(self, params: np.ndarray) -> float:
        u = _to_unit(params)
        return _gauss(u, self._CENTER, self._SIGMA)

# 2. Multimodal
class MultimodalSurface(RewardSurface):
    """
    Two distinct peaks with the same height (both are global optima). This
    tests whether an optimizer gets trapped in one basin.
    """
    name = "multimodal"

    _C1 = np.array([0.20, 0.25, 0.20, 0.70])   # in unit cube
    _C2 = np.array([0.80, 0.75, 0.80, 0.30])
    _SIGMA = 0.15

    def __init__(self):
        # Reporting C1 as the canonical optimum (equal value at C2).
        self.optimum = _from_unit(self._C1)
        self.optimum_value = self.f(self.optimum)

    def f(self, params: np.ndarray) -> float:
        u = _to_unit(params)
        return 0.5 * _gauss(u, self._C1, self._SIGMA) + 0.5 * _gauss(u, self._C2, self._SIGMA)


# 3. Asymmetric
class AsymmetricSurface(RewardSurface):
    """
    Same single-peak location as unimodal, but with a sharp rise (in
    peak_timing) and a slow fall. Tests optimizer performance on skewed
    surfaces where gradient direction is biased.
    """
    name = "asymmetric"

    _CENTER = np.array([0.45, 0.40, 0.40, 0.55])   # in unit cube
    _SIGMA_FAST = 0.08   # in +rise_time, +peak_timing direction
    _SIGMA_SLOW = 0.25   # in -rise_time, -peak_timing direction

    def __init__(self):
        self.optimum = _from_unit(self._CENTER)
        self.optimum_value = self.f(self.optimum)

    def f(self, params: np.ndarray) -> float:
        u = _to_unit(params)
        d = u - self._CENTER
        # Directionally-weighted sigma: narrower on the positive side for the
        # two "timing" dimensions (0 and 1); symmetric on dims 2 and 3.
        sigma_vec = np.where(d > 0, self._SIGMA_FAST, self._SIGMA_SLOW)
        # dims 2, 3 stay symmetric
        sigma_vec[2] = 0.18
        sigma_vec[3] = 0.18
        exponent = -0.5 * np.sum((d / sigma_vec) ** 2)
        return float(np.exp(exponent))

# 4. Timing-sensitive
class TimingSensitiveSurface(RewardSurface):
    """
    Reward depends strongly on peak_timing and barely on peak_torque.
    Motivation: the human JND for timing is small (~3% stride) whereas the
    JND for magnitude is larger (~10%). This surface mimics that perceptual
    asymmetry and is the one most relevant to our JND-aware optimizer.
    """
    name = "timing_sensitive"

    _CENTER = np.array([0.45, 0.50, 0.40, 0.55])   # in unit cube
    _SIGMA_TIMING = 0.10
    _SIGMA_MAG    = 0.35

    def __init__(self):
        self.optimum = _from_unit(self._CENTER)
        self.optimum_value = self.f(self.optimum)

    def f(self, params: np.ndarray) -> float:
        u = _to_unit(params)
        d = u - self._CENTER
        sigma_vec = np.array([self._SIGMA_TIMING, self._SIGMA_TIMING,
                              self._SIGMA_TIMING, self._SIGMA_MAG])
        exponent = -0.5 * np.sum((d / sigma_vec) ** 2)
        return float(np.exp(exponent))

# Registry
def all_surfaces() -> list[RewardSurface]:
    """Convenient list of all four surfaces (used by tests and demo)."""
    return [UnimodalSurface(), MultimodalSurface(),
            AsymmetricSurface(), TimingSensitiveSurface()]
