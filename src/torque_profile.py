"""
Md Mainuddin Sagar-
-
torque_profile.py constrained 4-parameter ankle-exoskeleton torque policy.

Parameter order (fixed project-wide):
    0: rise_time      [fraction of stride]   how long the torque takes to ramp up
    1: peak_timing    [fraction of stride]   when the peak torque occurs
    2: fall_time      [fraction of stride]   how long the torque takes to fall
    3: peak_torque    [N m / kg]             peak torque magnitude (per kg body mass)

The profile is::

    rise from 0 to peak between  (peak_timing - rise_time)  and  peak_timing
    fall from peak to 0 between  peak_timing  and  (peak_timing + fall_time)
    zero elsewhere in the stride
"""

from __future__ import annotations
import numpy as np

# Bounds (safety envelopes for a healthy-user ankle exoskeleton study)
PARAM_NAMES = ["rise_time", "peak_timing", "fall_time", "peak_torque"]

PARAM_BOUNDS = np.array([
    [0.10, 0.40],
    [0.30, 0.60],
    [0.05, 0.20],
    [0.10, 1.00],
], dtype=float)

# Provisional JNDs (perceptual thresholds) aligned with PARAM_NAMES.
PARAM_JND = np.array([0.030, 0.030, 0.030, 0.100], dtype=float)

def clamp_params(params: np.ndarray) -> np.ndarray:
    """Clamp a 4-vector of parameters into PARAM_BOUNDS."""
    params = np.asarray(params, dtype=float).reshape(4,)
    lo = PARAM_BOUNDS[:, 0]
    hi = PARAM_BOUNDS[:, 1]
    return np.clip(params, lo, hi)

def normalize_phase(phase: float) -> float:
    """Wrap any phase into [0, 1)."""
    return float(phase) % 1.0

def torque_profile(phase: float, params: np.ndarray, body_mass_kg: float = 70.0) -> float:
    """
    Return the ankle torque (N*m) at gait phase `phase` for parameter vector `params`.
    """
    p = normalize_phase(phase)
    rise_time, peak_timing, fall_time, peak_torque = clamp_params(params)

    tau_peak = peak_torque * body_mass_kg

    t_start = peak_timing - rise_time
    t_end   = peak_timing + fall_time

    if tau_peak <= 0.0:
        return 0.0

    if t_start <= p < peak_timing:
        eff_start = max(t_start, 0.0)
        denom = max(peak_timing - eff_start, 1e-9)
        return tau_peak * (p - eff_start) / denom

    if peak_timing <= p < t_end:
        denom = max(t_end - peak_timing, 1e-9)
        return tau_peak * (t_end - p) / denom

    return 0.0

def torque_profile_array(phases: np.ndarray, params: np.ndarray,
                         body_mass_kg: float = 70.0) -> np.ndarray:
    """Vectorized convenience wrapper for plotting."""
    phases = np.asarray(phases, dtype=float)
    return np.array([torque_profile(float(p), params, body_mass_kg) for p in phases])
