"""Tests for src.torque_profile."""
import numpy as np
import pytest

from src import torque_profile as tp


def test_constants_shapes():
    assert len(tp.PARAM_NAMES) == 4
    assert tp.PARAM_BOUNDS.shape == (4, 2)
    assert tp.PARAM_JND.shape == (4,)
    assert np.all(tp.PARAM_BOUNDS[:, 0] < tp.PARAM_BOUNDS[:, 1])


def test_clamp_params_inside_bounds_is_identity():
    p = np.array([0.20, 0.45, 0.10, 0.50])
    out = tp.clamp_params(p)
    assert np.allclose(out, p)


def test_clamp_params_outside_bounds_are_clamped():
    p = np.array([0.00, 1.00, -0.5, 5.0])
    out = tp.clamp_params(p)
    lo = tp.PARAM_BOUNDS[:, 0]
    hi = tp.PARAM_BOUNDS[:, 1]
    assert np.all(out >= lo) and np.all(out <= hi)


def test_torque_zero_outside_active_window():
    params = np.array([0.20, 0.45, 0.10, 0.50])
    assert tp.torque_profile(0.0, params) == pytest.approx(0.0, abs=1e-12)
    assert tp.torque_profile(0.9, params) == pytest.approx(0.0, abs=1e-12)


def test_torque_peaks_at_peak_timing():
    params = np.array([0.20, 0.45, 0.10, 0.50])
    mass = 70.0
    tau = tp.torque_profile(0.45, params, body_mass_kg=mass)
    expected = 0.50 * mass
    assert tau == pytest.approx(expected, rel=1e-6)


def test_torque_monotone_rise_and_fall():
    params = np.array([0.20, 0.45, 0.10, 0.50])
    mass = 70.0
    rise_phases = np.linspace(0.25, 0.449, 20)
    rise_tau = tp.torque_profile_array(rise_phases, params, mass)
    assert np.all(np.diff(rise_tau) > 0)
    fall_phases = np.linspace(0.451, 0.55, 20)
    fall_tau = tp.torque_profile_array(fall_phases, params, mass)
    assert np.all(np.diff(fall_tau) < 0)


def test_vectorized_wrapper_matches_scalar():
    params = np.array([0.20, 0.45, 0.10, 0.50])
    phases = np.array([0.0, 0.30, 0.45, 0.50, 0.80])
    vec = tp.torque_profile_array(phases, params)
    for p, v in zip(phases, vec):
        assert tp.torque_profile(float(p), params) == pytest.approx(v, rel=1e-12)
