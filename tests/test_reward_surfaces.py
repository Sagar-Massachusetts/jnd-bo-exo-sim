"""Tests for src.reward_surfaces."""
import numpy as np
import pytest

from src.reward_surfaces import (
    AsymmetricSurface, MultimodalSurface, RewardSurface,
    TimingSensitiveSurface, UnimodalSurface, all_surfaces,
)
from src.torque_profile import PARAM_BOUNDS


def test_all_surfaces_have_required_interface():
    for s in all_surfaces():
        assert hasattr(s, "name")
        assert hasattr(s, "optimum")
        assert hasattr(s, "optimum_value")
        assert s.optimum.shape == (4,)
        assert isinstance(s.optimum_value, float)


def test_optimum_is_bounded():
    for s in all_surfaces():
        for i, p in enumerate(s.optimum):
            lo, hi = PARAM_BOUNDS[i]
            assert lo <= p <= hi, f"{s.name}: param {i} out of bounds"


def test_reward_at_optimum_matches_optimum_value():
    for s in all_surfaces():
        assert s.f(s.optimum) == pytest.approx(s.optimum_value, rel=1e-8)


def test_reward_is_bounded_0_to_1_on_entire_domain():
    n = 6
    grid = [np.linspace(PARAM_BOUNDS[i, 0], PARAM_BOUNDS[i, 1], n) for i in range(4)]
    from itertools import product
    for s in all_surfaces():
        for combo in product(*grid):
            v = s.f(np.array(combo))
            assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9, f"{s.name}: reward out of [0,1]"


def test_unimodal_has_unique_global_max():
    s = UnimodalSurface()
    n = 6
    grid = [np.linspace(PARAM_BOUNDS[i, 0], PARAM_BOUNDS[i, 1], n) for i in range(4)]
    from itertools import product
    best = -1.0
    best_pt = None
    for combo in product(*grid):
        v = s.f(np.array(combo))
        if v > best:
            best = v
            best_pt = np.array(combo)
    gap = np.linalg.norm((best_pt - s.optimum) / (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]))
    assert gap < 0.30, f"unimodal: grid max far from true optimum, gap={gap}"


def test_multimodal_has_two_distinct_basins():
    s = MultimodalSurface()
    from src.reward_surfaces import _from_unit
    u1 = np.array([0.20, 0.25, 0.20, 0.70])
    u2 = np.array([0.80, 0.75, 0.80, 0.30])
    p1 = _from_unit(u1)
    p2 = _from_unit(u2)
    pm = _from_unit(0.5 * (u1 + u2))
    v1 = s.f(p1)
    v2 = s.f(p2)
    vm = s.f(pm)
    assert v1 > 0.4 and v2 > 0.4, f"multimodal: basin peaks too low: {v1}, {v2}"
    assert vm < min(v1, v2), f"multimodal: midpoint not a valley: {vm} vs {v1},{v2}"


def test_asymmetric_is_skewed_in_peak_timing():
    s = AsymmetricSurface()
    delta = 0.05
    opt = s.optimum.copy()
    p_plus = opt.copy();  p_plus[1]  += delta
    p_minus = opt.copy(); p_minus[1] -= delta
    assert s.f(p_minus) > s.f(p_plus), "asymmetric: peak_timing asymmetry not present"


def test_timing_sensitive_is_more_sensitive_to_timing_than_magnitude():
    s = TimingSensitiveSurface()
    opt = s.optimum.copy()
    delta = 0.05
    p_t = opt.copy(); p_t[1] += delta
    p_m = opt.copy(); p_m[3] += delta
    v0 = s.f(opt)
    d_t = v0 - s.f(p_t)
    d_m = v0 - s.f(p_m)
    assert d_t > d_m, f"timing_sensitive: not more sensitive to timing: {d_t} vs {d_m}"
