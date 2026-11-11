"""Tests for src.walker profile."""
import numpy as np
import pytest
from src.walker import Walker

def test_phase_starts_at_phase0():
    w = Walker(stride_seconds=1.0, phase0=0.25)
    assert w.gait_phase() == pytest.approx(0.25)

def test_phase_advances_at_expected_rate():
    w = Walker(stride_seconds=1.0, phase0=0.0)
    w.step(0.25)
    assert w.gait_phase() == pytest.approx(0.25)

def test_phase_wraps_into_unit_interval():
    w = Walker(stride_seconds=1.0, phase0=0.0)
    w.step(1.75)
    assert 0.0 <= w.gait_phase() < 1.0
    assert w.gait_phase() == pytest.approx(0.75)

def test_phase_is_periodic():
    w = Walker(stride_seconds=1.0, phase0=0.0)
    phases = [w.gait_phase_at(t) for t in (0.0, 1.0, 2.0, 3.0)]
    for ph in phases:
        assert ph == pytest.approx(phases[0])

def test_reset_returns_to_start():
    w = Walker(stride_seconds=1.0, phase0=0.10)
    w.step(0.60)
    w.reset()
    assert w.gait_phase() == pytest.approx(0.10)
    assert w.time() == 0.0

def test_negative_dt_raises():
    w = Walker()
    with pytest.raises(ValueError):
        w.step(-0.01)

def test_invalid_stride_raises():
    with pytest.raises(ValueError):
        Walker(stride_seconds=0.0)
