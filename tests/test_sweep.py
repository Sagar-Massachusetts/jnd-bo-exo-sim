"""Tests for src.sweep."""
import numpy as np
import pytest

from src.reward_surfaces import all_surfaces
from src.sweep import run_sweep


@pytest.fixture(scope="module")
def tiny_sweep():
    """A tiny sweep: 2 methods × 2 surfaces × 2 seeds to keep tests fast."""
    surfaces = all_surfaces()[:2]   # unimodal, multimodal
    return run_sweep(
        methods=["random", "ei"],
        surfaces=surfaces,
        n_seeds=2,
        budget=8,
        n_initial=3,
        noise_std=0.02,
        gp_noise_std=0.03,
        n_candidates=64,
    )


def test_sweep_produces_expected_keys(tiny_sweep):
    for key in ("best_so_far_mean", "best_so_far_lo", "best_so_far_hi",
                "regret_mean", "regret_std", "trials_to_reach_mean"):
        assert key in tiny_sweep


def test_sweep_curves_have_correct_length(tiny_sweep):
    budget = tiny_sweep["budget"]
    for m in tiny_sweep["methods"]:
        for s in tiny_sweep["surface_names"]:
            key = (m, s)
            assert tiny_sweep["best_so_far_mean"][key].shape == (budget,)
            assert tiny_sweep["best_so_far_lo"][key].shape == (budget,)
            assert tiny_sweep["best_so_far_hi"][key].shape == (budget,)


def test_sweep_lo_le_mean_le_hi(tiny_sweep):
    for m in tiny_sweep["methods"]:
        for s in tiny_sweep["surface_names"]:
            key = (m, s)
            lo = tiny_sweep["best_so_far_lo"][key]
            mid = tiny_sweep["best_so_far_mean"][key]
            hi = tiny_sweep["best_so_far_hi"][key]
            assert np.all(lo <= mid + 1e-9)
            assert np.all(mid <= hi + 1e-9)


def test_sweep_curves_are_monotone(tiny_sweep):
    for m in tiny_sweep["methods"]:
        for s in tiny_sweep["surface_names"]:
            mid = tiny_sweep["best_so_far_mean"][(m, s)]
            assert np.all(np.diff(mid) >= -1e-9), "best-so-far should be monotone"

