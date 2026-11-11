"""
Md Mainuddin Sagar-
-
metrics.py evaluation metrics for optimization histories.
"""
from __future__ import annotations
import numpy as np
from src.reward_surfaces import RewardSurface

def _true_best_curve(history: dict) -> np.ndarray:
    tr = np.asarray(history["true_rewards"], dtype=float)
    return np.maximum.accumulate(tr)

def simple_regret(history: dict, surface: RewardSurface) -> float:
    curve = _true_best_curve(history)
    return float(surface.optimum_value - curve[-1])

def best_so_far_curve(history: dict) -> np.ndarray:
    return _true_best_curve(history)

def trials_to_reach(history: dict, threshold: float) -> int | None:
    curve = _true_best_curve(history)
    idx = np.where(curve >= threshold)[0]
    return int(idx[0] + 1) if idx.size > 0 else None

def normalized_auc(history: dict, surface: RewardSurface) -> float:
    curve = _true_best_curve(history)
    T = len(curve)
    if T == 0 or surface.optimum_value <= 0:
        return 0.0
    # np.trapz was removed in NumPy 2.0; np.trapezoid is the replacement.
    a = np.trapezoid(curve, dx=1.0)
    ideal = surface.optimum_value * (T - 1)
    return float(a / max(ideal, 1e-9))
