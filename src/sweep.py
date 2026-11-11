"""
Md Mainuddin Sagar-
-
sweep.py multi-seed optimizer sweep. Runs each (method, surface) pair across N seeds
and returns the aggregate curves and metrics.

Public API
    run_sweep(methods, surfaces, n_seeds=20, budget=30, n_initial=6,
              noise_std=0.02, gp_noise_std=0.03, n_candidates=256,
              out_dir="results/tables")
        -> dict with keys:
            'histories': {(method, surface_name): [history_seed0, ...]}
            'best_so_far_mean': {(method, surface_name): np.ndarray (budget,)}
            'best_so_far_lo':   {(method, surface_name): np.ndarray (budget,)}  # 2.5%
            'best_so_far_hi':   {(method, surface_name): np.ndarray (budget,)}  # 97.5%
            'regret_mean':      {(method, surface_name): float}
            'regret_std':       {(method, surface_name): float}
            'trials_to_reach_mean': {(method, surface_name): float}
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from src.experiment import run_optimizer
from src.metrics import best_so_far_curve, simple_regret, trials_to_reach
from src.reward_surfaces import RewardSurface

def _pad_or_truncate(curve: np.ndarray, T: int) -> np.ndarray:
    if len(curve) >= T:
        return curve[:T]
    out = np.empty(T)
    out[:len(curve)] = curve
    out[len(curve):] = curve[-1]   # pad with last value
    return out

def run_sweep(methods: list[str],
              surfaces: list[RewardSurface],
              n_seeds: int = 20,
              budget: int = 30,
              n_initial: int = 6,
              noise_std: float = 0.02,
              gp_noise_std: float = 0.03,
              n_candidates: int = 256,
              out_dir: str = "results/tables") -> dict:

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    histories = {}
    best_so_far_mean = {}
    best_so_far_lo = {}
    best_so_far_hi = {}
    regret_mean = {}
    regret_std = {}
    trials_to_reach_mean = {}

    # Convergence target: 80% of optimum value
    target_fraction = 0.80
    rows = []
    total = len(methods) * len(surfaces) * n_seeds
    done = 0

    for method in methods:
        for surf in surfaces:
            key = (method, surf.name)
            hist_list = []
            regrets = []
            tt_reach = []
            for seed in range(n_seeds):
                h = run_optimizer(surf, method, budget=budget, n_initial=n_initial,
                                  noise_std=noise_std, gp_noise_std=gp_noise_std,
                                  seed=seed, n_candidates=n_candidates)
                hist_list.append(h)
                regrets.append(simple_regret(h, surf))
                target = target_fraction * surf.optimum_value
                t = trials_to_reach(h, target)
                tt_reach.append(t if t is not None else budget + 1)
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"[{done}/{total}] {method} on {surf.name} seed {seed}")

            histories[key] = hist_list
            curves = np.stack([_pad_or_truncate(best_so_far_curve(h), budget)
                               for h in hist_list], axis=0)   # (n_seeds, budget)
            best_so_far_mean[key] = curves.mean(axis=0)
            best_so_far_lo[key] = np.percentile(curves, 2.5, axis=0)
            best_so_far_hi[key] = np.percentile(curves, 97.5, axis=0)
            regret_mean[key] = float(np.mean(regrets))
            regret_std[key] = float(np.std(regrets))
            trials_to_reach_mean[key] = float(np.mean(tt_reach))

            rows.append({
                "method": method,
                "surface": surf.name,
                "n_seeds": n_seeds,
                "budget": budget,
                "regret_mean": regret_mean[key],
                "regret_std": regret_std[key],
                "trials_to_80pct_mean": trials_to_reach_mean[key],
                "final_best_mean": float(best_so_far_mean[key][-1]),
            })

    # Writing CSV
    csv_path = out_path / "sweep_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    return {
        "histories": histories,
        "best_so_far_mean": best_so_far_mean,
        "best_so_far_lo": best_so_far_lo,
        "best_so_far_hi": best_so_far_hi,
        "regret_mean": regret_mean,
        "regret_std": regret_std,
        "trials_to_reach_mean": trials_to_reach_mean,
        "methods": methods,
        "surface_names": [s.name for s in surfaces],
        "n_seeds": n_seeds,
        "budget": budget,
    }
