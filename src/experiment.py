"""
experiment.py
-------------
Main optimizer loop for the HIL experiment.

Given an acquisition function, a reward surface, and a budget, run a
sequential optimization and log every step.

Public API
----------
    run_optimizer(surface, acquire_fn, budget, n_initial, noise_std,
                  gp_noise_std, seed, jnd=None, **acq_kwargs)
        -> dict with keys:
            'params'    : list of np.ndarray (one per iteration)
            'rewards'   : list of float (noisy observed reward)
            'true_rewards' : list of float (noiseless reward for reference)
            'best_so_far': list of float (running best of noisy observed)
            'gp_mean_at_optimum' : list of float (posterior mean at true optimum)
            'gp_std_at_optimum'  : list of float (posterior std at true optimum)
            'gp_mean_at_best'    : list of float (posterior mean at best point)
"""
from __future__ import annotations

import numpy as np

from src.acquisitions import (
    expected_improvement,
    grid_acquire,
    jnd_aware_expected_improvement,
    random_acquire,
    thompson_sampling,
)
from src.gp_model import GPModel
from src.human_oracle import simulate_scalar_response
from src.reward_surfaces import RewardSurface
from src.torque_profile import PARAM_BOUNDS


def _run_one_step(surface, acquire_name, gp, X, y, best_so_far, best_params,
                  rng, acq_kwargs):
    """Dispatch to the right acquisition function."""
    if acquire_name == "random":
        return random_acquire(surface, gp, X, rng=rng)
    if acquire_name == "grid":
        return grid_acquire(surface, gp, X,
                            grid_resolution=acq_kwargs.get("grid_resolution", 6),
                            rng=rng)
    if acquire_name == "ei":
        return expected_improvement(surface, gp, X,
                                    y_seen=y, best_so_far=best_so_far,
                                    n_candidates=acq_kwargs.get("n_candidates", 2048),
                                    xi=acq_kwargs.get("xi", 0.01),
                                    rng=rng)
    if acquire_name == "ts":
        return thompson_sampling(surface, gp, X,
                                 n_candidates=acq_kwargs.get("n_candidates", 1024),
                                 n_samples=1, rng=rng)
    if acquire_name == "jnd_ei":
        return jnd_aware_expected_improvement(
            surface, gp, X,
            y_seen=y, best_so_far=best_so_far, best_params=best_params,
            jnd=acq_kwargs.get("jnd", None),
            n_candidates=acq_kwargs.get("n_candidates", 2048),
            xi=acq_kwargs.get("xi", 0.01),
            rng=rng,
        )
    raise ValueError(f"Unknown acquisition name: {acquire_name}")


def run_optimizer(surface: RewardSurface,
                  acquire_name: str,
                  budget: int = 30,
                  n_initial: int = 6,
                  noise_std: float = 0.02,
                  gp_noise_std: float = 0.03,
                  seed: int = 0,
                  jnd: np.ndarray | None = None,
                  **acq_kwargs) -> dict:
    """
    Run one optimization experiment on a reward surface.

    Parameters
    ----------
    surface : RewardSurface
    acquire_name : {"random", "grid", "ei", "ts", "jnd_ei"}
    budget : int
        Total number of reward evaluations (including n_initial).
    n_initial : int
        Random initial points evaluated before the acquisition starts.
    noise_std : float
        Standard deviation of the simulated human's scalar response.
    gp_noise_std : float
        GP observation noise (should be ~= noise_std).
    seed : int
    jnd : np.ndarray | None
        JND vector; only used by jnd_ei.
    **acq_kwargs :
        Forwarded to the acquisition function.
    """
    if n_initial >= budget:
        raise ValueError("n_initial must be < budget")
    rng = np.random.default_rng(seed)

    # --- Initial random sampling -----------------------------------------
    X = []
    y = []
    for _ in range(n_initial):
        u = rng.uniform(0, 1, size=4)
        p = PARAM_BOUNDS[:, 0] + u * (PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0])
        obs = simulate_scalar_response(surface, p, noise_std=noise_std, rng=rng)
        X.append(p)
        y.append(obs)

    # Logs
    params_log = list(X)
    rewards_log = list(y)
    true_rewards_log = [float(surface.f(p)) for p in X]
    best_so_far = [float(np.max(y))]
    gp_mean_at_opt = []
    gp_std_at_opt = []
    gp_mean_at_best = []

    for it in range(budget - n_initial):
        # Fit GP on all data so far
        gp = GPModel(noise_std=gp_noise_std, seed=int(rng.integers(0, 2**31 - 1)))
        gp.fit(np.array(X), np.array(y))

        # Log GP at the true optimum
        m_opt, s_opt = gp.predict(surface.optimum[None, :])
        gp_mean_at_opt.append(float(m_opt[0]))
        gp_std_at_opt.append(float(s_opt[0]))

        best_idx = int(np.argmax(y))
        gp_mean_at_best.append(float(gp.predict(np.array(X)[best_idx][None, :])[0][0]))

        best_params = np.array(X[best_idx])

        # Pick next query
        p_next = _run_one_step(
            surface, acquire_name, gp, np.array(X), np.array(y),
            best_so_far=float(np.max(y)), best_params=best_params,
            rng=rng, acq_kwargs={**acq_kwargs, "jnd": jnd},
        )

        # Evaluate on the simulated human
        obs = simulate_scalar_response(surface, p_next, noise_std=noise_std, rng=rng)
        X.append(p_next)
        y.append(obs)
        true_rewards_log.append(float(surface.f(p_next)))
        best_so_far.append(float(np.max(y)))
        params_log.append(p_next)
        rewards_log.append(obs)

    return {
        "params": params_log,
        "rewards": rewards_log,
        "true_rewards": true_rewards_log,
        "best_so_far": best_so_far,
        "gp_mean_at_optimum": gp_mean_at_opt,
        "gp_std_at_optimum": gp_std_at_opt,
        "gp_mean_at_best": gp_mean_at_best,
        "acquire_name": acquire_name,
        "surface_name": surface.name,
        "seed": seed,
        "budget": budget,
        "n_initial": n_initial,
    }
