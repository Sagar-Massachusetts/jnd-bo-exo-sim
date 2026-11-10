"""
demo_step9.py
-------------
Produces results/figures/fig9_tost_equivalence.png : a 4-panel figure showing
the TOST confidence interval vs. the JND equivalence bound for EI vs JND-EI,
on each of the four reward surfaces.

Reading the plot
----------------
Each panel is one surface (unimodal, multimodal, asymmetric, timing_sensitive).
For each parameter (rise_time, peak_timing, fall_time, peak_torque), we show
a horizontal error bar: the 90% CI of the mean difference between EI and JND-EI.
Two green dashed lines mark ±JND — the perceptual equivalence band.

Interpretation
--------------
- If the CI is ENTIRELY INSIDE ±JND → equivalent → "perceptually the same"
- If the CI CROSSES a JND bound  → inconclusive at that sample size
- If the CI is ENTIRELY OUTSIDE ±JND → not equivalent → "perceptually different"

We expect: on symmetric surfaces (unimodal, asymmetric, multimodal),
all 4 parameters are equivalent. On timing_sensitive, at least one
timing parameter is not equivalent — exactly the perceptual asymmetry.

Run:
    python demo_step9.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reward_surfaces import all_surfaces
from src.sweep import run_sweep
from src.tost import best_params_from_history, tost_on_optima
from src.torque_profile import PARAM_JND, PARAM_NAMES


FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path("results/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _gather_optima_by_method(res, method):
    """
    For a given method, gather best_params from each seed for each surface.
    Returns dict: surface_name -> ndarray (n_seeds, 4)
    """
    out = {}
    for s in res["surface_names"]:
        hist_list = res["histories"][(method, s)]
        params_list = np.stack([best_params_from_history(h) for h in hist_list], axis=0)
        out[s] = params_list
    return out


def _plot_one_surface(ax, surface_name, tost_res, title_prefix=""):
    """Plot 4 horizontal error bars (one per parameter) with JND bounds."""
    params = tost_res["params"]
    y_pos = np.arange(4)
    means = np.array([p["mean_diff"] for p in params])
    lo = np.array([p["ci_lo"] for p in params])
    hi = np.array([p["ci_hi"] for p in params])
    xerr = np.stack([means - lo, hi - means], axis=0)

    # Colours: green if equivalent, red if not
    colors = ["tab:green" if p["equivalent"] else "tab:red" for p in params]
    for y, m, e, c in zip(y_pos, means, xerr.T, colors):
        ax.errorbar(m, y, xerr=[[e[0]], [e[1]]], fmt="o", color=c,
                    capsize=4, markersize=8, linewidth=2)

    # JND bounds
    for i, j in enumerate(PARAM_JND):
        ax.plot([-j, -j], [i - 0.3, i + 0.3], color="black", linestyle="--",
                alpha=0.6, linewidth=1)
        ax.plot([j, j], [i - 0.3, i + 0.3], color="black", linestyle="--",
                alpha=0.6, linewidth=1)

    ax.axvline(0, color="gray", alpha=0.3, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(PARAM_NAMES)
    ax.set_xlabel("mean difference (EI - JND-EI)")
    ax.set_title(f"{title_prefix}{surface_name}  (all eq = {tost_res['all_equivalent']})")
    ax.grid(True, alpha=0.3, axis="x")


def make_fig9():
    surfaces = all_surfaces()
    print("Running sweep for TOST analysis (10 seeds, budget 20) — ~2-3 min ...")
    res = run_sweep(
        methods=["ei", "jnd_ei"],
        surfaces=surfaces,
        n_seeds=10,
        budget=20,
        n_initial=5,
        noise_std=0.02,
        gp_noise_std=0.03,
        n_candidates=256,
    )

    ei_opt = _gather_optima_by_method(res, "ei")
    jnd_opt = _gather_optima_by_method(res, "jnd_ei")

    tost_by_surface = {}
    for s in res["surface_names"]:
        tost_by_surface[s] = tost_on_optima(ei_opt[s], jnd_opt[s])

    # Save JSON summary
    out_json = LOGS_DIR / "tost_summary.json"
    with open(out_json, "w") as f:
        json.dump(
            {s: {"all_equivalent": r["all_equivalent"],
                 "worst_p": r["worst_p"],
                 "params": [{"name": p["param_name"],
                             "mean_diff": p["mean_diff"],
                             "ci_lo": p["ci_lo"],
                             "ci_hi": p["ci_hi"],
                             "bound": p["bound"],
                             "equivalent": p["equivalent"]}
                            for p in r["params"]]}
             for s, r in tost_by_surface.items()},
            f, indent=2,
        )
    print(f"wrote {out_json}")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
    axes = axes.ravel()
    for ax, s in zip(axes, res["surface_names"]):
        _plot_one_surface(ax, s, tost_by_surface[s])
    fig.suptitle("Step 9: TOST equivalence of EI vs JND-EI optima "
                 "(green = equivalent within JND)")
    out = FIG_DIR / "fig9_tost_equivalence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")

    print()
    print("Summary:")
    for s, r in tost_by_surface.items():
        print(f"  {s:20s}  all_equivalent={r['all_equivalent']}  worst_p={r['worst_p']:.4f}")


if __name__ == "__main__":
    make_fig9()
