"""
demo_step6.py
-------------
Produces two figures:

  fig6a_convergence.png : best-so-far vs iteration for 5 methods on the
                          unimodal and multimodal surfaces.
  fig6b_gp_evolution.png : GP posterior mean at the true optimum across
                          iterations for EI vs JND-EI (shows convergence).

Run:
    python demo_step6.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.experiment import run_optimizer
from src.metrics import best_so_far_curve
from src.reward_surfaces import all_surfaces


FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def make_fig6a():
    methods = ["random", "grid", "ei", "ts", "jnd_ei"]
    colors = {"random": "gray", "grid": "tab:brown",
              "ei": "tab:blue", "ts": "tab:green", "jnd_ei": "tab:red"}

    surfaces = all_surfaces()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    for ax, surf in zip(axes, [surfaces[0], surfaces[1]]):  # unimodal, multimodal
        for method in methods:
            h = run_optimizer(surf, method, budget=25, n_initial=5,
                              seed=0, noise_std=0.02, gp_noise_std=0.03,
                              n_candidates=512)
            curve = best_so_far_curve(h)
            ax.plot(np.arange(1, len(curve) + 1), curve,
                    label=method, color=colors[method], linewidth=2)
        ax.axhline(surf.optimum_value, linestyle="--", color="black",
                   alpha=0.5, label="true optimum")
        ax.set_title(f"surface: {surf.name}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("best true reward so far")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Step 6a: best-so-far curves for 5 acquisition strategies")
    out = FIG_DIR / "fig6a_convergence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def make_fig6b():
    surfaces = all_surfaces()
    surf = surfaces[0]   # unimodal

    h_ei = run_optimizer(surf, "ei", budget=25, n_initial=5,
                         seed=0, noise_std=0.02, gp_noise_std=0.03,
                         n_candidates=512)
    h_jnd = run_optimizer(surf, "jnd_ei", budget=25, n_initial=5,
                          seed=0, noise_std=0.02, gp_noise_std=0.03,
                          n_candidates=512)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(h_ei["gp_mean_at_optimum"], label="EI", color="tab:blue", linewidth=2)
    ax.plot(h_jnd["gp_mean_at_optimum"], label="JND-EI", color="tab:red", linewidth=2)
    ax.axhline(surf.optimum_value, linestyle="--", color="black", alpha=0.5, label="true optimum")
    ax.set_xlabel("iteration")
    ax.set_ylabel("GP posterior mean at true optimum")
    ax.set_title("GP mean at the optimum (convergence of the surrogate)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(h_ei["gp_std_at_optimum"], label="EI", color="tab:blue", linewidth=2)
    ax.plot(h_jnd["gp_std_at_optimum"], label="JND-EI", color="tab:red", linewidth=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("GP posterior std at true optimum")
    ax.set_title("GP uncertainty at the optimum (should decrease)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Step 6b: GP surrogate evolution — EI vs JND-EI")
    out = FIG_DIR / "fig6b_gp_evolution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_fig6a()
    make_fig6b()
