"""
Md Mainuddin Sagar-
-
demo_step7.py runs a 5-method × 4-surface × 20-seed sweep and produces two figures:
  fig7a_convergence_bands.png : best-so-far mean + 95% CI for all 5 methods
                                 on all 4 surfaces (2x2 grid).
  fig7b_summary_bars.png      : simple regret per method, 4 groups (one per surface).
Also writes results/tables/sweep_summary.csv.

Now the main run:
    python demo_step7.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.reward_surfaces import all_surfaces
from src.sweep import run_sweep

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["random", "grid", "ei", "ts", "jnd_ei"]
METHOD_LABEL = {
    "random": "Random",
    "grid": "Grid",
    "ei": "EI",
    "ts": "Thompson",
    "jnd_ei": "JND-EI",
}
METHOD_COLOR = {
    "random": "gray",
    "grid": "tab:brown",
    "ei": "tab:blue",
    "ts": "tab:green",
    "jnd_ei": "tab:red",
}

def make_fig7a(res: dict):
    surfaces = res["surface_names"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    axes = axes.ravel()

    budget = res["budget"]
    t = np.arange(1, budget + 1)

    for ax, s in zip(axes, surfaces):
        for m in METHODS:
            key = (m, s)
            mean = res["best_so_far_mean"][key]
            lo = res["best_so_far_lo"][key]
            hi = res["best_so_far_hi"][key]
            ax.plot(t, mean, label=METHOD_LABEL[m], color=METHOD_COLOR[m], linewidth=2)
            ax.fill_between(t, lo, hi, color=METHOD_COLOR[m], alpha=0.15)
        ax.set_title(f"surface: {s}")
        ax.set_xlabel("iteration")
        ax.set_ylabel("best true reward")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Step 7a: best-so-far curves (mean ± 95% CI across 20 seeds)")
    out = FIG_DIR / "fig7a_convergence_bands.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")

def make_fig7b(res: dict):
    surfaces = res["surface_names"]
    n_methods = len(METHODS)
    x = np.arange(len(surfaces))
    width = 0.16

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for i, m in enumerate(METHODS):
        means = [res["regret_mean"][(m, s)] for s in surfaces]
        stds = [res["regret_std"][(m, s)] for s in surfaces]
        ax.bar(x + (i - n_methods / 2 + 0.5) * width, means, width,
               yerr=stds, capsize=2, label=METHOD_LABEL[m], color=METHOD_COLOR[m])
    ax.set_xticks(x)
    ax.set_xticklabels(surfaces)
    ax.set_ylabel("simple regret (lower is better)")
    ax.set_title("Step 7b: simple regret across methods and surfaces (mean ± std over 20 seeds)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    out = FIG_DIR / "fig7b_summary_bars.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")

if __name__ == "__main__":
    surfaces = all_surfaces()
    print("Running sweep: 5 methods × 4 surfaces × 20 seeds, budget=30")
    res = run_sweep(
        methods=METHODS,
        surfaces=surfaces,
        n_seeds=20,
        budget=30,
        n_initial=6,
        noise_std=0.02,
        gp_noise_std=0.03,
        n_candidates=256,
    )
    make_fig7a(res)
    make_fig7b(res)
