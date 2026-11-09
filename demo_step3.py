"""
demo_step3.py
-------------
Produces results/figures/fig3_reward_surfaces.png : a 2x2 panel of the four
synthetic reward surfaces, sliced over (peak_timing, peak_torque) with the
other two parameters fixed at the surface's own optimum.

Run:
    python demo_step3.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.reward_surfaces import all_surfaces
from src.torque_profile import PARAM_BOUNDS, PARAM_NAMES


FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def make_fig3():
    surfaces = all_surfaces()
    pt_grid = np.linspace(PARAM_BOUNDS[1, 0], PARAM_BOUNDS[1, 1], 120)
    pq_grid = np.linspace(PARAM_BOUNDS[3, 0], PARAM_BOUNDS[3, 1], 120)
    PT, PQ = np.meshgrid(pt_grid, pq_grid, indexing="ij")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)
    axes = axes.ravel()

    for ax, surf in zip(axes, surfaces):
        opt = surf.optimum.copy()
        R = np.zeros((pt_grid.size, pq_grid.size))
        for i in range(pt_grid.size):
            for j in range(pq_grid.size):
                p = opt.copy()
                p[1] = PT[i, j]
                p[3] = PQ[i, j]
                R[i, j] = surf.f(p)

        im = ax.imshow(
            R,
            origin="lower",
            extent=[pq_grid[0], pq_grid[-1], pt_grid[0], pt_grid[-1]],
            aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0,
        )
        ax.scatter([opt[3]], [opt[1]], marker="*", s=220, c="white",
                   edgecolors="black", zorder=5, label="ground-truth optimum")
        ax.set_xlabel(f"{PARAM_NAMES[3]} (N*m/kg)")
        ax.set_ylabel(f"{PARAM_NAMES[1]} (fraction)")
        ax.set_title(surf.name)
        ax.legend(loc="upper right", fontsize=8)

    fig.colorbar(im, ax=axes, shrink=0.85, label="reward (0 to 1)")
    fig.suptitle("Step 3: four synthetic reward surfaces (peak_timing vs peak_torque)")
    out = FIG_DIR / "fig3_reward_surfaces.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_fig3()
