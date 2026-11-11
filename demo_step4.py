"""
Md Mainuddin Sagar-
-
demo_step4.py produces results/figures/fig4_human_oracle.png : a two-panel figure showing

  Left panel : Bradley-Terry preference probability
               P(A > B) = sigmoid(beta * (f(A) - f(B)))
               as a function of the reward difference f(A) - f(B).
  Right panel : JND-aware tie zone on a (peak_timing, peak_torque) slice.
               For each grid point, we ask: "is this grid point perceptually
               distinguishable from the optimum, given PARAM_JND?" It plots a
               binary map: 1 = tie zone (imperceptible), 0 = decisive.

Now the main run:
    python demo_step4.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.human_oracle import _normalized_distance
from src.reward_surfaces import all_surfaces
from src.torque_profile import PARAM_BOUNDS, PARAM_JND, PARAM_NAMES

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def make_fig4():
    surface = all_surfaces()[0]   # unimodal

    # Left panel: Bradley-Terry preference probability 
    deltas = np.linspace(-1.0, 1.0, 201)
    betas = [1.0, 5.0, 20.0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax = axes[0]
    for beta in betas:
        p = 1.0 / (1.0 + np.exp(-beta * deltas))
        ax.plot(deltas, p, label=f"beta = {beta:g}")
    ax.set_xlabel("reward difference  f(A) - f(B)")
    ax.set_ylabel("P(A preferred over B)")
    ax.set_title("Bradley-Terry preference probability")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right panel: JND-aware tie zone on a 2D slice 
    opt = surface.optimum
    pt_grid = np.linspace(PARAM_BOUNDS[1, 0], PARAM_BOUNDS[1, 1], 120)
    pq_grid = np.linspace(PARAM_BOUNDS[3, 0], PARAM_BOUNDS[3, 1], 120)
    PT, PQ = np.meshgrid(pt_grid, pq_grid, indexing="ij")

    tie_zone = np.zeros_like(PT)
    for i in range(pt_grid.size):
        for j in range(pq_grid.size):
            p = opt.copy()
            p[1] = PT[i, j]
            p[3] = PQ[i, j]
            dist = _normalized_distance(p, opt)
            tie_zone[i, j] = 1.0 if np.all(dist < PARAM_JND) else 0.0

    ax = axes[1]
    ax.imshow(
        tie_zone, origin="lower",
        extent=[pq_grid[0], pq_grid[-1], pt_grid[0], pt_grid[1]],
        aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
    )
    ax.scatter([opt[3]], [opt[1]], marker="*", s=220, c="black",
               edgecolors="white", zorder=5, label="optimum")
    ax.set_xlabel(f"{PARAM_NAMES[3]} (N*m/kg)")
    ax.set_ylabel(f"{PARAM_NAMES[1]} (fraction)")
    ax.set_title("JND-aware tie zone (green = perceptually indistinguishable)")
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Step 4: simulated human oracle - preference + JND-aware comparison")
    out = FIG_DIR / "fig4_human_oracle.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_fig4()
