"""
Md Mainuddin Sagar-
-
demo_step5.py produces results/figures/fig5_gp_surrogate.png : a four-panel figure showing
how the GP posterior evolves as more training data is observed, on a 2D slice
(peak_timing, peak_torque) of the unimodal reward surface.

Panels (top to bottom):
  1. True reward on the 2D slice
  2. GP posterior mean with N=8 training points
  3. GP posterior mean with N=30 training points
  4. GP posterior std with N=30 training points

Now the main run:
    python demo_step5.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.gp_model import GPModel
from src.reward_surfaces import all_surfaces
from src.torque_profile import PARAM_BOUNDS, PARAM_NAMES

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def _slice_grid(n=60):
    pt = np.linspace(PARAM_BOUNDS[1, 0], PARAM_BOUNDS[1, 1], n)
    pq = np.linspace(PARAM_BOUNDS[3, 0], PARAM_BOUNDS[3, 1], n)
    PT, PQ = np.meshgrid(pt, pq, indexing="ij")
    return pt, pq, PT, PQ

def _to_full_params(surface, PT, PQ):
    opt = surface.optimum
    P = np.zeros(PT.shape + (4,))
    P[..., 0] = opt[0]
    P[..., 1] = PT
    P[..., 2] = opt[2]
    P[..., 3] = PQ
    return P

def make_fig5():
    surface = all_surfaces()[0]
    pt, pq, PT, PQ = _slice_grid(n=60)
    P_grid = _to_full_params(surface, PT, PQ)
    R_true = np.zeros(PT.shape)
    for i in range(PT.shape[0]):
        for j in range(PT.shape[1]):
            R_true[i, j] = surface.f(P_grid[i, j])

    # Sample training points on the 2D slice 
    rng = np.random.default_rng(7)

    def _sample_train(n):
        pt_s = rng.uniform(pt[0], pt[-1], n)
        pq_s = rng.uniform(pq[0], pq[-1], n)
        P = np.zeros((n, 4))
        opt = surface.optimum
        P[:, 0] = opt[0]
        P[:, 1] = pt_s
        P[:, 2] = opt[2]
        P[:, 3] = pq_s
        y = np.array([surface.f(p) for p in P])
        return P, y

    P8, y8 = _sample_train(8)
    P30, y30 = _sample_train(30)

    # Fitting GPs
    gp8 = GPModel(noise_std=0.02, seed=0).fit(P8, y8)
    gp30 = GPModel(noise_std=0.02, seed=0).fit(P30, y30)

    # Predicting on the slice
    P_flat = P_grid.reshape(-1, 4)
    mu8, _ = gp8.predict(P_flat)
    mu30, s30 = gp30.predict(P_flat)
    mu8 = mu8.reshape(PT.shape)
    mu30 = mu30.reshape(PT.shape)
    s30 = s30.reshape(PT.shape)

    # Plotting figures
    fig, axes = plt.subplots(4, 1, figsize=(6.5, 14), constrained_layout=True)
    extent = [pq[0], pq[-1], pt[0], pt[-1]]

    def _show(ax, R, title, cmap="viridis", vmin=0, vmax=1, points=None):
        im = ax.imshow(R, origin="lower", extent=extent, aspect="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel(f"{PARAM_NAMES[3]} (N*m/kg)")
        ax.set_ylabel(f"{PARAM_NAMES[1]} (fraction)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.7)
        if points is not None:
            ax.scatter(points[:, 3], points[:, 1], c="white",
                       edgecolors="black", s=30, zorder=5)

    _show(axes[0], R_true, "True reward")
    _show(axes[1], mu8, "GP posterior mean (N=8 training points)", points=P8)
    _show(axes[2], mu30, "GP posterior mean (N=30 training points)", points=P30)
    _show(axes[3], s30, "GP posterior std (N=30 training points)", cmap="magma")

    fig.suptitle("Step 5: GP surrogate mean and uncertainty on the (peak_timing, peak_torque) slice")
    out = FIG_DIR / "fig5_gp_surrogate.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_fig5()
