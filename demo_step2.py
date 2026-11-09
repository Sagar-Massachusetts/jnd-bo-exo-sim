"""
demo_step2.py
-------------
Produces two figures into results/figures/:

  fig1_torque_profile.png  : torque vs gait phase for 4 parameter sets, with JND bands.
  fig2_walker_phase.png    : gait phase over 3 strides.

Run:
    python demo_step2.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.torque_profile import (
    PARAM_BOUNDS, PARAM_JND, PARAM_NAMES, torque_profile_array
)
from src.walker import Walker


FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def make_fig1():
    phases = np.linspace(0.0, 1.0, 501)
    mass = 70.0

    param_sets = [
        ("low magnitude",   np.array([0.20, 0.45, 0.10, 0.25])),
        ("mid magnitude",   np.array([0.20, 0.45, 0.10, 0.50])),
        ("high magnitude",  np.array([0.20, 0.45, 0.10, 0.85])),
        ("later peak",      np.array([0.20, 0.55, 0.10, 0.50])),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, params in param_sets:
        tau = torque_profile_array(phases, params, body_mass_kg=mass)
        ax.plot(phases, tau, label=label)

    ax.set_xlabel("Gait phase (fraction of stride)")
    ax.set_ylabel(f"Ankle torque (N*m), body mass {mass} kg")
    ax.set_title("4-parameter ankle torque profile")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "fig1_torque_profile.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def make_fig2():
    w = Walker(stride_seconds=1.0, phase0=0.0)
    dt = 0.01
    t = np.arange(0.0, 3.0, dt)
    ph = np.array([w.gait_phase_at(ti) for ti in t])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(t, ph, color="C0", linewidth=2)
    for k in range(4):
        ax.axvline(k, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gait phase")
    ax.set_title("Walker gait phase over 3 strides (period = 1.0 s)")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig2_walker_phase.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_fig1()
    make_fig2()
