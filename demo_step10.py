"""
Md Mainuddin Sagar-
-
demo_step10.py produces results/figures/fig10_master_summary.png : a 6-panel summary of the
entire project, suitable for a paper figure or thesis chapter.

Panels
1. Convergence bands       : EI vs JND-EI vs random on unimodal
2. Simple regret bar chart : all 5 methods × 4 surfaces
3. TOST equivalence        : summary (from Step 9)
4. GP mean at optimum      : convergence of the surrogate
5. Trials to 80%           : bar chart
6. Overall method winner   : a text panel summarizing who won where

Now the main run:
    python demo_step10.py
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.metrics import best_so_far_curve, simple_regret, trials_to_reach
from src.reward_surfaces import all_surfaces
from src.sweep import run_sweep

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path("results/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["random", "grid", "ei", "ts", "jnd_ei"]
METHOD_LABEL = {"random": "Random", "grid": "Grid", "ei": "EI",
                "ts": "Thompson", "jnd_ei": "JND-EI"}
METHOD_COLOR = {"random": "gray", "grid": "tab:brown",
                "ei": "tab:blue", "ts": "tab:green", "jnd_ei": "tab:red"}

def _pad_or_truncate(curve, T):
    curve = np.asarray(curve, dtype=float)
    if len(curve) >= T:
        return curve[:T]
    out = np.empty(T)
    out[:len(curve)] = curve
    out[len(curve):] = curve[-1]
    return out

def make_fig10():
    surfaces = all_surfaces()
    surface_names = [s.name for s in surfaces]

    print("Running full sweep for Step 10 (10 seeds, budget 25) — ~5 min ...")
    res = run_sweep(
        methods=METHODS, surfaces=surfaces, n_seeds=10, budget=25,
        n_initial=5, noise_std=0.02, gp_noise_std=0.03, n_candidates=256,
    )

    # Loading TOST from Step 9 (10 seeds)
    tost_path = LOGS_DIR / "tost_summary.json"
    if tost_path.exists():
        tost = json.load(open(tost_path))
    else:
        tost = {}

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])

    # Panel 1: convergence bands on unimodal
    ax1 = fig.add_subplot(gs[0, 0])
    budget = res["budget"]
    t = np.arange(1, budget + 1)
    for m in ["random", "ei", "jnd_ei"]:
        mean = res["best_so_far_mean"][(m, "unimodal")]
        lo = res["best_so_far_lo"][(m, "unimodal")]
        hi = res["best_so_far_hi"][(m, "unimodal")]
        ax1.plot(t, mean, label=METHOD_LABEL[m], color=METHOD_COLOR[m], lw=2)
        ax1.fill_between(t, lo, hi, color=METHOD_COLOR[m], alpha=0.15)
    ax1.axhline(1.0, linestyle="--", color="black", alpha=0.4)
    ax1.set_title("(1) Convergence bands (unimodal)")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("best true reward")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: regret bar chart (all surfaces)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(surface_names))
    width = 0.16
    for i, m in enumerate(METHODS):
        means = [res["regret_mean"][(m, s)] for s in surface_names]
        stds = [res["regret_std"][(m, s)] for s in surface_names]
        ax2.bar(x + (i - 2) * width, means, width, yerr=stds, capsize=2,
                label=METHOD_LABEL[m], color=METHOD_COLOR[m])
    ax2.set_xticks(x)
    ax2.set_xticklabels(surface_names, rotation=15, fontsize=8)
    ax2.set_ylabel("simple regret (lower is better)")
    ax2.set_title("(2) Simple regret across surfaces")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: TOST equivalence summary
    ax3 = fig.add_subplot(gs[0, 2])
    if tost:
        s_names = list(tost.keys())
        eq = [1 if tost[s]["all_equivalent"] else 0 for s in s_names]
        worst_p = [tost[s]["worst_p"] for s in s_names]
        colors = ["tab:green" if e else "tab:red" for e in eq]
        bars = ax3.bar(s_names, eq, color=colors)
        for bar, p in zip(bars, worst_p):
            ax3.text(bar.get_x() + bar.get_width() / 2, 0.5,
                     f"p={p:.3f}", ha="center", va="center", fontsize=8,
                     color="white", fontweight="bold")
        ax3.set_ylim(0, 1.2)
        ax3.set_ylabel("1 = equivalent")
        ax3.set_title("(3) TOST equivalence (EI vs JND-EI)")
        ax3.tick_params(axis="x", rotation=15, labelsize=8)
    else:
        ax3.text(0.5, 0.5, "TOST not available", ha="center", va="center",
                 transform=ax3.transAxes)
        ax3.set_title("(3) TOST equivalence")

    # Panel 4: GP convergence at optimum
    ax4 = fig.add_subplot(gs[1, 0])
    for m in ["ei", "jnd_ei"]:
        means = []
        for h in res["histories"][(m, "unimodal")]:
            gm = h.get("gp_mean_at_optimum", [])
            means.append(_pad_or_truncate(gm, budget))
        mm = np.mean(np.stack(means, axis=0), axis=0)
        ax4.plot(np.arange(1, len(mm) + 1), mm, label=METHOD_LABEL[m],
                 color=METHOD_COLOR[m], lw=2)
    ax4.axhline(1.0, linestyle="--", color="black", alpha=0.4)
    ax4.set_title("(4) GP posterior mean at the true optimum")
    ax4.set_xlabel("iteration")
    ax4.set_ylabel("posterior mean")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Panel 5: trials to 80% (bar chart)
    ax5 = fig.add_subplot(gs[1, 1])
    for i, m in enumerate(METHODS):
        means = [res["trials_to_reach_mean"][(m, s)] for s in surface_names]
        ax5.bar(x + (i - 2) * width, means, width,
                label=METHOD_LABEL[m], color=METHOD_COLOR[m])
    ax5.set_xticks(x)
    ax5.set_xticklabels(surface_names, rotation=15, fontsize=8)
    ax5.set_ylabel("trials to 80% of optimum (lower is better)")
    ax5.set_title("(5) Trials to 80% target reward")
    ax5.grid(True, alpha=0.3, axis="y")

    # Panel 6: overall winner summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    # computing winner per surface
    lines = ["Overall winner per surface:", ""]
    for s in surface_names:
        regrets = {m: res["regret_mean"][(m, s)] for m in METHODS}
        winner = min(regrets, key=regrets.get)
        lines.append(f"  {s:18s} → {METHOD_LABEL[winner]:>10s}  "
                     f"(regret={regrets[winner]:.3f})")
    # texting summary of Step 7/9 key insight
    lines.append("")
    lines.append("Key result:")
    lines.append("  Bayesian optimization (EI/JND-EI)")
    lines.append("  beats random/grid on 3/4 surfaces.")
    lines.append("  JND-EI ≈ EI on symmetric surfaces")
    lines.append("  (TOST-equivalent on unimodal).")
    ax6.text(0.02, 0.98, "\n".join(lines), ha="left", va="top",
             fontsize=9, family="monospace",
             transform=ax6.transAxes,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))

    fig.suptitle("Step 10: Master summary — JND-informed Bayesian optimization for ankle-exoskeleton tuning",
                 fontsize=13, fontweight="bold")
    out = FIG_DIR / "fig10_master_summary.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_fig10()
