"""
run_experiments.py
------------------
CLI runner for the JND-bo-exo-sim sweep.

Usage
-----
Quick smoke (default, ~30s, small budgets — for CI / sanity):
    python run_experiments.py --quick

Full sweep (uses ~20-30 minutes):
    python run_experiments.py --budget 30 --n-seeds 20

Custom out dir:
    python run_experiments.py --budget 30 --n-seeds 20 --out-dir results/run_big

What it writes
--------------
    <out_dir>/figures/fig7a_convergence_bands.png
    <out_dir>/figures/fig7b_summary_bars.png
    <out_dir>/tables/sweep_summary.csv
    <out_dir>/logs/full_histories.json     (only when --save-histories is set)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.reward_surfaces import all_surfaces
from src.sweep import run_sweep
import demo_step7 as demo7


def _to_jsonable(hist: dict) -> dict:
    """Convert a history dict with numpy arrays/lists into JSON-safe form."""
    out = {}
    for k, v in hist.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, list):
            out[k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _print_summary(res: dict):
    """Pretty-print a summary table to stdout."""
    methods = res["methods"]
    surfaces = res["surface_names"]
    print()
    print("=" * 100)
    print(f"Sweep summary  —  n_seeds={res['n_seeds']}, budget={res['budget']}")
    print("=" * 100)
    header = f"{'method':>10} | " + " | ".join(f"{s:>16}" for s in surfaces)
    print(header)
    print("-" * len(header))
    for m in methods:
        cells = []
        for s in surfaces:
            rm = res["regret_mean"][(m, s)]
            cells.append(f"{rm:16.4f}")
        print(f"{m:>10} | " + " | ".join(cells))
    print()
    print("Simple regret (lower is better). 'Best' method per column is bolded in the report.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=8,
                    help="Reward evaluations per run (default: 8 for quick mode)")
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="Number of seeds per (method, surface) pair (default: 3)")
    ap.add_argument("--n-initial", type=int, default=3,
                    help="Random initial points (default: 3)")
    ap.add_argument("--n-candidates", type=int, default=128,
                    help="BO acquisition candidate pool size (default: 128)")
    ap.add_argument("--noise-std", type=float, default=0.02,
                    help="Observation noise std for the simulated human (default: 0.02)")
    ap.add_argument("--gp-noise-std", type=float, default=0.03,
                    help="GP observation noise std (default: 0.03)")
    ap.add_argument("--out-dir", type=str, default="results",
                    help="Output root; writes figures/, tables/, logs/ under this")
    ap.add_argument("--quick", action="store_true",
                    help="Use tiny budgets for a fast smoke test")
    ap.add_argument("--save-histories", action="store_true",
                    help="Save full histories as JSON under <out_dir>/logs/")
    ap.add_argument("--methods", type=str,
                    default="random,grid,ei,ts,jnd_ei",
                    help="Comma-separated list of methods to compare")
    args = ap.parse_args()

    if args.quick:
        args.budget = 8
        args.n_seeds = 3
        args.n_initial = 3
        args.n_candidates = 128

    if args.n_initial >= args.budget:
        print("ERROR: n_initial must be < budget", file=sys.stderr)
        sys.exit(1)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    surfaces = all_surfaces()

    print(f"Running sweep: {len(methods)} methods × {len(surfaces)} surfaces × {args.n_seeds} seeds")
    print(f"budget={args.budget}, n_initial={args.n_initial}, n_candidates={args.n_candidates}")

    res = run_sweep(
        methods=methods,
        surfaces=surfaces,
        n_seeds=args.n_seeds,
        budget=args.budget,
        n_initial=args.n_initial,
        noise_std=args.noise_std,
        gp_noise_std=args.gp_noise_std,
        n_candidates=args.n_candidates,
        out_dir=str(Path(args.out_dir) / "tables"),
    )

    _print_summary(res)

    # Figures — reuse the demo_step7 machinery but with our own output dir
    demo7.FIG_DIR = Path(args.out_dir) / "figures"
    demo7.FIG_DIR.mkdir(parents=True, exist_ok=True)
    # renumber the output filenames so they don't clash with Step 7's
    orig_7a = demo7.make_fig7a
    orig_7b = demo7.make_fig7b
    try:
        demo7.make_fig7a(res)
        demo7.make_fig7b(res)
    except Exception as e:
        print(f"(figures skipped: {e})")

    if args.save_histories:
        log_dir = Path(args.out_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        out_path = log_dir / "full_histories.json"
        payload = {
            "meta": {
                "methods": methods,
                "surface_names": res["surface_names"],
                "n_seeds": res["n_seeds"],
                "budget": res["budget"],
                "n_initial": args.n_initial,
            },
            "histories": {
                f"{m}__{s}": [_to_jsonable(h) for h in hs]
                for (m, s), hs in res["histories"].items()
            },
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

