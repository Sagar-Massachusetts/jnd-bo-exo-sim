# jnd-bo-exo-sim

Simulation testbed for **JND-informed Bayesian optimization of ankle-exoskeleton assistance profiles**. Final project for ME 556 (Advanced Robotics), Penn State.

## What this project does
- Simulates a planar walker with a gait-phase-driven bilateral ankle exoskeleton.
- Encodes the exoskeleton torque profile with four parameters: rise time, peak timing, fall time, peak torque.
- Compares four optimization strategies (random, grid, EI, Thompson) against a proposed **JND-aware** acquisition.
- Evaluates perceptual equivalence of the resulting optima using TOST with JND-informed bounds.

## Layout
- `src/` — source modules.
- `configs/` — YAML configs.
- `notebooks/` — Colab notebooks.
- `results/` — figures, tables, logs.
- `tests/` — smoke tests.
- `docs/` — project log.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Status
- [x] Step 0: Proposal
- [x] Step 1: Repo + environment
- [ ] Step 2: Walker + torque profile
- [ ] Step 3: Reward surfaces
- [ ] Step 4: Simulated human oracle
- [ ] Step 5: GP preference model
- [ ] Step 6: Baselines (random, grid, EI, Thompson)
- [ ] Step 7: JND-aware acquisition
- [ ] Step 8: Experiment loop + metrics
- [ ] Step 9: Full simulation sweep
- [ ] Step 10: TOST + final figures
- [ ] Step 11: Report + README + optional video
