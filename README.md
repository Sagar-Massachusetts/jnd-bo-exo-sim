# jnd-bo-exo-sim

Simulation testbed for **JND-informed Bayesian optimization of ankle-exoskeleton assistance profiles**. Final project for ME 556 (Advanced Robotics), Penn State, Fall 2026.

## Overview

Human-in-the-loop (HIL) optimization is the dominant approach for tuning exoskeleton assistance profiles. Yet the human reward surface is **perceptually asymmetric** — the human's sensitivity to a parameter change depends on the JND of that parameter, which varies across the parameter space. Standard Bayesian optimization ignores this.

This repository implements a **perception-aware Bayesian optimization** framework that:

- Models the exoskeleton torque profile with **four parameters** (rise time, peak timing, fall time, peak torque).
- Uses **JND-informed bounds** to define "perceptually distinguishable" in parameter space.
- Penalizes acquisition candidates that are perceptually indistinguishable from the current best — trading trivial re-queries for exploratory ones.
- Compares **five acquisition strategies** (random, grid, EI, Thompson, JND-EI) on **four synthetic reward surfaces** with known optima.
- Reports **simple regret**, **trials-to-80%**, and **TOST equivalence** of the optima found by each method.

## Key results

- EI and JND-EI reduce simple regret by **4–7×** compared with random and grid baselines on **three of four surfaces**.
- **TOST equivalence** (alpha = 0.05, JND bounds) declares EI and JND-EI **equivalent** on the unimodal surface (p = 0.040).
- On the **asymmetric** and **timing-sensitive** surfaces, TOST p-values are **near threshold** (0.070 and 0.099), consistent with the hypothesis that perception-aware acquisition produces a **slightly different optimum** when timing dominates the reward.
- The **multimodal** surface is inconclusive under TOST because both methods trap in equal-height basins.

## Repository layout
src/ Python implementation
walker.py planar walker (deterministic gait phase)
torque_profile.py 4-parameter ankle torque profile
reward_surfaces.py four synthetic reward surfaces with known optima
human_oracle.py simulated human (scalar, preference, JND-aware, ordinal)
gp_model.py GP surrogate (Matern + White kernel)
acquisitions.py five acquisition strategies
experiment.py optimizer loop
metrics.py regret, best-so-far, trials-to-80%
sweep.py multi-seed sweep aggregator
tost.py TOST equivalence with JND-informed bounds
tests/ pytest test suite (66 tests, all pass)
configs/ YAML configs
results/
figures/ all publication figures (fig1 to fig10)
tables/ sweep summary CSV (20-seed and 10-seed)
logs/ TOST summary JSON
docs/ project log
run_experiments.py CLI runner for sweeps
demo_step2.py ... demo_step10.py one script per figure
report.md 3-page project report

text

## Reproducing the results

```bash
git clone https://github.com/Sagar-Massachusetts/jnd-bo-exo-sim.git
cd jnd-bo-exo-sim
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q tests/                    # 66 tests, all pass
python demo_step10.py               # full sweep + master summary figure
To reproduce any single step's figures:

bash
python demo_step2.py     # fig1 (torque profile), fig2 (gait phase)
python demo_step3.py     # fig3 (four reward surfaces)
python demo_step4.py     # fig4 (human oracle: Bradley-Terry + JND zone)
python demo_step5.py     # fig5 (GP surrogate)
python demo_step6.py     # fig6a/6b (convergence + GP evolution)
python demo_step7.py     # fig7a/7b (multi-seed sweep + bands)
python demo_step8.py     # quick CLI runner (figures in results/run_quick/)
python demo_step9.py     # fig9 (TOST equivalence)
python demo_step10.py    # fig10 (master summary) + full report data
Key technical details
Torque profile
tau(phi; theta) where theta = [rise_time, peak_timing, fall_time, peak_torque]. Rise and fall are piecewise-linear; the peak is a Gaussian bump. Bounds: rise_time in [0.10, 0.40], peak_timing in [0.30, 0.60], fall_time in [0.05, 0.20], peak_torque in [0.10, 1.00] N m / kg.

JND values
Provisional, based on the lab's prior work (Maberry and Martin 2024; Peng et al. 2022): 3% of stride for the three timing parameters, 10% for peak magnitude. These can be updated with the lab's actual JND measurements without changing any code.

Acquisition functions
Name	Description
random	Uniform random sample
grid	6^4 grid, cyclic
ei	Expected Improvement
ts	Thompson Sampling
jnd_ei	JND-aware EI (candidates within JND of best are penalized by 0.05x)
Simulated human oracle
Four modes: scalar (Gaussian noise), pairwise preference (Bradley-Terry), JND-aware preference (returns "tie" when params within JND), and ordinal rating (1 to 5 Likert). All modes accept an rng for reproducibility.

TOST equivalence
Two one-sided tests with Welch's t-distribution and JND-informed bounds. Output includes per-parameter CI, p-values, and an all_equivalent flag.

Status / Submission
Submitted as the ME 556 final project. All deliverables are on GitHub. Public repo, no private data, no secrets.

Final deliverables checklist:

☑ Public GitHub repo with README
☑ Reproducible Python environment (requirements.txt)
☑ Full test suite (pytest -q tests/ gives 66 passed)
☑ 10 publication figures in results/figures/
☑ Machine-readable results (results/tables/sweep_summary.csv, results/logs/tost_summary.json)
☑ 3-page report (report.md)
□ Optional short demo video (embedded in Canvas)
Citation
If you use this code, please cite:

text
@misc{sagar2026jndbo,
  title  = {JND-informed Bayesian optimization of ankle-exoskeleton assistance profiles},
  author = {Sagar, Md Mainuddin},
  year   = {2026},
  note   = {ME 556 Final Project, Penn State},
  url    = {https://github.com/Sagar-Massachusetts/jnd-bo-exo-sim}
}
License
MIT, for academic and research use.
