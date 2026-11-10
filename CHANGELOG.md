Changelog
All notable changes to jnd-bo-exo-sim are documented here.

[v1.0] — 2026-11-09 — Final project submission
Added
Full simulation testbed for JND-informed Bayesian optimization of an ankle-exoskeleton torque profile.

src/ modules: walker, torque profile, four reward surfaces, simulated human oracle (four feedback modes), Gaussian-process surrogate, five acquisition strategies, optimizer loop, metrics, multi-seed sweep, TOST equivalence with JND-informed bounds.

66 passing tests in tests/, covering every module.

Ten publication figures in results/figures/ (fig1 through fig10).

results/tables/sweep_summary.csv (20-seed full sweep) and sweep_summary_10seed.csv (10-seed for TOST).

results/logs/tost_summary.json — per-parameter TOST results.

report.md — three-page write-up with methodology, results, discussion, and limitations.

README.md — publication-quality README with abstract, reproducibility instructions, and citation.

run_experiments.py — CLI runner for the sweep.

demo_step2.py through demo_step10.py — one script per figure.

Results at v1.0
EI and JND-EI reduce simple regret by 4-7x vs random/grid on 3 of 4 surfaces.

TOST equivalence declares EI and JND-EI equivalent on the unimodal surface (p = 0.040).

Asymmetric and timing-sensitive surfaces are near-threshold non-equivalent (p = 0.070 and p = 0.099), consistent with the perceptual asymmetry hypothesis.

Multimodal surface is TOST-inconclusive (bimodal traps, honest limitation).

Fixed
GP sample_posterior rng handling (Generator / int / RandomState all accepted).

NumPy 2.0 compatibility: np.trapz to np.trapezoid; np.math.erf to math.erf.

Removed the Colab setup notebook (had a leaked token in saved outputs); the repo no longer contains any secrets and GitHub secret-scanning is clear.

Known limitations
Synthetic reward surfaces; real human HIL data is future work.

10 seeds for TOST; asymmetric and timing-sensitive results are near-threshold, not conclusively non-equivalent.

Multimodal surface traps both EI and JND-EI; a multi-basin acquisition would be needed to reach both peaks.

Security
No tokens, passwords, or API keys in the repository.

The repo was force-pushed once (Step 3) to remove a leaked token from history; GitHub secret-scanning now reports "no secrets found".
