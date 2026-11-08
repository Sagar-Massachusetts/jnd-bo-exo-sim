"""
jnd-bo-exo-sim
==============
Simulation testbed for JND-informed Bayesian optimization of ankle-exoskeleton
assistance profiles.

Modules
-------
walker           : planar simulated walker (dynamics + gait phase).
torque_profile   : 4-parameter ankle-exoskeleton torque policy.
reward_surfaces  : library of synthetic performance / preference surfaces.
human_oracle     : simulated human feedback (scalar performance or pairwise preference).
gp_model         : Gaussian-process posterior over the reward surface.
acquisitions     : acquisition functions (EI, Thompson, JND-aware).
experiment       : main optimization loop.
metrics          : regret, trials-to-convergence, validation accuracy.
tost             : two-one-sided-tests equivalence analysis.
"""
