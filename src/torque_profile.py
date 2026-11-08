"""
torque_profile.py
------------------
Constrained 4-parameter ankle-exoskeleton torque policy.

Parameters (fixed order throughout the project):
    0: rise_time      [fraction of stride]
    1: peak_timing    [fraction of stride]
    2: fall_time      [fraction of stride]
    3: peak_torque    [N m / kg]

Status: PLACEHOLDER (Step 2 will implement this file).
"""

import numpy as np

PARAM_NAMES = ["rise_time", "peak_timing", "fall_time", "peak_torque"]
PARAM_BOUNDS = np.array([
    [0.10, 0.40],
    [0.30, 0.60],
    [0.05, 0.20],
    [0.10, 1.00],
])
PARAM_JND = np.array([0.030, 0.030, 0.030, 0.100])   # provisional; see Step 2

def torque_profile(phase, params):
    """Return ankle torque for a given gait phase and 4-parameter policy."""
    raise NotImplementedError("torque_profile will be implemented in Step 2.")
