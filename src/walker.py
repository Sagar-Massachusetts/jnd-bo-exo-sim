"""
walker.py
---------
Planar simulated walker subject to gravity, ground contact, and friction.
Produces a gait-phase signal in [0, 1) that drives the exoskeleton torque policy.

Status: PLACEHOLDER (Step 2 will implement this file).
"""

class Walker:
    """Planar walker model that exposes `gait_phase(t) -> float in [0, 1)`."""
    def __init__(self, config: dict):
        raise NotImplementedError("Walker will be implemented in Step 2.")

    def step(self, dt: float) -> None:
        raise NotImplementedError

    def gait_phase(self) -> float:
        raise NotImplementedError
