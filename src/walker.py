"""
walker.py
---------
Planar simulated walker producing a deterministic gait-phase signal.

We deliberately keep the dynamical model simple and *deterministic* so that
the human-in-the-loop optimizer in later steps is exercised by the optimizer,
not by walking noise. The gait phase is the input to the exoskeleton torque
policy.

Conventions
-----------
* `gait_phase(t)` is in [0, 1).  0 = heel strike of the reference foot.
* The stride period is a constructor parameter (default 1.0 s).
* The phase wraps forward-only (never resets by going negative).
"""

from __future__ import annotations


class Walker:
    """Planar walker exposing `gait_phase()` and stepping time forward."""

    def __init__(self, config: dict | None = None,
                 stride_seconds: float = 1.0,
                 phase0: float = 0.0):
        self.config = config or {}
        self.stride_seconds = float(stride_seconds)
        if self.stride_seconds <= 0.0:
            raise ValueError("stride_seconds must be positive")

        self.phase0 = float(phase0) % 1.0
        self._t = 0.0
        self._stride_count = 0

    def step(self, dt: float) -> None:
        """Advance simulated time by dt seconds."""
        if dt < 0:
            raise ValueError("dt must be non-negative")
        self._t += float(dt)

    def reset(self) -> None:
        self._t = 0.0
        self._stride_count = 0

    def gait_phase(self) -> float:
        """Current gait phase in [0, 1)."""
        raw = (self.phase0 + self._t / self.stride_seconds) % 1.0
        return float(raw)

    def gait_phase_at(self, t: float) -> float:
        """Gait phase at an arbitrary time t (seconds from start)."""
        raw = (self.phase0 + float(t) / self.stride_seconds) % 1.0
        return float(raw)

    def time(self) -> float:
        return float(self._t)

    def stride_count(self) -> int:
        return int(self._t // self.stride_seconds)
