"""
gp_model.py
-----------
Gaussian-process surrogate model for the reward surface.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.utils import check_random_state

from src.torque_profile import PARAM_BOUNDS, clamp_params

_LO = PARAM_BOUNDS[:, 0]
_HI = PARAM_BOUNDS[:, 1]
_SPAN = _HI - _LO


def _to_unit(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    X = np.array([clamp_params(row) for row in X])
    return (X - _LO) / _SPAN


class GPModel:
    """Thin wrapper around sklearn's GP regressor for the HIL optimizer."""

    def __init__(self,
                 noise_std: float = 0.02,
                 seed: int = 0,
                 length_scale: float = 0.4,
                 length_scale_bounds: tuple[float, float] = (0.05, 2.0)):
        self.noise_std = float(noise_std)
        self.seed = int(seed)
        self.length_scale = float(length_scale)
        self.length_scale_bounds = tuple(length_scale_bounds)

        self._gp: GaussianProcessRegressor | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, X_physical: np.ndarray, y: np.ndarray) -> "GPModel":
        X = _to_unit(X_physical)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X rows ({X.shape[0]}) != y length ({y.shape[0]})")

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) if np.std(y) > 1e-9 else 1.0
        y_norm = (y - self._y_mean) / self._y_std

        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2))
            * Matern(length_scale=self.length_scale,
                     length_scale_bounds=self.length_scale_bounds,
                     nu=2.5)
            + WhiteKernel(noise_level=self.noise_std ** 2,
                          noise_level_bounds=(1e-6, 1.0))
        )

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            n_restarts_optimizer=3,
            random_state=self.seed,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            try:
                gp.fit(X, y_norm)
            except Exception:
                rng_fallback = np.random.default_rng(self.seed)
                X_flat = np.vstack([
                    X,
                    X + 1e-6 * rng_fallback.standard_normal(X.shape),
                ])
                y_flat = np.concatenate([y_norm, y_norm])
                gp.fit(X_flat, y_flat)

        self._gp = gp
        return self

    def _require_fit(self):
        if self._gp is None:
            raise RuntimeError("GPModel is not fitted. Call .fit(X, y) first.")

    def predict(self, X_physical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior mean and std (both shape (M,)) in original reward units."""
        self._require_fit()
        Xq = _to_unit(X_physical)
        mu_norm, sigma_norm = self._gp.predict(Xq, return_std=True)
        mu = mu_norm * self._y_std + self._y_mean
        sigma = sigma_norm * self._y_std
        return np.asarray(mu), np.asarray(sigma)

    def predict_best(self, X_physical: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Return (best mean, corresponding params, argmax row from X_physical)."""
        mu, sigma = self.predict(X_physical)
        idx = int(np.argmax(mu))
        return float(mu[idx]), float(sigma[idx]), np.atleast_2d(X_physical)[idx].copy()

    def sample_posterior(self, X_physical: np.ndarray,
                         n_samples: int = 1,
                         rng=None) -> np.ndarray:
        """
        Sample from the GP posterior at the given points.

        Returns
        -------
        np.ndarray, shape (n_samples, M)
            Each row is one sample function evaluated at the M query points.
        """
        self._require_fit()
        if rng is None:
            rng_state = check_random_state(self.seed)
        elif isinstance(rng, (int, np.integer)):
            rng_state = check_random_state(int(rng))
        elif isinstance(rng, np.random.Generator):
            seed_int = int(rng.integers(0, 2**31 - 1))
            rng_state = check_random_state(seed_int)
        else:
            rng_state = rng

        Xq = _to_unit(X_physical)
        # sklearn returns (M, n_samples) — we transpose to (n_samples, M).
        samples_norm = self._gp.sample_y(Xq, n_samples=n_samples, random_state=rng_state).T
        return samples_norm * self._y_std + self._y_mean
