"""
Md Mainuddin Sagar-
-
tost.py two One-Sided Tests (TOST) for perceptual equivalence, with JND-informed
equivalence bounds.

The core question
Given two sets of optima (from different acquisition strategies), are they
perceptually the same? Standard null-hypothesis testing can only REJECT the
null of zero difference — it can never CONFIRM the null. TOST solves this by
inverting the test: the null is "there is a meaningful difference" (larger
than the equivalence bound), and we test whether we can reject it.

JND-informed bounds
Instead of a generic bound (e.g., 5% of the range), we use PARAM_JND — the
same perceptual thresholds already baked into the acquisition function. This
means "equivalent" is defined as "below the threshold a human can perceive,"
which is exactly what our research question cares about.

Public API
    tost(sample_a, sample_b, bound, alpha=0.05)
        -> dict with keys: 'lower_ci', 'upper_ci', 'p_lower', 'p_upper',
                           'equivalent' (bool), 'mean_diff', 'bound'

    tost_on_optima(optima_a, optima_b, jnd=None, alpha=0.05)
        -> dict mapping each parameter index -> tost result, plus an
           'all_equivalent' flag (True iff all four parameters are equivalent)
"""
from __future__ import annotations
import math
import numpy as np
from scipy import stats
from src.torque_profile import PARAM_JND, PARAM_NAMES

# Core TOST
def tost(sample_a, sample_b, bound: float, alpha: float = 0.05) -> dict:
    """
    Two One-Sided Tests for the difference of means (two independent samples).

    H0_lower : mean_a - mean_b <= -bound   (difference too far below zero)
    H0_upper : mean_a - mean_b >=  bound   (difference too far above zero)

    We reject both to declare equivalence.

    Parameters
    sample_a, sample_b : array-like
        Two independent samples of the same scalar quantity.
    bound : float
        Equivalence bound (must be > 0). For a two-sided TOST this is the
        absolute threshold on |mean_a - mean_b| beyond which we consider
        the two samples "meaningfully different."
    alpha : float
        Overall significance level (default 0.05).

    Returns
    dict with:
        'mean_diff'  : float      mean_a - mean_b
        'se'         : float      standard error of the difference
        'ci_lo'      : float      1 - 2*alpha CI lower bound on the diff
        'ci_hi'      : float      1 - 2*alpha CI upper bound on the diff
        'p_lower'    : float      p-value of the lower one-sided test
        'p_upper'    : float      p-value of the upper one-sided test
        'p_max'      : float      max(p_lower, p_upper) — the TOST p-value
        'bound'      : float      the equivalence bound (positive)
        'equivalent' : bool       True iff we reject both H0's
    """
    a = np.asarray(sample_a, dtype=float).ravel()
    b = np.asarray(sample_b, dtype=float).ravel()
    if a.size < 2 or b.size < 2:
        raise ValueError("TOST requires at least 2 samples in each group")
    if bound <= 0:
        raise ValueError("bound must be > 0")

    mean_diff = float(np.mean(a) - np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    se = math.sqrt(var_a / a.size + var_b / b.size)

    # Welch–Satterthwaite degrees of freedom
    if se == 0.0:
        df = float(a.size + b.size - 2)
    else:
        df = (var_a / a.size + var_b / b.size) ** 2 / (
            (var_a / a.size) ** 2 / (a.size - 1) +
            (var_b / b.size) ** 2 / (b.size - 1)
        )

    t_stat = mean_diff / se if se > 0 else 0.0

    # Lower one-sided test: H0: diff <= -bound, H1: diff > -bound
    p_lower = 1.0 - stats.t.cdf((mean_diff + bound) / se, df) if se > 0 else 0.0
    # Upper one-sided test: H0: diff >= +bound, H1: diff < +bound
    p_upper = stats.t.cdf((mean_diff - bound) / se, df) if se > 0 else 0.0

    # 1 - 2*alpha confidence interval (the CI dual to the TOST)
    t_crit = stats.t.ppf(1.0 - alpha, df)
    ci_lo = mean_diff - t_crit * se
    ci_hi = mean_diff + t_crit * se

    equivalent = bool(p_lower < alpha and p_upper < alpha)

    return {
        "mean_diff": mean_diff,
        "se": se,
        "df": df,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_max": float(max(p_lower, p_upper)),
        "bound": float(bound),
        "equivalent": equivalent,
    }

# Multi-parameter TOST on optima
def tost_on_optima(optima_a, optima_b, jnd=None, alpha: float = 0.05) -> dict:
    """
    Per-parameter TOST on the best parameter vector found by two methods.

    Parameters
    optima_a, optima_b : array-like, shape (n_samples, 4)
        The best-found parameter vector from each seed of method A and method B.
        These are typically obtained by scanning histories for argmax of
        true_rewards.
    jnd : np.ndarray (4,) | None
        Per-parameter equivalence bounds. Defaults to PARAM_JND.
    alpha : float

    Returns
    dict with keys:
        'params' : list of 4 dicts (one per parameter, from tost())
        'all_equivalent' : bool  True iff all 4 params are equivalent
        'worst_p' : float  max p-value across parameters
    """
    A = np.asarray(optima_a, dtype=float)
    B = np.asarray(optima_b, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != 4 or B.shape[1] != 4:
        raise ValueError("optima_a and optima_b must have shape (n_samples, 4)")
    if A.shape[0] < 2 or B.shape[0] < 2:
        raise ValueError("Need at least 2 samples per method for TOST")

    if jnd is None:
        jnd = PARAM_JND
    jnd = np.asarray(jnd, dtype=float)

    results = []
    for i in range(4):
        r = tost(A[:, i], B[:, i], bound=float(jnd[i]), alpha=alpha)
        r["param_name"] = PARAM_NAMES[i]
        results.append(r)

    return {
        "params": results,
        "all_equivalent": bool(all(r["equivalent"] for r in results)),
        "worst_p": float(max(r["p_max"] for r in results)),
    }

# Helper: extract best-found parameter vector from a history
def best_params_from_history(history: dict) -> np.ndarray:
    """
    Return the parameter vector that achieved the highest TRUE reward in a
    single optimizer run.
    """
    tr = np.asarray(history["true_rewards"], dtype=float)
    idx = int(np.argmax(tr))
    return np.asarray(history["params"][idx], dtype=float)
