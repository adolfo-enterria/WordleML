"""Small statistics helpers for the benchmark (numpy only)."""
import math

import numpy as np

# Two-sided 95% t critical values for small samples (df = n - 1); 1.96 beyond 30.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
        18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def t95(df):
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    smaller = [k for k in _T95 if k <= df]
    return _T95[max(smaller)] if df <= 30 else 1.96


def mean_ci(values):
    """(mean, half-width of the 95% confidence interval) using the t distribution."""
    values = np.asarray(values, float)
    if len(values) < 2:
        return float(values.mean()), float("nan")
    return float(values.mean()), float(t95(len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))


def paired_bootstrap(a, b, reps=5000, seed=0):
    """Mean of (a - b) over paired items (e.g. the same secret words) and a 95% bootstrap CI."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(len(diff), size=(reps, len(diff)))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def games_to_tell_apart(std, difference, z_alpha=1.96, z_power=0.84):
    """Games per option needed to detect `difference` in the mean (95% confidence, 80% power)."""
    return int(math.ceil(2 * (z_alpha + z_power) ** 2 * std ** 2 / difference ** 2))


def ols(x, y, names):
    """Least squares with standard errors. Returns rows of (name, coefficient, 95% CI half-width)."""
    X = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    rows = [(name, float(c), float(1.96 * s)) for name, c, s in zip(["intercept"] + list(names), coef, se)]
    return rows, float(r2)
