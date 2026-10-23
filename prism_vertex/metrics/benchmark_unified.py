"""
Paper metric primitives for prism-mf-sbi.

Kept metrics (manuscript): NLTP, CRPS, NRMSE (in callers), TARP, SBC ranks/KS,
posterior STD (in callers). Extended bake-off metrics (C2ST/MMD/ESS/MEDDIST/…)
are not used in the paper and are omitted.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kstest

try:
    from scipy.stats import gaussian_kde
except ImportError:  # pragma: no cover
    gaussian_kde = None  # type: ignore[misc, assignment]


def crps_univariate(samples_1d: np.ndarray, y: float) -> float:
    x = np.asarray(samples_1d, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return float("nan")
    e1 = float(np.mean(np.abs(x - y)))
    if n > 800:
        rng = np.random.default_rng(0)
        x = x[rng.choice(n, size=800, replace=False)]
        n = 800
    dd = np.abs(x[:, None] - x[None, :])
    e2 = 0.5 * float(np.mean(dd))
    return float(e1 - e2)


def crps_mean_multivariate(samples: np.ndarray, theta_true: np.ndarray) -> float:
    d = min(samples.shape[1], len(theta_true))
    vals = [crps_univariate(samples[:, j], float(theta_true[j])) for j in range(d)]
    return float(np.nanmean(vals))


def nltp_kde(samples: np.ndarray, theta_true: np.ndarray) -> float:
    if gaussian_kde is None:
        return float("nan")
    x = np.asarray(samples, dtype=float)
    t = np.asarray(theta_true, dtype=float).reshape(-1)
    if x.shape[0] < x.shape[1] + 2:
        return float("nan")
    try:
        kde = gaussian_kde(x.T)
        lp = float(kde.logpdf(t))
        return float(-lp)
    except Exception:
        return float("nan")


def tarp_coverage_error_mean(
    samples: np.ndarray,
    theta_true: np.ndarray,
    nominal_levels: Sequence[float] = (0.5, 0.683, 0.9, 0.95, 0.99),
) -> float:
    errs: list[float] = []
    for j in range(min(samples.shape[1], len(theta_true))):
        vals = samples[:, j]
        for nl in nominal_levels:
            alpha = (1.0 - nl) / 2.0
            lo, hi = np.quantile(vals, [alpha, 1.0 - alpha])
            covered = 1.0 if (theta_true[j] >= lo and theta_true[j] <= hi) else 0.0
            errs.append(abs(nl - covered))
    return float(np.mean(errs)) if errs else float("nan")


def tarp_coverage_curve(
    samples: np.ndarray,
    theta_true: np.ndarray,
    *,
    alphas: np.ndarray | None = None,
) -> np.ndarray:
    """TARP-style expected coverage vs nominal level α."""
    if alphas is None:
        alphas = np.linspace(0.0, 1.0, 501)
    x = np.asarray(samples, dtype=float)
    t = np.asarray(theta_true, dtype=float).reshape(-1)
    d = min(x.shape[1], len(t))
    if d <= 0 or x.shape[0] <= 1:
        return np.full(int(np.asarray(alphas).shape[0]), np.nan, dtype=float)

    alphas = np.asarray(alphas, dtype=float).reshape(-1)
    x = x[:, :d]
    t = t[:d]
    ranks = np.mean(x <= t[None, :], axis=0)

    out = np.zeros_like(alphas, dtype=float)
    for i, a in enumerate(alphas):
        a = float(np.clip(a, 0.0, 1.0))
        if a <= 0.0:
            out[i] = 0.0
            continue
        if a >= 1.0:
            out[i] = 1.0
            continue
        qlo = (1.0 - a) / 2.0
        qhi = 1.0 - qlo
        out[i] = float(np.mean((ranks >= qlo) & (ranks <= qhi)))
    return out


def posterior_ranks(samples: np.ndarray, theta_true: np.ndarray) -> np.ndarray:
    """Per-dimension normalized ranks r_j = P(theta_j <= theta*_j)."""
    x = np.asarray(samples, dtype=float)
    t = np.asarray(theta_true, dtype=float).reshape(-1)
    d = min(x.shape[1], len(t))
    if d <= 0 or x.shape[0] <= 1:
        return np.asarray([], dtype=float)
    return np.mean(x[:, :d] <= t[:d][None, :], axis=0).astype(float)


def dataframe_sbc_pooled_ks(ranks_long: pd.DataFrame) -> pd.DataFrame:
    """Pooled SBC KS test of rank_normalized vs Uniform(0,1)."""
    rows: list[dict] = []
    for (method, budget, pidx), g in ranks_long.groupby(["method", "budget", "parameter_index"]):
        vals = g["rank_normalized"].to_numpy(dtype=float)
        if len(vals) < 3:
            continue
        ks = kstest(vals, "uniform")
        rows.append(
            {
                "method": method,
                "budget": budget,
                "parameter_index": int(pidx),
                "metric_name": "sbc_pooled_ks_pvalue",
                "metric_value_mean": float(ks.pvalue),
                "metric_value_std": 0.0,
                "n_ranks": len(vals),
            }
        )
    return pd.DataFrame(rows)
