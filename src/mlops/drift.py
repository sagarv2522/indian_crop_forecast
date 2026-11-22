"""Simple drift metrics for crop price time series and forecast logs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:
    """Compute PSI between two numeric distributions.

    PSI values above ~0.25 usually indicate a meaningful shift.
    """

    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)
    if expected.empty or actual.empty:
        raise ValueError("PSI requires non-empty expected and actual series")

    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = expected.quantile(quantiles).to_numpy()
    cut_points[0] = -np.inf
    cut_points[-1] = np.inf
    cut_points = np.unique(cut_points)
    if len(cut_points) < 3:
        return 0.0

    expected_hist, _ = np.histogram(expected, bins=cut_points)
    actual_hist, _ = np.histogram(actual, bins=cut_points)

    expected_pct = np.where(expected_hist == 0, 1e-6, expected_hist / expected_hist.sum())
    actual_pct = np.where(actual_hist == 0, 1e-6, actual_hist / actual_hist.sum())

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi, 4))


def window_drift_report(
    series: pd.Series,
    baseline_days: int = 365,
    recent_days: int = 90,
) -> dict:
    """Compare a historical baseline window against the latest window."""

    series = pd.Series(series).dropna().astype(float)
    if len(series) < baseline_days + recent_days:
        raise ValueError("Not enough history for the requested drift windows")

    baseline = series.iloc[-(baseline_days + recent_days) : -recent_days]
    recent = series.iloc[-recent_days:]

    return {
        "baseline_days": baseline_days,
        "recent_days": recent_days,
        "baseline_mean": round(float(baseline.mean()), 2),
        "recent_mean": round(float(recent.mean()), 2),
        "mean_shift_pct": round(float((recent.mean() - baseline.mean()) / baseline.mean() * 100), 2),
        "median_shift_pct": round(float((recent.median() - baseline.median()) / baseline.median() * 100), 2),
        "psi": population_stability_index(baseline, recent),
    }

