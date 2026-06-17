"""Small, network-free statistical helpers used across both tabs.

These are pure functions so they can be unit-tested with synthetic data
(see CLAUDE.md §8 "Tests").
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(series: pd.Series) -> pd.Series:
    """Population z-score of a series, robust to NaNs and zero variance.

    - Non-numeric entries are coerced to NaN.
    - NaNs are excluded from the mean/std but returned as 0 (neutral) so a
      single missing metric does not nuke a name's composite.
    - If the std is 0 or undefined (one usable value), returns all zeros.
    """
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index, dtype=float)
    return ((s - mu) / sd).fillna(0.0)


def winsorized_zscore(series: pd.Series, limit: float = 3.0) -> pd.Series:
    """z-score with the result clipped to +/- ``limit`` to tame outliers."""
    return zscore(series).clip(-limit, limit)


def rank_best_first(series: pd.Series, ascending: bool = False) -> pd.Series:
    """Rank where 1 == best. ``ascending=False`` means higher value is better.

    Used for Greenblatt's Magic Formula (rank ROIC/ROE and earnings yield,
    then sum the ranks; lowest combined rank == best).
    """
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(ascending=ascending, method="min", na_option="bottom")


def percentile_of(value: float, history: pd.Series) -> float:
    """Percentile (0-100) of ``value`` within a historical series.

    Returns NaN if there is not enough history.
    """
    h = pd.to_numeric(pd.Series(history), errors="coerce").dropna()
    if len(h) < 2 or not np.isfinite(value):
        return float("nan")
    return float((h < value).mean() * 100.0)
