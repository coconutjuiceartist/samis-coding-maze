"""Volatility frontiers for Tab 1: realised vol, the skew (smile) frontier,
the yield-vs-risk capital frontier, and the composite richness score.

All functions are pure (DataFrames / arrays in, DataFrames / floats out) so
they can be tested without a network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .options import TRADING_DAYS
from .stats import zscore


def realized_vol(close: pd.Series, window: int) -> float:
    """Annualised close-to-close realised volatility over the last ``window``
    trading days. Returns NaN if there is not enough history.
    """
    s = pd.to_numeric(pd.Series(close), errors="coerce").dropna()
    if len(s) < window + 1:
        if len(s) < 3:
            return float("nan")
        window = len(s) - 1
    log_ret = np.log(s / s.shift(1)).dropna().tail(window)
    if len(log_ret) < 2:
        return float("nan")
    return float(log_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))


def realized_vol_panel(close: pd.Series, windows=(20, 60, 200)) -> dict:
    """Realised vol at several lookbacks, e.g. {20: .., 60: .., 200: ..}."""
    return {w: realized_vol(close, w) for w in windows}


def fit_skew(df: pd.DataFrame, deg: int = 2) -> pd.DataFrame:
    """Fit IV vs log-moneyness (deg-2 by default) per (ticker, expiry) group
    and attach the fitted IV and the residual.

    The fitted curve *is* the local frontier; ``skew_resid`` = actual - fitted
    flags a contract rich (+) or cheap (-) relative to its own smile
    (CLAUDE.md §4, frontier 1).
    """
    out = df.copy()
    out["iv_fit"] = np.nan
    out["skew_resid"] = np.nan
    if out.empty or "log_moneyness" not in out or "iv" not in out:
        return out

    group_cols = [c for c in ("ticker", "expiry") if c in out.columns]
    if not group_cols:
        out["_grp"] = 0
        group_cols = ["_grp"]

    for _, idx in out.groupby(group_cols).groups.items():
        sub = out.loc[idx]
        mask = np.isfinite(sub["iv"]) & np.isfinite(sub["log_moneyness"])
        x = sub.loc[mask, "log_moneyness"].to_numpy()
        y = sub.loc[mask, "iv"].to_numpy()
        if mask.sum() <= deg:
            continue
        local_deg = min(deg, mask.sum() - 1)
        coeffs = np.polyfit(x, y, local_deg)
        fitted = np.polyval(coeffs, sub["log_moneyness"].to_numpy())
        out.loc[idx, "iv_fit"] = fitted
        out.loc[idx, "skew_resid"] = sub["iv"].to_numpy() - fitted

    return out.drop(columns=[c for c in ("_grp",) if c in out.columns])


def yield_risk_frontier(df: pd.DataFrame, n_bins: int = 12) -> pd.DataFrame:
    """Yield-vs-risk capital frontier (CLAUDE.md §4, frontier 3).

    Bins puts by |delta|; within each bin the max annualised CSP yield is the
    frontier. ``value_ratio`` = a contract's yield / the frontier yield at its
    risk bucket (>1 == pays more than peers for the same |delta|).
    """
    out = df.copy()
    out["frontier_yield"] = np.nan
    out["value_ratio"] = np.nan
    if out.empty or "abs_delta" not in out or "csp_yield" not in out:
        return out

    mask = np.isfinite(out["abs_delta"]) & np.isfinite(out["csp_yield"]) & (out["csp_yield"] > 0)
    if mask.sum() < 2:
        return out

    sub = out[mask]
    edges = np.linspace(0.0, max(sub["abs_delta"].max(), 1e-6), n_bins + 1)
    bins = np.clip(np.digitize(sub["abs_delta"], edges) - 1, 0, n_bins - 1)
    frontier_by_bin = pd.Series(sub["csp_yield"].to_numpy()).groupby(bins).max()
    frontier = pd.Series(bins, index=sub.index).map(frontier_by_bin)
    out.loc[mask, "frontier_yield"] = frontier.to_numpy()
    out.loc[mask, "value_ratio"] = (sub["csp_yield"] / frontier).to_numpy()
    return out


def add_richness(df: pd.DataFrame) -> pd.DataFrame:
    """Composite richness = mean of standardised skew-residual and standardised
    VRP, with a categorical label (CLAUDE.md §4).

    RICH (>+0.5) => premium is dear, lean sell; CHEAP (<-0.5) => lean buy.
    """
    out = df.copy()
    z_skew = zscore(out["skew_resid"]) if "skew_resid" in out else pd.Series(0.0, index=out.index)
    z_vrp = zscore(out["vrp"]) if "vrp" in out else pd.Series(0.0, index=out.index)
    out["richness"] = (z_skew + z_vrp) / 2.0
    out["signal"] = np.select(
        [out["richness"] > 0.5, out["richness"] < -0.5],
        ["RICH → sell", "CHEAP → buy"],
        default="fair",
    )
    return out


def vix_regime(vix_level: float, vix_percentile: float) -> str:
    """One-line regime read tying VIX level/percentile to a stance.

    Low VIX => optionality cheap => favour buying convexity; high VIX =>
    favour selling premium (Blankfein: sell insurance when it's dear).
    """
    if not np.isfinite(vix_level):
        return "VIX unavailable — confirm the vol regime manually."
    pct = vix_percentile if np.isfinite(vix_percentile) else float("nan")
    pct_txt = f"{pct:.0f}th pct of the last year" if np.isfinite(pct) else "percentile unavailable"
    if np.isfinite(pct) and pct >= 70:
        stance = "premium is dear — favour SELLING premium (and hedge the tail)."
    elif np.isfinite(pct) and pct <= 30:
        stance = "optionality is cheap — favour BUYING convexity."
    else:
        stance = "a middling regime — be selective; let relative value decide."
    return f"VIX {vix_level:.1f} ({pct_txt}): {stance}"
