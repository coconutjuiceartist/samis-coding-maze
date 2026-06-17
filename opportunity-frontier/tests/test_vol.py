"""Tests for realised vol, skew fit, yield/risk frontier, richness."""
import numpy as np
import pandas as pd
import pytest

from core import vol as V


def test_realized_vol_constant_series_is_zero():
    s = pd.Series([100.0] * 100)
    assert V.realized_vol(s, 60) == pytest.approx(0.0, abs=1e-9)


def test_realized_vol_known_scale():
    rng = np.random.default_rng(0)
    daily = 0.01
    rets = rng.normal(0, daily, 5000)
    prices = 100 * np.exp(np.cumsum(rets))
    rv = V.realized_vol(pd.Series(prices), 252)
    # annualised ~ daily * sqrt(252)
    assert rv == pytest.approx(daily * np.sqrt(252), rel=0.15)


def test_realized_vol_insufficient_history():
    assert np.isnan(V.realized_vol(pd.Series([100.0]), 60))


def test_fit_skew_residual_zero_on_quadratic():
    x = np.linspace(-0.2, 0.2, 11)
    iv = 0.2 + 0.5 * x ** 2  # perfect parabola
    df = pd.DataFrame({"ticker": "X", "expiry": "2025-01-01",
                       "log_moneyness": x, "iv": iv})
    out = V.fit_skew(df)
    assert np.allclose(out["skew_resid"].to_numpy(), 0.0, atol=1e-9)


def test_yield_risk_frontier_value_ratio_le_one():
    df = pd.DataFrame({
        "abs_delta": [0.1, 0.1, 0.3, 0.3, 0.5],
        "csp_yield": [0.05, 0.08, 0.10, 0.12, 0.20],
    })
    out = V.yield_risk_frontier(df, n_bins=5)
    vr = out["value_ratio"].dropna()
    assert (vr <= 1.0 + 1e-9).all()
    assert (vr > 0).all()


def test_add_richness_labels():
    df = pd.DataFrame({
        "skew_resid": [0.05, -0.05, 0.0],
        "vrp": [0.10, -0.10, 0.0],
    })
    out = V.add_richness(df)
    assert out["signal"].iloc[0] == "RICH → sell"
    assert out["signal"].iloc[1] == "CHEAP → buy"


def test_vix_regime_branches():
    assert "SELLING" in V.vix_regime(30, 85)
    assert "BUYING" in V.vix_regime(12, 10)
    assert "unavailable" in V.vix_regime(float("nan"), float("nan"))
