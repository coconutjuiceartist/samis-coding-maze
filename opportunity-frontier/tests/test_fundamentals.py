"""Tests for fundamentals metrics, composites, ROIC/WACC and the F-score."""
import numpy as np
import pandas as pd
import pytest

from core import fundamentals as F
from core import stats as S


def test_zscore_basic():
    z = S.zscore(pd.Series([1, 2, 3, 4, 5]))
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.iloc[0] < 0 and z.iloc[-1] > 0


def test_zscore_zero_variance_and_nan():
    assert (S.zscore(pd.Series([7, 7, 7])) == 0).all()
    z = S.zscore(pd.Series([1, 2, np.nan, 4]))
    assert z.isna().sum() == 0  # NaN filled neutral


def test_metrics_from_info():
    info = {
        "totalRevenue": 1000, "freeCashflow": 200, "marketCap": 5000,
        "ebitda": 300, "totalDebt": 400, "totalCash": 100, "trailingPE": 25,
        "revenueGrowth": 0.1, "grossMargins": 0.5, "operatingMargins": 0.3,
        "profitMargins": 0.2, "returnOnEquity": 0.25, "returnOnAssets": 0.12,
        "enterpriseToEbitda": 15, "priceToBook": 8, "debtToEquity": 120,
        "currentRatio": 1.5,
    }
    m = F.metrics_from_info(info)
    assert m["fcf_margin"] == pytest.approx(0.2)
    assert m["fcf_yield"] == pytest.approx(0.04)
    assert m["earnings_yield"] == pytest.approx(0.04)
    assert m["net_debt_ebitda"] == pytest.approx((400 - 100) / 300)
    assert m["debt_equity"] == pytest.approx(1.2)  # normalised from %


def test_compute_composites_star_flag():
    df = pd.DataFrame({
        "gross_margin": [0.6, 0.2], "op_margin": [0.4, 0.1], "net_margin": [0.3, 0.05],
        "fcf_margin": [0.25, 0.05], "roe": [0.3, 0.05], "roa": [0.15, 0.03],
        "fcf_yield": [0.06, 0.02], "earnings_yield": [0.06, 0.02],
        "ev_ebitda": [10, 30], "pe": [15, 40], "pb": [3, 12],
        "rev_growth": [0.2, 0.0], "current_ratio": [2.0, 0.8],
        "net_debt_ebitda": [0.5, 4.0], "debt_equity": [0.3, 2.0],
    }, index=["GOOD", "BAD"])
    out = F.compute_composites(df)
    assert out.loc["GOOD", "Composite"] > out.loc["BAD", "Composite"]
    assert out.loc["GOOD", "star"]  # high quality and cheap
    assert not out.loc["BAD", "star"]
    assert out.loc["GOOD", "MF_rank"] < out.loc["BAD", "MF_rank"]


def test_compute_roic():
    # EBIT 100, tax 20%, invested = 400+600-100 = 900 -> NOPAT 80 / 900
    roic = F.compute_roic(100, 0.2, 400, 600, 100)
    assert roic == pytest.approx(80 / 900)
    assert np.isnan(F.compute_roic(100, 0.2, 0, 0, 1000))  # non-positive invested


def test_compute_wacc():
    # equity 800, debt 200; cost equity = 0.04 + 1.2*0.05 = 0.10
    # after-tax cost debt = 0.06*(1-0.21)=0.0474
    w = F.compute_wacc(1.2, 0.04, 0.05, 0.06, 0.21, 800, 200)
    expected = 0.8 * 0.10 + 0.2 * 0.0474
    assert w == pytest.approx(expected, rel=1e-6)


def test_piotroski_perfect_and_zero():
    curr = {"net_income": 10, "op_cash_flow": 15, "roa": 0.12, "long_term_debt": 50,
            "current_ratio": 2.0, "shares": 100, "gross_margin": 0.5, "asset_turnover": 0.9}
    prev = {"net_income": 5, "op_cash_flow": 8, "roa": 0.10, "long_term_debt": 60,
            "current_ratio": 1.8, "shares": 100, "gross_margin": 0.45, "asset_turnover": 0.8}
    assert F.piotroski_fscore(curr, prev) == 9

    bad_curr = {"net_income": -5, "op_cash_flow": -10, "roa": 0.05, "long_term_debt": 80,
                "current_ratio": 1.0, "shares": 120, "gross_margin": 0.4, "asset_turnover": 0.7}
    bad_prev = {"net_income": 10, "op_cash_flow": 12, "roa": 0.10, "long_term_debt": 50,
                "current_ratio": 1.5, "shares": 100, "gross_margin": 0.5, "asset_turnover": 0.9}
    assert F.piotroski_fscore(bad_curr, bad_prev) == 0
