"""Tests for the cross-tab unified risk frontier and the IV snapshot store."""
import numpy as np
import pandas as pd
import pytest

from core import capital as CAP
from core import frontier as FR
from core import iv_store


def test_build_frontier_shape_adjustment():
    rows = [
        FR.cash_row(0.04),
        FR.short_option_row(0.14, 0.04, risk=0.20, label="short put"),
        FR.long_option_row(0.10, 0.20, label="long call"),
    ]
    df = FR.build_frontier(rows)
    short = df[df["label"] == "short put"].iloc[0]
    long = df[df["label"] == "long call"].iloc[0]
    # Same-ish base Sharpe, but convex is rewarded and concave penalised.
    assert long["shape_adj_score"] > long["sharpe_like"]
    assert short["shape_adj_score"] < short["sharpe_like"]


def test_efficient_envelope_monotone():
    df = pd.DataFrame({
        "label": list("abcd"),
        "risk": [0.05, 0.10, 0.15, 0.20],
        "expected_excess_return": [0.02, 0.01, 0.05, 0.04],
    })
    env = FR.efficient_envelope(df)
    # envelope keeps only non-dominated points by return as risk rises
    assert list(env["label"]) == ["a", "c"]


def test_iv_store_roundtrip(tmp_path):
    db = tmp_path / "iv.sqlite"
    import datetime as dt
    base = dt.date(2025, 1, 1)
    for i, iv in enumerate([0.2, 0.25, 0.3, 0.35, 0.4]):
        iv_store.snapshot("TEST", iv, asof=base + dt.timedelta(days=i), db_path=db)
    h = iv_store.history("TEST", lookback_days=10000, db_path=db)
    assert len(h) == 5
    assert iv_store.iv_rank(0.3, h) == pytest.approx(50.0)
    assert iv_store.iv_percentile(0.3, h) == pytest.approx(40.0)


def test_atm_iv_picks_closest_strike():
    puts = pd.DataFrame({"strike": [90, 100, 110], "iv": [0.30, 0.25, 0.28]})
    assert iv_store.atm_iv_from_chain(puts, spot=101) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# capital ↔ frontier integration: the carry-correct mapping (Tab 3)
# --------------------------------------------------------------------------- #
def test_short_put_uses_premium_as_excess_not_premium_minus_rf():
    """The frontier excess for a carry-secured short put is the premium itself,
    NOT premium − rf (the bug the rewrite fixes). Verify the row we'd build
    carries the full premium as its excess-over-cash coordinate."""
    rf = 0.043
    m = CAP.csp_metrics(mid=2.04, strike=25.0, dte=912.0, rf_matched=rf, assume_carry=True)
    row = FR.FrontierRow("SELL SPCX 25p", "short_option",
                         m["excess_yield"], CAP.short_put_risk(0.10, 0.50), -1, "")
    assert row.expected_excess_return == pytest.approx(m["premium_yield"])
    assert row.expected_excess_return > rf - rf  # strictly positive, not net of rf


def test_low_premium_far_dated_short_put_ranks_below_credit():
    """A thin-premium, high-vol short put (the SPCX-style trade) must sort below
    cash-like HY credit once risk is measured honestly by the stock's vol."""
    rf = 0.043
    m = CAP.csp_metrics(mid=2.04, strike=25.0, dte=912.0, rf_matched=rf, assume_carry=True)
    rows = [
        FR.cash_row(rf),
        FR.credit_row(0.078, rf, "HY credit", risk=0.11),
        FR.FrontierRow("SELL SPCX 25p", "short_option",
                       m["excess_yield"], CAP.short_put_risk(0.10, 0.50), -1, ""),
    ]
    df = FR.build_frontier(rows)
    order = list(df["label"])
    assert order.index("HY credit") < order.index("SELL SPCX 25p")
