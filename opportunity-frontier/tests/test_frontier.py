"""Tests for the cross-tab unified risk frontier and the IV snapshot store."""
import numpy as np
import pandas as pd
import pytest

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
