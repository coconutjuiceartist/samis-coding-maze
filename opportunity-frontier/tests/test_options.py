"""Tests for the sacred options math (CLAUDE.md §8)."""
import math

import numpy as np
import pandas as pd
import pytest

from core import options as O


def test_atm_call_reference():
    """The locked reference: S=K=100, T=1, r=0, sigma=0.2."""
    g = O.bs(100, 100, 1.0, 0.0, 0.2, kind="call")
    assert g["price"] == pytest.approx(7.97, abs=0.02)
    assert g["delta"] == pytest.approx(0.540, abs=0.005)


def test_put_call_parity():
    S, K, T, r, sig = 100, 95, 0.5, 0.03, 0.25
    c = O.bs(S, K, T, r, sig, kind="call")["price"]
    p = O.bs(S, K, T, r, sig, kind="put")["price"]
    # c - p = S - K e^{-rT}
    assert (c - p) == pytest.approx(S - K * math.exp(-r * T), abs=1e-6)


def test_greeks_signs():
    call = O.bs(100, 100, 1.0, 0.02, 0.2, kind="call")
    put = O.bs(100, 100, 1.0, 0.02, 0.2, kind="put")
    assert call["delta"] > 0 and put["delta"] < 0
    assert call["gamma"] > 0 and call["vega"] > 0
    assert call["theta"] < 0  # long option decays


def test_degenerate_inputs_return_intrinsic():
    g = O.bs(110, 100, 0.0, 0.01, 0.2, kind="call")
    assert g["price"] == pytest.approx(10.0)
    assert g["gamma"] == 0.0 and g["vega"] == 0.0


def test_implied_vol_roundtrip():
    true_sig = 0.27
    price = O.bs(100, 105, 0.5, 0.03, true_sig, kind="put")["price"]
    solved = O.implied_vol(price, 100, 105, 0.5, 0.03, kind="put")
    assert solved == pytest.approx(true_sig, abs=1e-4)


def test_implied_vol_garbage_returns_nan():
    # Price below intrinsic is non-arbitrageable.
    assert math.isnan(O.implied_vol(0.01, 100, 80, 0.5, 0.0, kind="call"))
    assert math.isnan(O.implied_vol(-1, 100, 100, 0.5, 0.0))


def test_csp_yield():
    # mid 2, strike 100, 30 dte -> 2/98 * 365/30
    y = O.csp_annualized_yield(2.0, 100.0, 30.0)
    assert y == pytest.approx(2 / 98 * 365 / 30, rel=1e-9)
    assert math.isnan(O.csp_annualized_yield(120, 100, 30))  # negative denom


def test_enrich_chain_columns():
    chain = pd.DataFrame({
        "strike": [90, 95, 100, 105],
        "bid": [1.0, 2.0, 3.0, 4.0],
        "ask": [1.2, 2.2, 3.2, 4.2],
        "lastPrice": [1.1, 2.1, 3.1, 4.1],
        "openInterest": [10, 20, 30, 40],
        "volume": [1, 2, 3, 4],
        "impliedVolatility": [0.3, 0.28, 0.26, 0.27],
    })
    enr = O.enrich_chain(chain, spot=100, r=0.03, expiry_dte=30, kind="put", realized_vol=0.2)
    for col in ("mid", "iv", "delta", "gamma", "theta", "vega", "csp_yield",
                "vrp", "spread_pct", "open_int", "log_moneyness", "abs_delta"):
        assert col in enr.columns
    assert (enr["mid"] > 0).all()
    assert np.isfinite(enr["vrp"]).all()
