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
    assert call["rho"] > 0 and put["rho"] < 0  # call gains, put loses as rates rise


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


def test_enrich_chain_clamps_garbage_iv_and_otm_only_csp():
    # row0: OTM put (strike<spot), sane quote -> solvable IV, finite CSP yield.
    # row1: deep-ITM put with mid far below intrinsic -> solver fails, Yahoo IV
    #       is garbage (15.0 == 1500%) -> IV dropped to NaN; ITM -> CSP NaN.
    chain = pd.DataFrame({
        "strike": [95.0, 150.0],
        "bid": [2.0, 0.9],
        "ask": [2.2, 1.1],
        "lastPrice": [2.1, 1.0],
        "openInterest": [100, 5],
        "volume": [10, 1],
        "impliedVolatility": [0.30, 15.0],
    })
    enr = O.enrich_chain(chain, spot=100, r=0.03, expiry_dte=30, kind="put", realized_vol=0.2)
    # OTM put
    assert np.isfinite(enr.loc[0, "iv"]) and 0 < enr.loc[0, "iv"] <= O.MAX_PLAUSIBLE_IV
    assert np.isfinite(enr.loc[0, "csp_yield"]) and enr.loc[0, "csp_yield"] > 0
    assert enr.loc[0, "csp_yield"] < 5.0  # sane, not 400,000%
    # garbage / ITM row
    assert np.isnan(enr.loc[1, "iv"])
    assert enr.loc[1, "iv_source"] == "n/a"
    assert np.isnan(enr.loc[1, "csp_yield"])
    assert bool(enr.loc[0, "two_sided"]) and bool(enr.loc[1, "two_sided"])


def test_enrich_drops_degenerate_near_expiry_iv():
    # 0-DTE-ish deep-ITM put: mid ~= intrinsic, so the solver returns ~0. That
    # degenerate "IV 0.0%" must be dropped to NaN (below the iv floor), not
    # passed through to pollute richness (the Run-4 NVDA bug).
    chain = pd.DataFrame({
        "strike": [380.0], "bid": [239.9], "ask": [240.1], "lastPrice": [240.0],
        "openInterest": [5000], "volume": [3000], "impliedVolatility": [0.0],
    })
    enr = O.enrich_chain(chain, spot=140.0, r=0.036, expiry_dte=0.5, kind="put", realized_vol=0.41)
    assert np.isnan(enr.loc[0, "iv"])


def test_liquidity_filter_dte_and_moneyness():
    df = pd.DataFrame({
        "iv": [0.3, 0.3, 0.3, 0.3],
        "mid": [2.0, 2.0, 2.0, 2.0],
        "two_sided": [True, True, True, True],
        "open_int": [1000, 1000, 1000, 1000],
        "spread_pct": [5, 5, 5, 5],
        "dte": [0.5, 30, 30, 30],          # row0 is 0-DTE -> dropped
        "moneyness": [1.0, 1.0, 2.5, 0.5],  # row2 deep ITM, row3 deep OTM -> dropped
    })
    out = O.liquidity_filter(df, min_open_int=10, min_dte=7, moneyness_range=(0.7, 1.3))
    assert list(out.index) == [1]


def test_liquidity_report_names_binding_constraint():
    # Every row has a one-sided quote -> that screen should dominate the report.
    df = pd.DataFrame({
        "iv": [0.3, 0.3, 0.3],
        "mid": [2.0, 2.0, 2.0],
        "two_sided": [False, False, False],
        "open_int": [1000, 1000, 5],
        "spread_pct": [5, 5, 5],
        "dte": [30, 30, 30],
        "moneyness": [1.0, 1.0, 1.0],
    })
    rep = O.liquidity_report(df, min_open_int=10, require_two_sided=True,
                             max_spread_pct=25, min_dte=7, moneyness_range=(0.7, 1.3))
    assert rep["one-sided quote (bid or ask = 0)"] == 3
    assert max(rep.items(), key=lambda kv: kv[1])[0] == "one-sided quote (bid or ask = 0)"
    assert rep["open interest < 10"] == 1


def test_delta_band_excludes_deep_itm_and_far_otm():
    # Deep-ITM (|delta|~0.99, ~zero vega -> unstable IV) and far-OTM (|delta|~0.01)
    # must be dropped; the ATM/OTM premium zone kept.
    df = pd.DataFrame({
        "iv": [0.2, 0.2, 0.2, 0.2],
        "mid": [1, 1, 1, 1],
        "two_sided": [True, True, True, True],
        "open_int": [1000, 1000, 1000, 1000],
        "spread_pct": [5, 5, 5, 5],
        "dte": [30, 30, 30, 30],
        "moneyness": [1.05, 0.98, 0.92, 0.80],
        "abs_delta": [0.99, 0.45, 0.20, 0.01],  # deep ITM, ATM, OTM, far OTM
    })
    out = O.liquidity_filter(df, delta_band=(0.05, 0.55))
    assert list(out.index) == [1, 2]  # only ATM + OTM survive


def test_liquidity_filter():
    df = pd.DataFrame({
        "iv": [0.30, np.nan, 0.25, 0.40, 0.35],
        "mid": [2.0, 1.0, 0.0, 3.0, 2.0],
        "two_sided": [True, True, True, False, True],
        "open_int": [100, 50, 200, 500, 5],
        "spread_pct": [5, 5, 5, 5, 80],
    })
    out = O.liquidity_filter(df, min_open_int=10, require_two_sided=True, max_spread_pct=50)
    # row0 kept; row1 dropped (NaN iv); row2 dropped (mid 0);
    # row3 dropped (one-sided); row4 dropped (OI 5 < 10 and spread 80 > 50)
    assert list(out.index) == [0]


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
                "vrp", "iv_rv_ratio", "spread_pct", "open_int", "log_moneyness", "abs_delta"):
        assert col in enr.columns
    assert (enr["mid"] > 0).all()
    assert np.isfinite(enr["vrp"]).all()
    # IV/RV ratio is consistent with VRP = IV - RV at RV=0.2
    assert enr["iv_rv_ratio"].iloc[0] == pytest.approx(enr["iv"].iloc[0] / 0.2)
