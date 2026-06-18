"""Tests for core.capital — the capital-/risk-aware option framing.

Pure functions, no network. The SPCX Dec-2028 $25 put ($2.04 mid, ~2.5y) from
the design conversation is used as a concrete fixture so the numbers stay tied
to the real motivating example.
"""
import math

import numpy as np
import pytest

from core import capital as C


# --------------------------------------------------------------------------- #
# matched_rf: interpolation + flat extrapolation
# --------------------------------------------------------------------------- #
CURVE = {0.25: 0.052, 5.0: 0.043, 10.0: 0.044, 30.0: 0.046}


def test_matched_rf_interpolates_between_points():
    # 2.5y sits between the 0.25y and 5y points; linear interp.
    expected = np.interp(2.5, [0.25, 5, 10, 30], [0.052, 0.043, 0.044, 0.046])
    assert C.matched_rf(2.5, CURVE) == pytest.approx(expected)


def test_matched_rf_flat_extrapolates_past_ends():
    assert C.matched_rf(0.05, CURVE) == pytest.approx(0.052)   # below shortest tenor
    assert C.matched_rf(50.0, CURVE) == pytest.approx(0.046)   # beyond longest tenor


def test_matched_rf_empty_curve_is_nan():
    assert math.isnan(C.matched_rf(2.5, {}))


# --------------------------------------------------------------------------- #
# csp_metrics: the carry decision is the crux
# --------------------------------------------------------------------------- #
def test_csp_premium_yield_matches_spcx_handcalc():
    m = C.csp_metrics(mid=2.04, strike=25.0, dte=912.0, rf_matched=0.043)
    # 2.04 / (25 - 2.04) * 365/912 ≈ 3.55%/yr
    assert m["premium_yield"] == pytest.approx(0.0355, abs=5e-4)
    assert m["collateral"] == pytest.approx(22.96)
    assert m["breakeven"] == pytest.approx(22.96)
    assert m["assignment_price"] == 25.0


def test_carry_makes_premium_the_excess_and_stacks_blended():
    """Under carry: excess == premium (you still earn rf on the cash), and the
    blended return is carry + premium. This is the fix for the old double-count."""
    rf = 0.043
    m = C.csp_metrics(mid=2.04, strike=25.0, dte=912.0, rf_matched=rf, assume_carry=True)
    assert m["excess_yield"] == pytest.approx(m["premium_yield"])
    assert m["blended_return"] == pytest.approx(rf + m["premium_yield"])


def test_no_carry_charges_the_riskfree_against_idle_cash():
    rf = 0.043
    m = C.csp_metrics(mid=2.04, strike=25.0, dte=912.0, rf_matched=rf, assume_carry=False)
    assert m["excess_yield"] == pytest.approx(m["premium_yield"] - rf)
    assert m["blended_return"] == pytest.approx(m["premium_yield"])


def test_tail_loss_grows_with_vol_and_is_fraction_of_collateral():
    base = dict(mid=2.04, strike=25.0, dte=912.0, rf_matched=0.043, spot=24.0)
    calm = C.csp_metrics(**base, iv=0.30)["tail_loss"]
    wild = C.csp_metrics(**base, iv=0.80)["tail_loss"]
    assert 0.0 <= calm < wild                      # more vol ⇒ bigger tail loss
    assert wild <= 1.0 + 1e-9                       # capped at total collateral


def test_tail_loss_nan_without_spot_or_iv():
    m = C.csp_metrics(mid=2.04, strike=25.0, dte=912.0, rf_matched=0.043)
    assert math.isnan(m["tail_loss"])


# --------------------------------------------------------------------------- #
# risk coordinates / edge
# --------------------------------------------------------------------------- #
def test_short_put_risk_prefers_iv_over_delta():
    assert C.short_put_risk(abs_delta=0.10, iv=0.45) == pytest.approx(0.45)


def test_short_put_risk_falls_back_to_delta_when_iv_missing():
    assert C.short_put_risk(abs_delta=0.30, iv=float("nan")) == pytest.approx(0.30)


def test_long_option_edge_positive_when_iv_below_realized():
    assert C.long_option_edge(iv=0.20, realized_vol=0.35) == pytest.approx(0.15)
    assert math.isnan(C.long_option_edge(iv=float("nan"), realized_vol=0.3))
