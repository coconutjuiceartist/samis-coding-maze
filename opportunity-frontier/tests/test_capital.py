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


# --------------------------------------------------------------------------- #
# crash_scenarios: the visceral Taleb/Marks downside
# --------------------------------------------------------------------------- #
def test_crash_loss_grows_with_move_size():
    # SPY 701p, $0.75 mid, spot 747: breakeven 700.25.
    s = C.crash_scenarios(mid=0.75, strike=701.0, spot=747.0)
    assert s["loss_10"] < s["loss_15"] < s["loss_20"]
    # −10%: 747*0.90 = 672.3 ⇒ (700.25 − 672.3)*100 ≈ $2,795/contract.
    assert s["loss_10"] == pytest.approx(2795.0, abs=5.0)
    # −20%: 747*0.80 = 597.6 ⇒ (700.25 − 597.6)*100 ≈ $10,265/contract.
    assert s["loss_20"] == pytest.approx(10265.0, abs=5.0)


def test_crash_loss_zero_above_breakeven():
    # A tiny 1% dip stays well above the 700.25 breakeven ⇒ no loss.
    s = C.crash_scenarios(mid=0.75, strike=701.0, spot=747.0, moves=(-0.01,))
    assert s["loss_1"] == 0.0
    assert s["loss_pct_1"] == 0.0


def test_crash_sigma_needs_iv_and_dte():
    base = dict(mid=0.75, strike=701.0, spot=747.0)
    assert math.isnan(C.crash_scenarios(**base)["loss_sigma"])
    assert C.crash_scenarios(**base, iv=0.21, dte=14.0)["loss_sigma"] >= 0.0


# --------------------------------------------------------------------------- #
# excess_per_tail: capital efficiency, and the SPY-vs-TSLA ranking
# --------------------------------------------------------------------------- #
def test_excess_per_tail_floors_denominator():
    # Near-zero modelled tail must not blow the ratio up: 0.02 / floor(0.005).
    assert C.excess_per_tail(0.02, 0.0) == pytest.approx(0.02 / 0.005)
    assert C.excess_per_tail(0.02, 0.10) == pytest.approx(0.2)
    assert math.isnan(C.excess_per_tail(float("nan"), 0.10))


def test_thin_spy_sell_ranks_below_fatter_tsla_sell_by_capital_efficiency():
    """The crux: a thin-premium SPY 5-delta sell should rank BELOW a
    fatter-premium TSLA sell once pay is measured PER unit of a real (fixed,
    deep) crash — the honest denominator. A 2-sigma move would flatter SPY,
    because its cheap priced vol makes the tail look tiny (the Taleb trap)."""
    spy = C.csp_metrics(mid=0.75, strike=701.0, dte=14.0, rf_matched=0.044, spot=747.0, iv=0.21)
    tsla = C.csp_metrics(mid=3.0, strike=305.0, dte=36.0, rf_matched=0.044, spot=330.0, iv=0.55)
    spy_crash = C.crash_scenarios(mid=0.75, strike=701.0, spot=747.0)["loss_pct_20"]
    tsla_crash = C.crash_scenarios(mid=3.0, strike=305.0, spot=330.0)["loss_pct_20"]
    spy_eff = C.excess_per_tail(spy["excess_yield"], spy_crash)
    tsla_eff = C.excess_per_tail(tsla["excess_yield"], tsla_crash)
    assert tsla_eff > spy_eff
