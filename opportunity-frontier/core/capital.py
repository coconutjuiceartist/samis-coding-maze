"""Capital- and risk-aware framing of an option against its alternatives
(CLAUDE.md §1, §6 — the north-star).

The governing question this module answers, per contract:

    "A trade is only as good as the best alternative use of the same capital
     and risk." How much does THIS option pay *over* the matched-maturity
     risk-free rate, once you assume the collateral itself earns that
     risk-free carry — and what could you actually lose if you're wrong
     (Marks), measured by the tail, not by |delta| (Taleb)?

Design decisions baked in here (confirmed with the user):
  * **Carry is assumed.** A cash-secured put ties up collateral that, held in
    T-bills, earns the risk-free rate. So the premium is the *incremental*
    risk premium you earn over cash for warehousing short-vol risk — NOT
    premium minus the risk-free rate. The blended return stacks carry +
    premium. (The old Tab-3 code subtracted rf from the CSP yield, which
    double-counted: it charged you the risk-free rate you are in fact still
    earning on the collateral.)
  * **Honest risk.** |delta| is the risk-neutral probability of assignment —
    it sees the *odds* of a loss, never its *size*. We add a tail-loss number:
    the fraction of collateral you'd lose in an adverse z-sigma move by expiry.

All functions are pure (numbers in, numbers/dicts out) so they unit-test
without a network (CLAUDE.md §8).
"""
from __future__ import annotations

import math

import numpy as np

from .options import csp_annualized_yield


def matched_rf(years: float, curve: dict[float, float]) -> float:
    """Risk-free rate for a given horizon, linearly interpolated from a sparse
    Treasury curve and flat-extrapolated past the ends.

    ``curve`` maps tenor-in-years -> annualised rate (decimal), e.g.
    ``{0.25: 0.052, 5: 0.043, 10: 0.043, 30: 0.045}``. This is what lets a
    2.5-year option be compared against a *2.5-year* Treasury rather than the
    13-week bill — an option is only "good" relative to the cash alternative of
    its own maturity.

    Returns NaN if the curve is empty / unusable.
    """
    pts = {float(k): float(v) for k, v in (curve or {}).items()
           if np.isfinite(k) and np.isfinite(v)}
    if not pts:
        return float("nan")
    xs = np.array(sorted(pts))
    ys = np.array([pts[x] for x in xs])
    y = float(np.interp(max(years, 0.0), xs, ys))  # np.interp clamps (flat) at both ends
    return y


def csp_metrics(
    mid: float,
    strike: float,
    dte: float,
    rf_matched: float,
    spot: float | None = None,
    iv: float | None = None,
    z: float = 2.0,
    assume_carry: bool = True,
) -> dict:
    """Capital-/risk-aware metrics for selling one cash-secured put.

    Parameters
    ----------
    mid, strike, dte : option mid price, strike, days to expiry.
    rf_matched       : risk-free rate of the option's *own* maturity (decimal).
    spot, iv         : underlying price and its (decimal) implied vol — needed
                       only for the tail-loss figure; omit for yield-only.
    z                : how many sigmas of adverse move to price the tail at
                       (2.0 ≈ a ~1-in-40 down move by expiry).
    assume_carry     : if True (default), the collateral earns ``rf_matched``,
                       so the premium is the excess over cash and the blended
                       return stacks carry + premium.

    Returns a dict (all annualised yields are decimals):

    ``collateral``        cash you must set aside per share (strike − mid).
    ``premium_yield``     annualised return from the premium, on collateral.
    ``rf_matched``        the matched-maturity risk-free rate (echoed back).
    ``excess_yield``      how much MORE per year than the matched Treasury —
                          the pay for taking the short-vol risk. Under carry
                          this equals ``premium_yield``; without carry it is
                          ``premium_yield − rf_matched``.
    ``blended_return``    total annualised return: carry + premium (carry on),
                          else just the premium.
    ``breakeven``         price at expiry below which the trade loses (= strike − mid).
    ``assignment_price``  the strike: where you're forced to buy the stock.
    ``tail_loss``         fraction of collateral lost in a z-sigma down move by
                          expiry — the Marks "how much could I lose" number,
                          NaN when spot/iv are unavailable.
    """
    collateral = strike - mid
    premium_yield = csp_annualized_yield(mid, strike, dte)
    rf_m = float(rf_matched) if np.isfinite(rf_matched) else 0.0

    if not np.isfinite(premium_yield):
        excess_yield = float("nan")
        blended_return = float("nan")
    elif assume_carry:
        excess_yield = premium_yield               # premium IS the excess over cash carry
        blended_return = rf_m + premium_yield
    else:
        excess_yield = premium_yield - rf_m        # idle collateral: must beat the bill itself
        blended_return = premium_yield

    breakeven = strike - mid
    tail_loss = float("nan")
    if (spot is not None and iv is not None and collateral > 0
            and np.isfinite(spot) and np.isfinite(iv) and iv > 0 and dte > 0):
        T = dte / 365.0
        tail_price = spot * math.exp(-z * iv * math.sqrt(T))  # lognormal down-move
        loss_per_share = max(breakeven - tail_price, 0.0)
        tail_loss = loss_per_share / collateral

    return {
        "collateral": collateral,
        "premium_yield": premium_yield,
        "rf_matched": rf_m,
        "excess_yield": excess_yield,
        "blended_return": blended_return,
        "breakeven": breakeven,
        "assignment_price": strike,
        "tail_loss": tail_loss,
    }


def short_put_risk(abs_delta: float, iv: float) -> float:
    """Honest risk coordinate for a short put on the cross-asset frontier, in
    the same annualised-vol units as the cash/Treasury/credit/equity rows.

    We use the underlying's implied vol — the volatility of the stock you'd be
    *forced to own* if assigned — rather than |delta|. |delta| is only the
    probability of that assignment; it cannot see the size of the loss, which
    is exactly Taleb's complaint about volatility-blind risk on short options.
    Falls back to |delta| when IV is missing.
    """
    if iv is not None and np.isfinite(iv) and iv > 0:
        return float(iv)
    if abs_delta is not None and np.isfinite(abs_delta):
        return float(min(max(abs_delta, 0.05), 0.95))
    return float("nan")


def long_option_edge(iv: float, realized_vol: float) -> float:
    """Rough expected-excess proxy for a convex (long-vol) candidate: how cheap
    its implied vol is versus the stock's realised vol (RV − IV).

    Positive ⇒ you're paying less vol than the stock actually delivers, the
    Taleb/PTJ "buy convexity when it's cheap" case. This is a heuristic edge for
    ranking candidates, not a forecast — the payoff's value is its convexity,
    which ``build_frontier`` already rewards.
    """
    if not (np.isfinite(iv) and np.isfinite(realized_vol)):
        return float("nan")
    return float(realized_vol - iv)
