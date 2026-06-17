"""Cross-tab 'available risk frontier' (CLAUDE.md §6, the north-star).

Places every candidate use of capital — cash, Treasuries, IG/HY credit,
equities (earnings yield), screened options, and the user's own positions —
onto one expected-excess-return vs risk chart, then penalises concave
(short-vol) payoffs and rewards convex ones beyond what a volatility-based
Sharpe implies (Taleb / Marks).

Pure functions; the app feeds in rows it has already assembled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# How much we tilt the score for payoff shape, on top of plain Sharpe.
CONVEXITY_BONUS = 0.15
CONCAVITY_PENALTY = 0.25


@dataclass
class FrontierRow:
    label: str
    asset_class: str
    expected_excess_return: float  # over the risk-free rate, decimal
    risk: float                    # annualised vol / loss proxy, decimal
    convexity: int = 0             # +1 convex (long option), -1 concave (short), 0 linear
    note: str = ""


def cash_row(rf: float) -> FrontierRow:
    return FrontierRow("Cash (risk-free)", "cash", 0.0, 0.0, 0, "the floor every trade competes with")


def treasury_row(yield_: float, rf: float, label="10y Treasury", risk=0.07) -> FrontierRow:
    return FrontierRow(label, "rates", yield_ - rf, risk, 0)


def credit_row(yield_: float, rf: float, label, risk) -> FrontierRow:
    return FrontierRow(label, "credit", yield_ - rf, risk, 0)


def equity_row(earnings_yield: float, rf: float, vol: float, label) -> FrontierRow:
    return FrontierRow(label, "equity", earnings_yield - rf, vol, 0)


def short_option_row(annual_yield: float, rf: float, risk: float, label) -> FrontierRow:
    """A premium-selling candidate: concave payoff (pennies in front of a
    steamroller), so it carries the concavity flag."""
    return FrontierRow(label, "short_option", annual_yield - rf, risk, -1,
                       "short vol: negatively skewed, tail risk understated by |delta|")


def long_option_row(expected_excess: float, risk: float, label) -> FrontierRow:
    return FrontierRow(label, "long_option", expected_excess, risk, +1,
                       "long convexity: capped loss, fat right tail")


def build_frontier(rows: list[FrontierRow]) -> pd.DataFrame:
    """Score and rank candidates. ``shape_adj_score`` is a Sharpe-like
    excess-return-per-unit-risk, nudged by payoff convexity.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([r.__dict__ for r in rows])

    eps = 1e-6
    base = df["expected_excess_return"] / (df["risk"] + eps)
    adj = np.where(df["convexity"] > 0, 1.0 + CONVEXITY_BONUS,
                   np.where(df["convexity"] < 0, 1.0 - CONCAVITY_PENALTY, 1.0))
    df["sharpe_like"] = base
    df["shape_adj_score"] = base * adj
    # Cash sits at the origin; give it a neutral score so it sorts sensibly.
    df.loc[df["asset_class"] == "cash", ["sharpe_like", "shape_adj_score"]] = 0.0
    return df.sort_values("shape_adj_score", ascending=False).reset_index(drop=True)


def efficient_envelope(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-left envelope: the best expected excess return seen at or below
    each risk level — the realised 'frontier' line to plot behind the points.
    """
    if df.empty:
        return df
    s = df.sort_values("risk")
    best, keep = -np.inf, []
    for _, row in s.iterrows():
        if row["expected_excess_return"] >= best:
            best = row["expected_excess_return"]
            keep.append(row)
    return pd.DataFrame(keep)
