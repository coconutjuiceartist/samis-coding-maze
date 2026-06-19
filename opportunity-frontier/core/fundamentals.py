"""Fundamentals frontier for Tab 2: peer-relative z-scored composites, the
Magic-Formula rank, and the roadmap upgrades — true ROIC / ROIC-WACC
(Damodaran) and the Piotroski F-score.

Everything is relative to the supplied peer set. Pure functions only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import rank_best_first, zscore

# Metrics where a *higher* raw value is better vs peers.
HIGHER_BETTER = [
    "rev_growth", "gross_margin", "op_margin", "net_margin", "fcf_margin",
    "roe", "roa", "roic", "fcf_yield", "earnings_yield", "current_ratio",
]
# Metrics where a *lower* raw value is better (valuation multiples, leverage).
LOWER_BETTER = [
    "ev_ebitda", "pe", "pb", "net_debt_ebitda", "debt_equity",
]


def metrics_from_info(info: dict) -> dict:
    """Pull the Tab-2 metric panel from a yfinance ``.info`` dict.

    Missing fields come back as NaN. yfinance reports debtToEquity as a
    percentage and trailingPE directly; we normalise here.
    """
    def g(key):
        v = info.get(key) if info else None
        try:
            v = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return v if np.isfinite(v) else float("nan")

    revenue = g("totalRevenue")
    fcf = g("freeCashflow")
    mktcap = g("marketCap")
    ebitda = g("ebitda")
    total_debt = g("totalDebt")
    cash = g("totalCash")
    trailing_pe = g("trailingPE")

    fcf_margin = fcf / revenue if np.isfinite(fcf) and np.isfinite(revenue) and revenue else float("nan")
    fcf_yield = fcf / mktcap if np.isfinite(fcf) and np.isfinite(mktcap) and mktcap else float("nan")
    earnings_yield = 1.0 / trailing_pe if np.isfinite(trailing_pe) and trailing_pe else float("nan")
    net_debt_ebitda = (
        (total_debt - cash) / ebitda
        if np.isfinite(total_debt) and np.isfinite(cash) and np.isfinite(ebitda) and ebitda
        else float("nan")
    )
    debt_equity = g("debtToEquity")
    if np.isfinite(debt_equity):
        debt_equity = debt_equity / 100.0  # yahoo reports as a percentage

    return {
        "name": info.get("shortName") if info else None,
        "sector": info.get("sector") if info else None,
        "industry": info.get("industry") if info else None,
        "market_cap": mktcap,
        "rev_growth": g("revenueGrowth"),
        "gross_margin": g("grossMargins"),
        "op_margin": g("operatingMargins"),
        "net_margin": g("profitMargins"),
        "fcf_margin": fcf_margin,
        "roe": g("returnOnEquity"),
        "roa": g("returnOnAssets"),
        "fcf_yield": fcf_yield,
        "earnings_yield": earnings_yield,
        "ev_ebitda": g("enterpriseToEbitda"),
        "pe": trailing_pe,
        "pb": g("priceToBook"),
        "net_debt_ebitda": net_debt_ebitda,
        "debt_equity": debt_equity,
        "current_ratio": g("currentRatio"),
    }


def compute_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score within the peer set and build the Quality/Value/Growth/Safety
    composites and the overall Composite + Magic-Formula rank (CLAUDE.md §5).
    """
    out = df.copy()

    def Z(col):
        return zscore(out[col]) if col in out.columns else pd.Series(0.0, index=out.index)

    quality = (Z("gross_margin") + Z("op_margin") + Z("net_margin")
               + Z("fcf_margin") + Z("roe") + Z("roa")) / 6.0
    # ROIC strengthens Quality when available.
    if "roic" in out.columns and out["roic"].notna().any():
        quality = (quality * 6.0 + Z("roic")) / 7.0

    value = (Z("fcf_yield") + Z("earnings_yield")
             - Z("ev_ebitda") - Z("pe") - Z("pb")) / 5.0
    growth = Z("rev_growth")
    safety = (Z("current_ratio") - Z("net_debt_ebitda") - Z("debt_equity")) / 3.0

    out["Quality"] = quality
    out["Value"] = value
    out["Growth"] = growth
    out["Safety"] = safety
    out["Composite"] = quality + value + 0.5 * growth + 0.5 * safety

    # Magic Formula: rank on returns on capital (ROIC if present, else ROE) and
    # earnings yield; lower combined rank == better.
    roc_col = "roic" if ("roic" in out.columns and out["roic"].notna().any()) else "roe"
    mf = rank_best_first(out[roc_col]) + rank_best_first(out["earnings_yield"])
    out["MF_rank"] = mf.rank(method="min").astype("Int64")

    # Outlier flag: high quality AND cheap (both z > 0.5) => the ★ names.
    out["star"] = (out["Quality"] > 0.5) & (out["Value"] > 0.5)
    return out


def compute_roic(
    ebit: float,
    tax_rate: float,
    total_debt: float,
    total_equity: float,
    cash: float,
) -> float:
    """True ROIC = NOPAT / invested capital (Damodaran, CLAUDE.md §5 roadmap 1).

    NOPAT = EBIT * (1 - tax_rate); invested capital = debt + equity - cash.
    """
    if not all(np.isfinite(x) for x in (ebit, tax_rate, total_debt, total_equity, cash)):
        return float("nan")
    invested = total_debt + total_equity - cash
    if invested <= 0:
        return float("nan")
    nopat = ebit * (1.0 - tax_rate)
    return nopat / invested


def compute_wacc(
    beta: float,
    rf: float,
    erp: float,
    cost_of_debt: float,
    tax_rate: float,
    equity_value: float,
    debt_value: float,
) -> float:
    """WACC from CAPM cost of equity and after-tax cost of debt, weighted by
    market values (Damodaran). Returns NaN on degenerate inputs.
    """
    if not all(np.isfinite(x) for x in (beta, rf, erp, cost_of_debt, tax_rate, equity_value, debt_value)):
        return float("nan")
    total = equity_value + debt_value
    if total <= 0:
        return float("nan")
    cost_equity = rf + beta * erp
    we, wd = equity_value / total, debt_value / total
    return we * cost_equity + wd * cost_of_debt * (1.0 - tax_rate)


def piotroski_fscore(curr: dict, prev: dict) -> int:
    """Piotroski 9-point fundamental-health score (CLAUDE.md §5 roadmap 2).

    ``curr``/``prev`` are dicts of statement values for the two most recent
    fiscal years. Unknown components simply score 0. Keys used:
        net_income, op_cash_flow, roa, total_assets, long_term_debt,
        current_ratio, shares, gross_margin, asset_turnover
    """
    def gc(d, k):
        try:
            v = float(d.get(k))
            return v if np.isfinite(v) else float("nan")
        except (TypeError, ValueError, AttributeError):
            return float("nan")

    score = 0
    ni = gc(curr, "net_income")
    cfo = gc(curr, "op_cash_flow")
    roa_c, roa_p = gc(curr, "roa"), gc(prev, "roa")

    # Profitability (4)
    if np.isfinite(ni) and ni > 0:
        score += 1
    if np.isfinite(cfo) and cfo > 0:
        score += 1
    if np.isfinite(roa_c) and np.isfinite(roa_p) and roa_c > roa_p:
        score += 1
    if np.isfinite(cfo) and np.isfinite(ni) and cfo > ni:  # accruals
        score += 1

    # Leverage / liquidity / dilution (3)
    ltd_c, ltd_p = gc(curr, "long_term_debt"), gc(prev, "long_term_debt")
    if np.isfinite(ltd_c) and np.isfinite(ltd_p) and ltd_c < ltd_p:
        score += 1
    cr_c, cr_p = gc(curr, "current_ratio"), gc(prev, "current_ratio")
    if np.isfinite(cr_c) and np.isfinite(cr_p) and cr_c > cr_p:
        score += 1
    sh_c, sh_p = gc(curr, "shares"), gc(prev, "shares")
    if np.isfinite(sh_c) and np.isfinite(sh_p) and sh_c <= sh_p:
        score += 1

    # Operating efficiency (2)
    gm_c, gm_p = gc(curr, "gross_margin"), gc(prev, "gross_margin")
    if np.isfinite(gm_c) and np.isfinite(gm_p) and gm_c > gm_p:
        score += 1
    at_c, at_p = gc(curr, "asset_turnover"), gc(prev, "asset_turnover")
    if np.isfinite(at_c) and np.isfinite(at_p) and at_c > at_p:
        score += 1

    return score
