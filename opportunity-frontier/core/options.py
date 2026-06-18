"""Black-Scholes pricing, Greeks, an implied-vol solver, and per-contract
enrichment of an option chain.

MATH IS SACRED (CLAUDE.md §8). The reference numbers, locked by tests:

    bs(100, 100, 1.0, 0.0, 0.2, kind="call")  -> price ~= 7.97, delta ~= 0.540

Conventions for the returned Greeks:
    delta  : per $1 move in spot (dimensionless, signed)
    gamma  : per $1 move in spot
    vega   : per 1 percentage-point change in vol (i.e. raw vega / 100)
    theta  : per calendar day (i.e. annual theta / 365), negative for longs
    rho    : per 1 percentage-point change in the rate (raw rho / 100)
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

TRADING_DAYS = 252
_N = norm.cdf
_n = norm.pdf

# Above this, an "implied vol" is almost certainly a stale/garbage Yahoo field or
# a non-arbitrageable quote, not a real vol. 8.0 == 800% leaves headroom for
# genuinely wild small-caps/meme names while still rejecting Yahoo's nonsense
# (which typically prints 1000%+ or near-zero on illiquid strikes).
MAX_PLAUSIBLE_IV = 8.0


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
) -> Dict[str, float]:
    """Black-Scholes-Merton price and Greeks for a European option.

    Parameters
    ----------
    S, K : spot and strike
    T    : time to expiry in years
    r    : continuously-compounded risk-free rate (decimal, e.g. 0.05)
    sigma: volatility (decimal, e.g. 0.20)
    q    : continuous dividend yield (decimal)
    kind : "call" or "put"

    Returns a dict with price, delta, gamma, vega, theta (see module docstring
    for units). Degrades to intrinsic value with zeroed Greeks when the inputs
    are degenerate (T<=0 or sigma<=0), so a stale/expired row never crashes.
    """
    kind = kind.lower()
    is_call = kind.startswith("c")

    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        delta = (1.0 if S > K else 0.0) if is_call else (-1.0 if S < K else 0.0)
        return {"price": intrinsic, "delta": delta, "gamma": 0.0, "vega": 0.0,
                "theta": 0.0, "rho": 0.0}

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrtT = math.sqrt(T)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    pdf_d1 = _n(d1)

    if is_call:
        price = S * disc_q * _N(d1) - K * disc_r * _N(d2)
        delta = disc_q * _N(d1)
        theta_annual = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
            - r * K * disc_r * _N(d2)
            + q * S * disc_q * _N(d1)
        )
        rho = K * T * disc_r * _N(d2) / 100.0
    else:
        price = K * disc_r * _N(-d2) - S * disc_q * _N(-d1)
        delta = disc_q * (_N(d1) - 1.0)
        theta_annual = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrtT)
            + r * K * disc_r * _N(-d2)
            - q * S * disc_q * _N(-d1)
        )
        rho = -K * T * disc_r * _N(-d2) / 100.0

    gamma = disc_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * disc_q * pdf_d1 * sqrtT / 100.0  # per 1 vol point
    theta = theta_annual / 365.0  # per calendar day

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    kind: str = "call",
    lo: float = 1e-4,
    hi: float = 5.0,
) -> float:
    """Solve for the BS implied vol that reproduces ``price``.

    Returns NaN when the price is below intrinsic / non-arbitrageable or the
    solver cannot bracket a root (illiquid, garbage quote). Callers should fall
    back to the provider's IV field in that case (CLAUDE.md §3, §4).
    """
    if not np.isfinite(price) or price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    is_call = kind.lower().startswith("c")
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    # Lower/upper no-arbitrage bounds on the option price.
    if is_call:
        intrinsic = max(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:
        intrinsic = max(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r
    if price < intrinsic - 1e-8 or price > upper + 1e-8:
        return float("nan")

    def objective(sig: float) -> float:
        return bs(S, K, T, r, sig, q, kind)["price"] - price

    try:
        return float(brentq(objective, lo, hi, maxiter=100, xtol=1e-6))
    except (ValueError, RuntimeError):
        return float("nan")


def csp_annualized_yield(mid: float, strike: float, dte: float) -> float:
    """Annualised cash-secured-put yield = mid / (strike - mid) * 365 / DTE.

    This is the capital-efficiency number from CLAUDE.md §4: premium collected
    against the cash you must set aside (strike less premium), annualised.
    """
    denom = strike - mid
    if denom <= 0 or dte <= 0 or not np.isfinite(mid):
        return float("nan")
    return mid / denom * 365.0 / dte


def enrich_chain(
    chain: pd.DataFrame,
    spot: float,
    r: float,
    expiry_dte: float,
    kind: str,
    realized_vol: float | None = None,
    q: float = 0.0,
    max_iv: float = MAX_PLAUSIBLE_IV,
) -> pd.DataFrame:
    """Enrich a raw yfinance option chain with mids, own-solved IV, Greeks and
    the per-contract signals described in CLAUDE.md §4.

    ``chain`` is expected to have columns: strike, bid, ask, lastPrice, volume,
    openInterest, impliedVolatility. Missing columns degrade gracefully.

    Two robustness rules keep the downstream plots/tables sane:
      * IV outside (0, ``max_iv``] is treated as missing — this is what stops
        garbage Yahoo IV (e.g. 15.0 == 1500%) from polluting the surface.
      * the cash-secured-put yield is computed only for OTM/ATM puts
        (strike <= spot); for deep-ITM puts ``mid`` approaches ``strike`` and the
        yield formula explodes, which is meaningless as a CSP candidate anyway.
    """
    if chain is None or len(chain) == 0:
        return pd.DataFrame()

    df = chain.copy()
    T = max(expiry_dte, 0.0) / 365.0
    is_call = kind.lower().startswith("c")

    def col(name, default=np.nan):
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    bid = pd.to_numeric(col("bid"), errors="coerce")
    ask = pd.to_numeric(col("ask"), errors="coerce")
    last = pd.to_numeric(col("lastPrice"), errors="coerce")
    df["bid"] = bid
    df["ask"] = ask
    df["two_sided"] = (bid > 0) & (ask > 0)  # tradeable (can actually be exited)
    df["mid"] = np.where((bid > 0) & (ask > 0), (bid + ask) / 2.0, last)
    df["spread_pct"] = np.where(
        (df["mid"] > 0) & (ask >= bid) & (ask > 0), (ask - bid) / df["mid"] * 100.0, np.nan
    )
    df["open_int"] = pd.to_numeric(col("openInterest"), errors="coerce")
    df["volume"] = pd.to_numeric(col("volume"), errors="coerce")
    strike = pd.to_numeric(col("strike"), errors="coerce")
    df["dte"] = expiry_dte
    df["log_moneyness"] = np.log(strike / spot)

    # %OTM: how far out-of-the-money the strike sits.
    if is_call:
        df["pct_otm"] = (strike / spot - 1.0) * 100.0
    else:
        df["pct_otm"] = (1.0 - strike / spot) * 100.0

    yahoo_iv = pd.to_numeric(col("impliedVolatility"), errors="coerce").to_numpy()
    solved_iv, used_solver = [], []
    for m, k in zip(df["mid"].to_numpy(), strike.to_numpy()):
        iv = implied_vol(m, spot, k, T, r, q, kind)
        solved_iv.append(iv)
        used_solver.append(np.isfinite(iv))
    solved_iv = np.array(solved_iv, dtype=float)
    used_solver = np.array(used_solver, dtype=bool)
    # Own-solved IV first (CLAUDE.md §3); only fall back to Yahoo's field.
    iv = np.where(used_solver, solved_iv, yahoo_iv)
    # Reject implausible IV so garbage values never reach the plots/table.
    plausible = np.isfinite(iv) & (iv > 0) & (iv <= max_iv)
    df["iv"] = np.where(plausible, iv, np.nan)
    df["iv_source"] = np.where(used_solver & plausible, "solved",
                               np.where(plausible, "yahoo", "n/a"))

    deltas, gammas, thetas, vegas, rhos = [], [], [], [], []
    for k, iv in zip(strike.to_numpy(), df["iv"].to_numpy()):
        g = bs(spot, k, T, r, iv if np.isfinite(iv) else 0.0, q, kind)
        deltas.append(g["delta"]); gammas.append(g["gamma"])
        thetas.append(g["theta"]); vegas.append(g["vega"]); rhos.append(g["rho"])
    df["delta"] = deltas
    df["gamma"] = gammas
    df["theta"] = thetas
    df["vega"] = vegas
    df["rho"] = rhos
    df["abs_delta"] = np.abs(df["delta"])

    # Annualised cash-secured-put yield — OTM/ATM puts only (strike <= spot).
    if not is_call:
        raw_csp = np.array([
            csp_annualized_yield(m, k, expiry_dte)
            for m, k in zip(df["mid"].to_numpy(), strike.to_numpy())
        ])
        otm = strike.to_numpy() <= spot
        df["csp_yield"] = np.where(otm, raw_csp, np.nan)
    else:
        df["csp_yield"] = np.nan

    # Volatility risk premium: implied minus trailing realised, and the ratio.
    if realized_vol is not None and np.isfinite(realized_vol) and realized_vol > 0:
        df["vrp"] = df["iv"] - realized_vol
        df["iv_rv_ratio"] = df["iv"] / realized_vol  # >1 vol rich, <1 vol cheap
    else:
        df["vrp"] = np.nan
        df["iv_rv_ratio"] = np.nan

    return df


def liquidity_filter(
    df: pd.DataFrame,
    min_open_int: int = 0,
    require_two_sided: bool = True,
    max_spread_pct: float | None = None,
    need_iv: bool = True,
) -> pd.DataFrame:
    """Keep only contracts that are tradeable and have a usable IV.

    Liquidity is a first-class screen (CLAUDE.md §2, Blankfein): a desk only
    warehouses risk it can exit. Drops rows with no plausible IV, no positive
    mid, no two-sided market, thin open interest, or a too-wide spread.
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    mask = pd.Series(True, index=df.index)
    if need_iv and "iv" in df.columns:
        mask &= df["iv"].notna()
    if "mid" in df.columns:
        mask &= pd.to_numeric(df["mid"], errors="coerce") > 0
    if require_two_sided and "two_sided" in df.columns:
        mask &= df["two_sided"].fillna(False)
    if min_open_int and "open_int" in df.columns:
        mask &= pd.to_numeric(df["open_int"], errors="coerce").fillna(0) >= min_open_int
    if max_spread_pct is not None and "spread_pct" in df.columns:
        mask &= pd.to_numeric(df["spread_pct"], errors="coerce").fillna(np.inf) <= max_spread_pct
    return df[mask].copy()
