"""
Opportunity Frontier — a trader/investor relative-value scanner.

Tab 1  Volatility & Options Frontier
       Pull VIX + option chains (yfinance), compute Greeks (Black-Scholes),
       fit each expiry's volatility skew, compare implied vol to the
       trailing-200d realised vol (the "200-day frontier"), and flag
       contracts that are RICH (sell candidates) or CHEAP (buy candidates)
       relative to their own surface and history.

Tab 2  Fundamentals Frontier
       Pull fundamentals for a peer group, z-score every metric within the
       group, build Quality / Value / Growth / Safety composites, and plot a
       Quality-vs-Value frontier that surfaces the outliers (high quality AND
       cheap) the way Greenblatt / Damodaran / Buffett-style screens do.

Run:  streamlit run app.py
Data: free via yfinance (delayed/EOD). See CLAUDE.md for paid upgrades.
NOT INVESTMENT ADVICE. Volatility understates tail risk — read the caveats.
"""

from __future__ import annotations
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq

try:
    import yfinance as yf
except Exception as e:  # pragma: no cover
    yf = None
    _YF_ERR = str(e)

st.set_page_config(page_title="Opportunity Frontier", layout="wide", page_icon="📈")

# --------------------------------------------------------------------------- #
#  Black-Scholes: price + Greeks, and implied vol solved from a market price   #
# --------------------------------------------------------------------------- #
def bs(S, K, T, r, sigma, kind="put", q=0.0):
    """Black-Scholes price and Greeks. theta is per-day, vega/rho per 1 vol/rate point."""
    out = {k: np.nan for k in ("price", "delta", "gamma", "theta", "vega", "rho")}
    if not all(np.isfinite([S, K, T, r, sigma])) or S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return out
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf, disc, dvd = norm.pdf(d1), math.exp(-r * T), math.exp(-q * T)
    out["gamma"] = dvd * pdf / (S * sigma * math.sqrt(T))
    out["vega"] = S * dvd * pdf * math.sqrt(T) / 100.0
    if kind == "call":
        out["price"] = S * dvd * norm.cdf(d1) - K * disc * norm.cdf(d2)
        out["delta"] = dvd * norm.cdf(d1)
        out["theta"] = (-S * dvd * pdf * sigma / (2 * math.sqrt(T)) - r * K * disc * norm.cdf(d2)
                        + q * S * dvd * norm.cdf(d1)) / 365.0
        out["rho"] = K * T * disc * norm.cdf(d2) / 100.0
    else:
        out["price"] = K * disc * norm.cdf(-d2) - S * dvd * norm.cdf(-d1)
        out["delta"] = -dvd * norm.cdf(-d1)
        out["theta"] = (-S * dvd * pdf * sigma / (2 * math.sqrt(T)) + r * K * disc * norm.cdf(-d2)
                        - q * S * dvd * norm.cdf(-d1)) / 365.0
        out["rho"] = -K * T * disc * norm.cdf(-d2) / 100.0
    return out


def implied_vol(price, S, K, T, r, kind="put", q=0.0):
    """Solve BS for sigma given a market price. Returns NaN if no solution in [0.1%, 500%]."""
    if not np.isfinite(price) or price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan
    f = lambda s: bs(S, K, T, r, s, kind, q)["price"] - price
    try:
        if f(1e-3) * f(5.0) > 0:
            return np.nan
        return brentq(f, 1e-3, 5.0, maxiter=100, xtol=1e-5)
    except Exception:
        return np.nan


def annualised(total_return_pct, days):
    """Compound-annualise a holding-period return."""
    if days <= 0:
        return np.nan
    return ((1 + total_return_pct / 100.0) ** (365.0 / days) - 1) * 100.0


# --------------------------------------------------------------------------- #
#  Cached data access (yfinance). TTLs keep it "fresh enough to act on".       #
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def get_riskfree():
    """13-week T-bill yield (^IRX, quoted in %) as the risk-free rate."""
    try:
        h = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        if len(h):
            return float(h.iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.04


@st.cache_data(ttl=900, show_spinner=False)
def get_vix():
    try:
        h = yf.Ticker("^VIX").history(period="1y")["Close"].dropna()
        last = float(h.iloc[-1])
        rank = float((h <= last).mean() * 100)  # percentile within trailing year
        return last, rank, h
    except Exception:
        return np.nan, np.nan, pd.Series(dtype=float)


@st.cache_data(ttl=900, show_spinner=False)
def get_underlying(ticker):
    """Spot + trailing realised vols (20/60/200d) from 1y of daily closes."""
    t = yf.Ticker(ticker)
    h = t.history(period="1y")["Close"].dropna()
    if len(h) < 30:
        return None
    rets = np.log(h / h.shift(1)).dropna()
    rv = {w: float(rets.tail(w).std() * math.sqrt(252) * 100) for w in (20, 60, 200)}
    try:
        spot = float(t.fast_info["lastPrice"])
    except Exception:
        spot = float(h.iloc[-1])
    return {"spot": spot, "rv": rv, "history": h}


@st.cache_data(ttl=900, show_spinner=False)
def get_chain(ticker, max_expiries, r):
    """Enriched option chain: own-computed IV + Greeks + yields, across the first N expiries."""
    info = get_underlying(ticker)
    if info is None:
        return pd.DataFrame()
    S, rv30 = info["spot"], info["rv"][60]
    t = yf.Ticker(ticker)
    expiries = list(t.options)[:max_expiries]
    rows = []
    today = datetime.now(timezone.utc).date()
    for exp in expiries:
        try:
            oc = t.option_chain(exp)
        except Exception:
            continue
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        if dte <= 0:
            continue
        T = dte / 365.0
        for kind, df in (("call", oc.calls), ("put", oc.puts)):
            for _, row in df.iterrows():
                K = float(row["strike"])
                bid, ask = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0)
                last = float(row.get("lastPrice", 0) or 0)
                mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
                if mid <= 0:
                    continue
                iv = implied_vol(mid, S, K, T, r, kind)
                if not np.isfinite(iv):
                    iv = float(row.get("impliedVolatility", np.nan))  # fall back to Yahoo's field
                g = bs(S, K, T, r, iv, kind) if np.isfinite(iv) else {}
                pct_otm = (S - K) / S * 100 if kind == "put" else (K - S) / S * 100
                # cash-secured put yield (annualised), the trader's core metric
                csp_yield = annualised(mid / (K - mid) * 100, dte) if (kind == "put" and K > mid) else np.nan
                rows.append({
                    "ticker": ticker, "kind": kind, "expiry": exp, "dte": dte, "strike": K,
                    "spot": round(S, 2), "bid": bid, "ask": ask, "mid": round(mid, 3),
                    "spread_%": round((ask - bid) / mid * 100, 1) if mid else np.nan,
                    "IV_%": round(iv * 100, 1) if np.isfinite(iv) else np.nan,
                    "RV60_%": round(rv30, 1),
                    "VRP_%": round(iv * 100 - rv30, 1) if np.isfinite(iv) else np.nan,  # implied minus realised
                    "log_m": math.log(K / S), "pct_otm": round(pct_otm, 1),
                    "delta": round(g.get("delta", np.nan), 3), "gamma": round(g.get("gamma", np.nan), 4),
                    "theta": round(g.get("theta", np.nan), 4), "vega": round(g.get("vega", np.nan), 4),
                    "open_int": int(row.get("openInterest", 0) or 0),
                    "volume": int(row.get("volume", 0) or 0),
                    "csp_yield_%": round(csp_yield, 2) if np.isfinite(csp_yield) else np.nan,
                })
    return pd.DataFrame(rows)


@st.cache_data(ttl=21600, show_spinner=False)
def get_fundamentals(ticker):
    """Pull a robust fundamentals panel from yfinance .info, with derived ratios."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None
    g = info.get
    mc, rev, fcf, ebitda = g("marketCap"), g("totalRevenue"), g("freeCashflow"), g("ebitda")
    debt, cash, pe = g("totalDebt"), g("totalCash"), g("trailingPE")
    safe = lambda a, b: (a / b) if (a is not None and b not in (None, 0)) else np.nan
    return {
        "ticker": ticker, "name": g("shortName", ticker), "mktcap": mc,
        "rev_growth_%": (g("revenueGrowth") or np.nan) * 100 if g("revenueGrowth") is not None else np.nan,
        "gross_%": (g("grossMargins") or np.nan) * 100 if g("grossMargins") is not None else np.nan,
        "oper_%": (g("operatingMargins") or np.nan) * 100 if g("operatingMargins") is not None else np.nan,
        "net_%": (g("profitMargins") or np.nan) * 100 if g("profitMargins") is not None else np.nan,
        "fcf_margin_%": safe(fcf, rev) * 100,
        "roe_%": (g("returnOnEquity") or np.nan) * 100 if g("returnOnEquity") is not None else np.nan,
        "roa_%": (g("returnOnAssets") or np.nan) * 100 if g("returnOnAssets") is not None else np.nan,
        "fcf_yield_%": safe(fcf, mc) * 100,
        "earnings_yield_%": safe(1.0, pe) * 100,
        "ev_ebitda": g("enterpriseToEbitda"),
        "pe": pe, "pb": g("priceToBook"),
        "net_debt_ebitda": safe((debt or 0) - (cash or 0), ebitda),
        "debt_equity": g("debtToEquity"),
        "current_ratio": g("currentRatio"),
    }


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * 0.0


# --------------------------------------------------------------------------- #
#  Analytics: skew residuals, richness score, yield-vs-risk frontier           #
# --------------------------------------------------------------------------- #
def add_skew_signal(df):
    """Per (ticker, expiry): fit IV vs log-moneyness (deg-2). Residual = rich(+)/cheap(-) vs own smile.
    Richness blends the standardised skew residual with the standardised vol-risk-premium."""
    df = df.copy()
    df["skew_resid"] = np.nan
    for (_, _), grp in df.groupby(["ticker", "expiry"]):
        v = grp.dropna(subset=["IV_%", "log_m"])
        if len(v) >= 4:
            coefs = np.polyfit(v["log_m"], v["IV_%"], 2)
            df.loc[v.index, "skew_resid"] = v["IV_%"] - np.polyval(coefs, v["log_m"])
    z_resid = df.groupby(["ticker", "expiry"])["skew_resid"].transform(zscore)
    z_vrp = df.groupby("ticker")["VRP_%"].transform(zscore)
    df["richness"] = np.nanmean(np.vstack([z_resid, z_vrp]), axis=0)
    df["signal"] = np.where(df["richness"] > 0.7, "RICH → sell",
                     np.where(df["richness"] < -0.7, "CHEAP → buy", "fair"))
    return df


def yield_frontier(puts, n_bins=12):
    """Upper envelope of annualised CSP yield vs |delta|. value_ratio>1 == above peer frontier."""
    p = puts.dropna(subset=["csp_yield_%", "delta"]).copy()
    if len(p) < 6:
        p["value_ratio"] = np.nan
        return p, pd.DataFrame()
    p["abs_delta"] = p["delta"].abs()
    p["bin"] = pd.cut(p["abs_delta"], bins=n_bins)
    frontier = p.groupby("bin", observed=True).agg(
        abs_delta=("abs_delta", "mean"), csp_yield_=("csp_yield_%", "max")).dropna()
    if len(frontier) >= 2:
        p["front_y"] = np.interp(p["abs_delta"], frontier["abs_delta"], frontier["csp_yield_"])
        p["value_ratio"] = (p["csp_yield_%"] / p["front_y"]).round(2)
    else:
        p["value_ratio"] = np.nan
    return p, frontier.rename(columns={"csp_yield_": "csp_yield_%"})


def fundamentals_panel(tickers):
    rows = [r for r in (get_fundamentals(t) for t in tickers) if r]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("ticker")
    # composites (peer-relative z-scores) — see CLAUDE.md for the framework attributions
    q = ["gross_%", "oper_%", "net_%", "fcf_margin_%", "roe_%", "roa_%"]
    val = ["fcf_yield_%", "earnings_yield_%"]
    inv_val = ["ev_ebitda", "pe", "pb"]              # cheaper is better -> invert
    saf_pos, saf_neg = ["current_ratio"], ["net_debt_ebitda", "debt_equity"]
    df["Quality"] = pd.concat([zscore(df[c]) for c in q if c in df], axis=1).mean(axis=1)
    df["Value"] = pd.concat(
        [zscore(df[c]) for c in val if c in df] + [-zscore(df[c]) for c in inv_val if c in df],
        axis=1).mean(axis=1)
    df["Growth"] = zscore(df["rev_growth_%"]) if "rev_growth_%" in df else 0.0
    df["Safety"] = pd.concat(
        [zscore(df[c]) for c in saf_pos if c in df] + [-zscore(df[c]) for c in saf_neg if c in df],
        axis=1).mean(axis=1)
    df["Composite"] = df["Quality"] + df["Value"] + 0.5 * df["Growth"].fillna(0) + 0.5 * df["Safety"].fillna(0)
    # Greenblatt "magic formula" style rank (approx): rank quality + rank cheapness, lower=better
    df["MF_rank"] = (df["roe_%"].rank(ascending=False) + df["earnings_yield_%"].rank(ascending=False))
    df["outlier"] = (df["Quality"] > 0.5) & (df["Value"] > 0.5)
    return df


# --------------------------------------------------------------------------- #
#  UI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    st.title("📈 Opportunity Frontier")
    st.caption("Relative-value scanner for options & fundamentals. Free data via yfinance (delayed). "
               "Not investment advice — and remember: **volatility understates tail risk.**")
    if yf is None:
        st.error(f"yfinance failed to import: {_YF_ERR}. Run `pip install -r requirements.txt`.")
        return

    r = get_riskfree()
    tab_vol, tab_fund = st.tabs(["⚡ Volatility & Options Frontier", "🏛️ Fundamentals Frontier"])

    # ---------------- Tab 1 ----------------
    with tab_vol:
        c1, c2, c3 = st.columns([3, 1, 1])
        tickers = [s.strip().upper() for s in c1.text_input(
            "Tickers (comma-separated)", "SPY, GOOG, TSLA").split(",") if s.strip()]
        n_exp = c2.number_input("Expiries / name", 1, 12, 4)
        side = c3.selectbox("Show", ["puts", "calls", "both"])
        st.caption(f"Risk-free (13wk T-bill): {r*100:.2f}%  ·  data cached ~15 min")

        vix, vix_rank, vix_h = get_vix()
        m1, m2 = st.columns(2)
        m1.metric("VIX", f"{vix:.2f}" if np.isfinite(vix) else "n/a",
                  f"{vix_rank:.0f}th pct (1y)" if np.isfinite(vix_rank) else "")
        if np.isfinite(vix_rank):
            m2.info("Low VIX rank ⇒ optionality is **cheap**: a poor time to *sell* premium, a good time to *buy* "
                    "convexity. High rank ⇒ the reverse. (Sell insurance when it's dear, not cheap.)")

        if st.button("🔄 Refresh data"):
            st.cache_data.clear()
            st.rerun()

        with st.spinner("Pulling chains & computing Greeks…"):
            chain = pd.concat([get_chain(t, int(n_exp), r) for t in tickers], ignore_index=True) \
                if tickers else pd.DataFrame()

        if chain.empty:
            st.warning("No option data returned. Check tickers, or Yahoo may be rate-limiting — try Refresh.")
        else:
            chain = add_skew_signal(chain)
            if side != "both":
                view = chain[chain["kind"] == side[:-1]]
            else:
                view = chain

            st.subheader("Implied vs realised volatility (the vol-risk-premium read)")
            atm = (chain.assign(absm=chain["log_m"].abs()).sort_values("absm")
                   .groupby("ticker").head(6).groupby("ticker")
                   .agg(ATM_IV_=("IV_%", "median"), RV60_=("RV60_%", "first")).reset_index())
            atm["VRP_%"] = (atm["ATM_IV_"] - atm["RV60_"]).round(1)
            atm = atm.rename(columns={"ATM_IV_": "ATM_IV_%", "RV60_": "RV60_%"})
            st.dataframe(atm, hide_index=True, use_container_width=True)

            st.subheader("Yield-vs-risk frontier (cash-secured puts)")
            st.caption("Each dot is a put. Up = more annualised premium; right = more risk (|delta|). "
                       "Dots **on/above the line** pay more than peers for the same risk.")
            puts_scored, front = yield_frontier(chain[chain["kind"] == "put"])
            if not puts_scored.empty and "value_ratio" in puts_scored:
                fig = px.scatter(
                    puts_scored, x="abs_delta", y="csp_yield_%", color="ticker",
                    size=puts_scored["open_int"].clip(1, 5000), hover_data=["expiry", "strike", "IV_%", "VRP_%"],
                    labels={"abs_delta": "|delta| (risk)", "csp_yield_%": "annualised CSP yield %"})
                if not front.empty:
                    fig.add_trace(go.Scatter(x=front["abs_delta"], y=front["csp_yield_%"],
                                             mode="lines+markers", name="frontier",
                                             line=dict(dash="dash", color="black")))
                fig.add_hline(y=r * 100, line_dash="dot", annotation_text="risk-free")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Volatility skew (pick a name & expiry)")
            sc1, sc2 = st.columns(2)
            stk = sc1.selectbox("Ticker", sorted(chain["ticker"].unique()))
            exp = sc2.selectbox("Expiry", sorted(chain[chain["ticker"] == stk]["expiry"].unique()))
            sk = chain[(chain["ticker"] == stk) & (chain["expiry"] == exp)].sort_values("strike")
            if not sk.empty:
                figk = px.scatter(sk, x="strike", y="IV_%", color="kind", hover_data=["mid", "delta", "richness"],
                                  labels={"IV_%": "implied vol %"})
                figk.add_vline(x=sk["spot"].iloc[0], line_dash="dot", annotation_text="spot")
                st.plotly_chart(figk, use_container_width=True)

            st.subheader("🎯 Opportunities (sorted by richness)")
            st.caption("RICH = implied vol high vs the contract's own smile & vs realised → **sell** candidate. "
                       "CHEAP = the reverse → **buy** candidate. Watch the spread_% and open_int columns: "
                       "thin, wide contracts (low OI, high spread_%) can't be exited cheaply.")
            cols = ["ticker", "kind", "expiry", "dte", "strike", "pct_otm", "mid", "IV_%", "RV60_%",
                    "VRP_%", "delta", "csp_yield_%", "spread_%", "open_int", "richness", "signal"]
            flagged = view[view["signal"] != "fair"].sort_values("richness", ascending=False)
            st.dataframe((flagged if not flagged.empty else view)[cols],
                         hide_index=True, use_container_width=True, height=380)
            st.download_button("⬇ Download full chain (CSV)", view.to_csv(index=False),
                               "option_frontier.csv")

    # ---------------- Tab 2 ----------------
    with tab_fund:
        peers = [s.strip().upper() for s in st.text_input(
            "Peer group (comma-separated)", "GOOG, MSFT, AMZN, META, AAPL").split(",") if s.strip()]
        st.caption("Metrics are z-scored **within this peer group**, so the comparison is relative. "
                   "Add or remove names to reframe the peer set. Data cached ~6h.")
        if st.button("🔄 Refresh fundamentals"):
            st.cache_data.clear()
            st.rerun()

        with st.spinner("Pulling fundamentals…"):
            fdf = fundamentals_panel(peers) if peers else pd.DataFrame()

        if fdf.empty:
            st.warning("No fundamentals returned. Check tickers, or Yahoo may be rate-limiting.")
        else:
            st.subheader("Quality-vs-Value frontier")
            st.caption("Top-right = **high quality AND cheap** — the Greenblatt/Buffett sweet spot, and where "
                       "outliers (highlighted) live. Bubble = market cap, colour = revenue growth.")
            plot = fdf.reset_index()
            fig = px.scatter(plot, x="Quality", y="Value", text="ticker",
                             size=plot["mktcap"].fillna(plot["mktcap"].median()).clip(lower=1),
                             color="Growth", color_continuous_scale="RdYlGn",
                             hover_data=["roe_%", "fcf_yield_%", "net_debt_ebitda", "rev_growth_%"])
            fig.update_traces(textposition="top center")
            fig.add_hline(y=0, line_dash="dot"); fig.add_vline(x=0, line_dash="dot")
            for _, row in plot[plot["outlier"]].iterrows():
                fig.add_annotation(x=row["Quality"], y=row["Value"], text="★ outlier",
                                   showarrow=True, arrowhead=2, ay=-30)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Composite scores (peer-relative, z)")
            comp = fdf[["Quality", "Value", "Growth", "Safety", "Composite", "MF_rank", "outlier"]] \
                .round(2).sort_values("Composite", ascending=False)
            st.dataframe(comp, use_container_width=True)

            outliers = comp[comp["outlier"]].index.tolist()
            if outliers:
                st.success("★ Outliers (high quality **and** cheap vs peers): " + ", ".join(outliers))
            best_mf = fdf["MF_rank"].idxmin()
            st.info(f"Magic-Formula-style top pick (best blend of return-on-equity + earnings yield): **{best_mf}**")

            st.subheader("Raw metric panel")
            metric_cols = ["name", "rev_growth_%", "gross_%", "oper_%", "net_%", "fcf_margin_%",
                           "roe_%", "roa_%", "fcf_yield_%", "earnings_yield_%", "ev_ebitda", "pe",
                           "net_debt_ebitda", "debt_equity", "current_ratio"]
            st.dataframe(fdf[[c for c in metric_cols if c in fdf]].round(2), use_container_width=True)
            st.download_button("⬇ Download fundamentals (CSV)", fdf.to_csv(), "fundamentals_frontier.csv")

    st.divider()
    st.caption("⚠️ Educational tool, not advice. The yield-vs-risk frontier plots |delta| as risk — but "
               "a deep-OTM short put's real risk is a rare crash, which |delta| barely sees. Treat 'RICH→sell' "
               "as 'where the premium is', not 'free money'. Confirm liquidity (open interest, spread) before acting.")


if __name__ == "__main__":
    main()
