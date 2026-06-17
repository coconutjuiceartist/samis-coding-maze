"""Opportunity Frontier — a relative-value scanner.

Ranks opportunities against each other rather than judging them in isolation
(CLAUDE.md §1). Three surfaces:
    Tab 1  Volatility & Options Frontier  (where is premium rich vs cheap)
    Tab 2  Fundamentals Frontier          (high quality AND cheap vs peers)
    Tab 3  Available Risk Frontier         (everything on one risk/return chart)

Run:  streamlit run app.py
Data access is behind providers/ (CLAUDE.md §3); caching + freshness live here.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import fundamentals as F
from core import frontier as FR
from core import iv_store
from core import options as O
from core import peers as PEERS
from core import vol as V
from providers import edgar, get_provider

st.set_page_config(page_title="Opportunity Frontier", layout="wide")

PROVIDER = get_provider()
SPOT_TTL = 15 * 60       # options / spot ~15 min (CLAUDE.md §8)
FUND_TTL = 6 * 60 * 60   # fundamentals ~6 h

TAIL_CAVEAT = (
    "⚠️ **|delta| is not the real risk.** A short option is negatively skewed "
    "(pennies in front of a steamroller). 'RICH → sell' marks *where the "
    "premium is*, not free money. Confirm liquidity (spread %, open interest) "
    "and size for the tail before acting. This is a screen, never advice."
)


# ----------------------------------------------------------------------------
# Cached data access (thin wrappers over the provider so TTLs live in one place)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=SPOT_TTL, show_spinner=False)
def c_spot(ticker: str) -> float:
    return PROVIDER.spot(ticker)


@st.cache_data(ttl=SPOT_TTL, show_spinner=False)
def c_history(ticker: str, period="1y") -> pd.DataFrame:
    return PROVIDER.history(ticker, period=period)


@st.cache_data(ttl=SPOT_TTL, show_spinner=False)
def c_expiries(ticker: str) -> list[str]:
    return PROVIDER.option_expiries(ticker)


@st.cache_data(ttl=SPOT_TTL, show_spinner=False)
def c_chain(ticker: str, expiry: str):
    return PROVIDER.chain(ticker, expiry)


@st.cache_data(ttl=FUND_TTL, show_spinner=False)
def c_fundamentals(ticker: str) -> dict:
    return PROVIDER.fundamentals(ticker)


@st.cache_data(ttl=SPOT_TTL, show_spinner=False)
def c_rates_vix():
    return PROVIDER.risk_free_rate(), PROVIDER.vix()


@st.cache_data(ttl=FUND_TTL, show_spinner=False)
def c_edgar(ticker: str) -> dict:
    """SEC EDGAR companyfacts bundle (clean ROIC inputs + F-score components).
    Fails soft to {'available': False} if EDGAR hosts are not allow-listed."""
    return edgar.fundamentals_bundle(ticker)


def parse_tickers(text: str) -> list[str]:
    seen, out = set(), []
    for raw in text.replace("\n", ",").split(","):
        t = raw.strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def dte_of(expiry: str) -> float:
    try:
        d = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
        return max((d - dt.date.today()).days, 0) + 0.5  # half-day so today != 0
    except Exception:
        return float("nan")


def stamp():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------
# Header: provider, refresh, risk-free + VIX regime
# ----------------------------------------------------------------------------
st.title("Opportunity Frontier")
st.caption("A trade is only as good as the best alternative use of the same capital and risk.")

with st.sidebar:
    st.subheader("Data")
    st.write(f"Provider: **{PROVIDER.name}**")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    rfr_window = st.selectbox("Realised-vol lookback (days)", [20, 60, 200], index=1)
    st.caption(f"Loaded {stamp()}")

rf, (vix_level, vix_pct) = c_rates_vix()
rf = rf if np.isfinite(rf) else 0.045  # safe default if ^IRX is unavailable

c1, c2, c3 = st.columns(3)
c1.metric("Risk-free (13w T-bill)", f"{rf*100:.2f}%")
c2.metric("VIX", f"{vix_level:.1f}" if np.isfinite(vix_level) else "n/a",
          f"{vix_pct:.0f}th pct" if np.isfinite(vix_pct) else None)
c3.metric("As of", dt.date.today().isoformat())
st.info(V.vix_regime(vix_level, vix_pct))

tab1, tab2, tab3 = st.tabs(
    ["📈 Volatility & Options Frontier", "🏛️ Fundamentals Frontier", "🧭 Available Risk Frontier"]
)


# ============================================================================
# TAB 1 — Volatility & Options Frontier
# ============================================================================
def build_options_table(tickers, expiries_per_name, side, rv_window,
                        min_oi=0, require_two_sided=True, max_spread=None):
    rows = []
    realized_by_ticker = {}
    spot_by_ticker = {}
    for t in tickers:
        spot = c_spot(t)
        if not np.isfinite(spot) or spot <= 0:
            continue
        spot_by_ticker[t] = spot
        hist = c_history(t, period="2y")
        rv = V.realized_vol(hist["Close"], rv_window) if "Close" in hist else float("nan")
        realized_by_ticker[t] = rv

        exps = c_expiries(t)[:expiries_per_name]
        for exp in exps:
            dte = dte_of(exp)
            if not np.isfinite(dte):
                continue
            calls, puts = c_chain(t, exp)
            frames = []
            if side in ("puts", "both") and not puts.empty:
                frames.append(("put", puts))
            if side in ("calls", "both") and not calls.empty:
                frames.append(("call", calls))
            for kind, raw in frames:
                enr = O.enrich_chain(raw, spot, rf, dte, kind, realized_vol=rv)
                if enr.empty:
                    continue
                enr["ticker"] = t
                enr["expiry"] = exp
                enr["kind"] = kind
                enr["spot"] = spot
                rows.append(enr)
    empty = pd.DataFrame()
    if not rows:
        return empty, realized_by_ticker, spot_by_ticker, empty
    raw_df = pd.concat(rows, ignore_index=True)
    # Drop illiquid / no-IV junk BEFORE fitting so the surface, frontier and
    # richness are computed only on tradeable, sane contracts.
    df = O.liquidity_filter(raw_df, min_open_int=min_oi,
                            require_two_sided=require_two_sided, max_spread_pct=max_spread)
    if df.empty:
        return empty, realized_by_ticker, spot_by_ticker, empty
    df = V.fit_skew(df)
    df, frontier = V.yield_risk_frontier(df)
    df = V.add_richness(df)
    return df, realized_by_ticker, spot_by_ticker, frontier


with tab1:
    st.markdown("**Where is option premium rich (sell) vs cheap (buy)** — vs its own "
                "skew, vs trailing realised vol, and vs the VIX regime.")

    # VIX + risk-free regime read, up top of the tab (CLAUDE.md §4).
    v1, v2, v3 = st.columns(3)
    v1.metric("Risk-free (13w T-bill)", f"{rf*100:.2f}%")
    v2.metric("VIX", f"{vix_level:.1f}" if np.isfinite(vix_level) else "n/a",
              f"{vix_pct:.0f}th pct (1y)" if np.isfinite(vix_pct) else None)
    v3.metric("As of", dt.date.today().isoformat())
    st.info(V.vix_regime(vix_level, vix_pct))

    cc = st.columns([3, 1, 1])
    tk_text = cc[0].text_input("Tickers (comma-separated)", "AAPL, MSFT, NVDA", key="t1_tk")
    exp_n = cc[1].number_input("Expiries / name", 1, 6, 2)
    side = cc[2].selectbox("Side", ["puts", "calls", "both"], index=0)

    with st.expander("Liquidity filters — defaults target actionable contracts; loosen for exotic names"):
        fc = st.columns(3)
        min_oi = fc[0].number_input("Min open interest", 0, 100000, 100, step=10,
                                    help="Open contracts at this strike. Higher = more liquid.")
        two_sided = fc[1].checkbox("Require two-sided quote (bid & ask > 0)", value=True)
        max_spread = fc[2].slider("Max bid/ask spread %", 0, 200, 25,
                                  help="(ask−bid)/mid. Wide spreads are costly to enter/exit.")

    if st.button("Scan options", type="primary"):
        st.session_state["t1_run"] = True

    if st.session_state.get("t1_run"):
        tickers = parse_tickers(tk_text)
        with st.spinner("Pulling chains and solving IV…"):
            df, rv_map, spot_map, yfrontier = build_options_table(
                tickers, int(exp_n), side, rfr_window,
                min_oi=int(min_oi), require_two_sided=two_sided, max_spread=float(max_spread))

        st.session_state["t1_last_df"] = df
        if df.empty:
            st.warning("No tradeable contracts after filtering. Yahoo may be rate-limiting, "
                       "the tickers have no liquid options, or the filters are too strict — "
                       "loosen them, reduce names, or hit Refresh.")
        else:
            st.caption(f"Realised vol ({rfr_window}d): " +
                       "  ".join(f"{t}={rv*100:.0f}%" for t, rv in rv_map.items() if np.isfinite(rv))
                       + f"  ·  {len(df)} tradeable contracts  ·  loaded {stamp()}")

            n_rich = int((df["signal"] == "RICH → sell").sum())
            n_cheap = int((df["signal"] == "CHEAP → buy").sum())
            top_rich = (df[df["signal"] == "RICH → sell"]
                        .sort_values("richness", ascending=False)["ticker"].head(3).tolist())
            msg = f"**{n_rich} RICH (sell)**, **{n_cheap} CHEAP (buy)**, {len(df) - n_rich - n_cheap} fair."
            if top_rich:
                msg += f" Richest names: {', '.join(top_rich)}."
            st.success(msg)
            st.caption("⚠️ |delta| is **not** the tail risk of a short option, and 'RICH → sell' "
                       "marks where premium is, not free money. Check spread_% and open_int before acting.")

            # -------- Opportunities table (the actionable output, shown first) --------
            st.subheader("🎯 Opportunities (ranked by richness)")
            only_flagged = st.checkbox(
                "Show only flagged opportunities (hide 'fair')", value=True,
                help="A 'fair' contract is in line with its own smile and realised vol — "
                     "no relative-value edge. Untick to see the full filtered chain.")
            rename = {"kind": "side", "dte": "DTE", "iv": "IV", "vrp": "VRP",
                      "csp_yield": "annual_CSP_yield", "spread_pct": "spread_%",
                      "abs_delta": "|delta|", "signal": "flag", "iv_source": "IV_src"}
            order = ["ticker", "expiry", "strike", "side", "DTE", "log_moneyness", "mid", "IV",
                     "VRP", "annual_CSP_yield", "spread_%", "open_int", "volume", "|delta|",
                     "skew_resid", "value_ratio", "richness", "flag", "IV_src"]
            disp = (df.rename(columns=rename)
                      .replace([np.inf, -np.inf], np.nan))
            disp = disp[[c for c in order if c in disp.columns]]
            disp = disp.sort_values("richness", ascending=False, na_position="last")
            if only_flagged:
                disp = disp[disp["flag"] != "fair"]
            if disp.empty:
                st.info("No flagged opportunities in this scan — every liquid contract is fairly "
                        "priced vs its own smile and realised vol. Untick the box above to see all, "
                        "or widen the ticker list / expiries.")

            def _flag_row(row):
                f = str(row.get("flag", ""))
                bg = ("background-color:#ffe3e3" if f.startswith("RICH")
                      else "background-color:#e3ffe6" if f.startswith("CHEAP") else "")
                return [bg] * len(row)

            if not disp.empty:
                st.dataframe(
                    disp.style.format({
                        "strike": "{:.1f}", "log_moneyness": "{:.3f}", "mid": "{:.2f}",
                        "IV": "{:.1%}", "VRP": "{:.1%}", "annual_CSP_yield": "{:.1%}",
                        "spread_%": "{:.0f}%", "|delta|": "{:.2f}", "skew_resid": "{:.3f}",
                        "value_ratio": "{:.2f}", "richness": "{:.2f}", "DTE": "{:.0f}",
                    }, na_rep="—").apply(_flag_row, axis=1),
                    use_container_width=True, height=440,
                )
            st.download_button("⬇ Download full chain (CSV)", df.to_csv(index=False),
                               "option_frontier.csv", "text/csv")

            # -------- The three frontiers --------
            left, right = st.columns(2)

            # Frontier 1: skew (smile), coloured by residual (rich red / cheap blue).
            with left:
                st.markdown("**Skew frontier** — IV vs log-moneyness; the fitted curve is the "
                            "local frontier. Red = rich vs its own smile, blue = cheap.")
                sk = df.dropna(subset=["iv", "log_moneyness"])
                if not sk.empty:
                    fig = px.scatter(sk, x="log_moneyness", y="iv", color="skew_resid",
                                     color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                                     symbol="ticker",
                                     hover_data=["ticker", "expiry", "strike", "skew_resid"])
                    for (t, e), g in sk.groupby(["ticker", "expiry"]):
                        gg = g.dropna(subset=["iv_fit"]).sort_values("log_moneyness")
                        if len(gg) > 1:
                            fig.add_trace(go.Scatter(x=gg["log_moneyness"], y=gg["iv_fit"],
                                                     mode="lines", name=f"{t} {e} fit",
                                                     line=dict(dash="dot"), showlegend=False))
                    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                                      yaxis_tickformat=".0%", coloraxis_colorbar_title="resid")
                    st.plotly_chart(fig, use_container_width=True)

            # Frontier 3: yield vs |delta| capital frontier (puts), axis clipped to stay readable.
            with right:
                st.markdown("**Yield vs risk frontier** — annualised CSP yield vs |delta|; the "
                            "upper envelope is the frontier, the dashed line is the risk-free floor.")
                ydf = df.dropna(subset=["csp_yield", "abs_delta"])
                ydf = ydf[ydf["csp_yield"] > 0]
                if not ydf.empty:
                    fig = px.scatter(ydf, x="abs_delta", y="csp_yield", color="ticker",
                                     hover_data=["strike", "expiry", "value_ratio", "spread_pct"])
                    if isinstance(yfrontier, pd.DataFrame) and len(yfrontier) > 1:
                        fig.add_trace(go.Scatter(x=yfrontier["abs_delta"], y=yfrontier["csp_yield"],
                                                 mode="lines+markers", name="frontier",
                                                 line=dict(color="black", dash="dash")))
                    fig.add_hline(y=rf, line_dash="dash", annotation_text="risk-free floor")
                    # clip the y-axis to a robust max so one weekly outlier can't blow it up
                    cap = float(np.nanpercentile(ydf["csp_yield"], 95)) * 1.25
                    cap = max(cap, rf * 2, 0.05)
                    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                                      yaxis_tickformat=".0%", yaxis_range=[0, cap])
                    st.plotly_chart(fig, use_container_width=True)

            # Frontier 2: IV vs realised vol, coloured by VRP.
            st.markdown("**Realised-vol frontier** — IV vs trailing realised vol; above the line "
                        "(IV ≫ RV) is rich → sell, below (IV ≪ RV) is cheap → buy. The "
                        "implementable stand-in for 'vs the 200-day'.")
            ivrv = df.copy()
            ivrv["realized"] = ivrv["ticker"].map(rv_map)
            ivrv = ivrv.dropna(subset=["iv", "realized"])
            if not ivrv.empty:
                fig = px.scatter(ivrv, x="realized", y="iv", color="vrp",
                                 color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                                 symbol="ticker",
                                 hover_data=["ticker", "expiry", "strike", "vrp"])
                lim = float(np.nanmax([ivrv["iv"].max(), ivrv["realized"].max()])) * 1.05
                fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                         name="IV = RV", line=dict(dash="dash", color="grey")))
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                                  xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                                  xaxis_range=[0, lim], yaxis_range=[0, lim],
                                  coloraxis_colorbar_title="VRP")
                st.plotly_chart(fig, use_container_width=True)

            # IV snapshot store (roadmap): persist ATM IV for future IV Rank/Percentile.
            with st.expander("IV history (local SQLite snapshots → IV Rank/Percentile)"):
                st.caption("yfinance has no implied-vol history. Snapshot ATM IV daily and "
                           "an IV Rank/Percentile vs trailing 252d builds up over time.")
                if st.button("📸 Snapshot today's ATM IV"):
                    for t in spot_map:
                        puts_t = df[(df["ticker"] == t) & (df["kind"] == "put")]
                        atm = iv_store.atm_iv_from_chain(puts_t, spot_map[t])
                        iv_store.snapshot(t, atm)
                    st.success("Snapshotted.")
                rank_rows = []
                for t in spot_map:
                    h = iv_store.history(t)
                    puts_t = df[(df["ticker"] == t) & (df["kind"] == "put")]
                    cur = iv_store.atm_iv_from_chain(puts_t, spot_map[t])
                    rank_rows.append({
                        "ticker": t, "atm_iv": cur,
                        "iv_rank": iv_store.iv_rank(cur, h),
                        "iv_pctile": iv_store.iv_percentile(cur, h),
                        "snapshots": len(h),
                    })
                st.dataframe(pd.DataFrame(rank_rows), use_container_width=True)

            st.warning(TAIL_CAVEAT)


# ============================================================================
# TAB 2 — Fundamentals Frontier
# ============================================================================
def best_effort_roic(info: dict, stmts: dict, rf: float) -> tuple[float, float]:
    """Compute ROIC and ROIC-WACC where the data allows; NaNs otherwise."""
    try:
        ebit = float(info.get("ebitda", np.nan))
        # prefer EBIT from the income statement if present
        inc = stmts.get("income", pd.DataFrame())
        if isinstance(inc, pd.DataFrame) and not inc.empty:
            for key in ("EBIT", "Operating Income"):
                if key in inc.index:
                    ebit = float(inc.loc[key].dropna().iloc[0]); break
        tax = 0.21
        total_debt = float(info.get("totalDebt", np.nan))
        equity = float(info.get("marketCap", np.nan))  # market value of equity
        cash = float(info.get("totalCash", np.nan))
        book_equity = float(info.get("bookValue", np.nan)) * float(info.get("sharesOutstanding", np.nan))
        roic = F.compute_roic(ebit, tax, total_debt, book_equity, cash)
        beta = float(info.get("beta", 1.0))
        cost_of_debt = rf + 0.015
        wacc = F.compute_wacc(beta, rf, 0.05, cost_of_debt, tax, equity, total_debt)
        spread = roic - wacc if np.isfinite(roic) and np.isfinite(wacc) else float("nan")
        return roic, spread
    except Exception:
        return float("nan"), float("nan")


def fundamentals_quality(info: dict, rf: float, bundle: dict) -> dict:
    """ROIC, ROIC-WACC and Piotroski F-score for one name.

    Prefers SEC EDGAR (authoritative 10-K XBRL) for ROIC inputs and the
    two-year F-score; falls back to a yfinance-derived ROIC when EDGAR is
    unavailable (e.g. hosts not allow-listed). WACC uses the market value of
    equity + beta from yfinance either way (EDGAR has no market data).
    """
    if bundle.get("available"):
        ri = bundle["roic_inputs"]
        roic = F.compute_roic(ri["ebit"], ri["tax_rate"], ri["total_debt"],
                              ri["book_equity"], ri["cash"])
        try:
            beta = float(info.get("beta", 1.0))
            equity_mv = float(info.get("marketCap", np.nan))
            debt = ri["total_debt"]
            wacc = F.compute_wacc(beta, rf, 0.05, rf + 0.015, ri["tax_rate"], equity_mv, debt)
        except Exception:
            wacc = float("nan")
        spread = roic - wacc if np.isfinite(roic) and np.isfinite(wacc) else float("nan")
        fscore = F.piotroski_fscore(bundle["fscore_curr"], bundle["fscore_prev"])
        return {"roic": roic, "roic_minus_wacc": spread, "fscore": fscore, "roic_source": "EDGAR"}

    roic, spread = best_effort_roic(info, PROVIDER.financial_statements(info.get("symbol", "")), rf)
    return {"roic": roic, "roic_minus_wacc": spread, "fscore": float("nan"), "roic_source": "yfinance"}


with tab2:
    st.markdown("**Which companies are outliers — high quality AND cheap — vs a peer "
                "group.** Everything is z-scored within the peers, so the peer set frames the screen.")
    pc = st.columns([3, 1, 1])
    peer_text = pc[0].text_input("Peer group (comma-separated)", "AAPL, MSFT, GOOGL, META, NVDA, AVGO",
                                 key="t2_peers")
    seed = pc[1].text_input("…or auto-suggest from one ticker", "", key="t2_seed")
    if pc[2].button("Suggest peers") and seed.strip():
        s = seed.strip().upper()
        info = c_fundamentals(s)
        sug = PEERS.suggest_peers(s, info.get("sector"))
        if len(sug) > 1:
            st.session_state["t2_peers"] = ", ".join(sug)
            st.rerun()
        else:
            st.warning(f"No built-in peer basket for sector '{info.get('sector')}'. Enter peers manually.")

    if st.button("Scan fundamentals", type="primary"):
        st.session_state["t2_run"] = True

    if st.session_state.get("t2_run"):
        peers = parse_tickers(st.session_state.get("t2_peers", peer_text))
        with st.spinner("Pulling fundamentals…"):
            recs = []
            for t in peers:
                info = c_fundamentals(t)
                if not info:
                    continue
                m = F.metrics_from_info(info)
                m["ticker"] = t
                info.setdefault("symbol", t)
                m.update(fundamentals_quality(info, rf, c_edgar(t)))
                recs.append(m)
        if not recs:
            st.warning("No fundamentals returned. Yahoo may be rate-limiting — try Refresh.")
        else:
            df = pd.DataFrame(recs).set_index("ticker")
            df = F.compute_composites(df)

            st.markdown("**Quality vs Value frontier** — top-right = high quality AND cheap "
                        "= the outliers (★ when both z > 0.5). Bubble = market cap, colour = growth.")
            plot = df.reset_index()
            plot["size"] = plot["market_cap"].fillna(plot["market_cap"].median()).clip(lower=1)
            fig = px.scatter(plot, x="Quality", y="Value", size="size", color="Growth",
                             text="ticker", color_continuous_scale="RdYlGn",
                             hover_data=["roic", "roic_minus_wacc", "MF_rank", "Composite"])
            fig.add_hline(y=0, line_dash="dot", line_color="grey")
            fig.add_vline(x=0, line_dash="dot", line_color="grey")
            fig.update_traces(textposition="top center")
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            stars = df[df["star"]].index.tolist()
            if stars:
                st.success("★ Outliers (high quality **and** cheap vs peers): " + ", ".join(stars))

            sources = set(df.get("roic_source", pd.Series(dtype=str)).dropna())
            if "EDGAR" in sources:
                st.caption("ROIC inputs & Piotroski F-score sourced from **SEC EDGAR** "
                           "(authoritative 10-K XBRL); WACC uses market equity + beta from yfinance.")
            else:
                st.caption("⚠️ SEC EDGAR unavailable (hosts `www.sec.gov` / `data.sec.gov` not in the "
                           "network egress allow-list) — ROIC is a best-effort yfinance proxy and the "
                           "F-score is blank. Allow-list those hosts to get authoritative figures.")
            st.caption("Value without quality is a trap (Marks). ROIC−WACC is the real "
                       "value-creation axis (Damodaran).")

            metric_cols = ["name", "Composite", "Quality", "Value", "Growth", "Safety",
                           "MF_rank", "fscore", "roic", "roic_minus_wacc", "roic_source",
                           "rev_growth", "gross_margin",
                           "op_margin", "net_margin", "fcf_margin", "roe", "roa", "fcf_yield",
                           "earnings_yield", "ev_ebitda", "pe", "pb", "net_debt_ebitda",
                           "debt_equity", "current_ratio"]
            show = df[[c for c in metric_cols if c in df.columns]]
            st.dataframe(
                show.sort_values("Composite", ascending=False).style.format({
                    "Composite": "{:.2f}", "Quality": "{:.2f}", "Value": "{:.2f}",
                    "Growth": "{:.2f}", "Safety": "{:.2f}", "roic": "{:.1%}",
                    "roic_minus_wacc": "{:.1%}", "rev_growth": "{:.1%}", "gross_margin": "{:.1%}",
                    "op_margin": "{:.1%}", "net_margin": "{:.1%}", "fcf_margin": "{:.1%}",
                    "roe": "{:.1%}", "roa": "{:.1%}", "fcf_yield": "{:.1%}",
                    "earnings_yield": "{:.1%}", "ev_ebitda": "{:.1f}", "pe": "{:.1f}",
                    "pb": "{:.1f}", "net_debt_ebitda": "{:.1f}", "debt_equity": "{:.2f}",
                    "current_ratio": "{:.2f}",
                }, na_rep="—"),
                use_container_width=True, height=420,
            )
            st.download_button("⬇ Download fundamentals (CSV)", df.to_csv(),
                               "fundamentals_frontier.csv", "text/csv")
            st.caption(f"Data as of {stamp()} · peer-relative z-scores recompute when you change the peer set.")


# ============================================================================
# TAB 3 — Available Risk Frontier (cross-tab synthesis)
# ============================================================================
with tab3:
    st.markdown("**Where does this trade rank among *everything*?** One expected-excess-return "
                "vs risk chart spanning cash, Treasuries, credit, equities, and screened options. "
                "Convex (long-vol) payoffs are rewarded, concave (short-vol) penalised beyond Sharpe "
                "(Taleb / Marks).")

    g = st.columns(4)
    ust10 = g[0].number_input("10y UST yield %", 0.0, 15.0, 4.3, 0.1) / 100
    ig = g[1].number_input("IG credit yield %", 0.0, 20.0, 5.4, 0.1) / 100
    hy = g[2].number_input("HY credit yield %", 0.0, 30.0, 7.8, 0.1) / 100
    eq_vol = g[3].number_input("Equity index vol %", 1.0, 60.0, 16.0, 0.5) / 100

    rows = [
        FR.cash_row(rf),
        FR.treasury_row(ust10, rf, "10y Treasury", risk=0.07),
        FR.credit_row(ig, rf, "IG credit", risk=0.06),
        FR.credit_row(hy, rf, "HY credit", risk=0.11),
        FR.equity_row(1 / 20.0, rf, eq_vol, "Equity index (E/Y≈5%)"),
    ]

    # Pull the best screened short-premium candidates from Tab 1 if available.
    t1_df = st.session_state.get("t1_last_df")
    note = ""
    if isinstance(t1_df, pd.DataFrame) and not t1_df.empty:
        cand = t1_df.dropna(subset=["csp_yield", "abs_delta"])
        cand = cand[(cand["csp_yield"] > 0)].sort_values("richness", ascending=False).head(5)
        for _, r in cand.iterrows():
            rows.append(FR.short_option_row(
                float(r["csp_yield"]), rf,
                risk=float(min(max(r["abs_delta"], 0.05), 0.95)),
                label=f"{r['ticker']} {r['expiry']} {r['strike']:.0f}p"))
        note = " Screened short-put candidates pulled from Tab 1."
    else:
        st.caption("Run a scan on Tab 1 to drop screened short-premium candidates onto this chart." + note)

    # Optional: parse a CSV of the user's holdings.
    up = st.file_uploader("Optional: holdings CSV (columns: label, expected_excess_return, risk, convexity)",
                          type=["csv"])
    if up is not None:
        try:
            hold = pd.read_csv(up)
            for _, r in hold.iterrows():
                rows.append(FR.FrontierRow(
                    str(r.get("label", "position")), "position",
                    float(r["expected_excess_return"]), float(r["risk"]),
                    int(r.get("convexity", 0)), "your position"))
            st.success(f"Loaded {len(hold)} positions.")
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

    fr = FR.build_frontier(rows)
    if not fr.empty:
        fig = px.scatter(fr, x="risk", y="expected_excess_return", color="asset_class",
                         text="label", hover_data=["sharpe_like", "shape_adj_score", "note"])
        env = FR.efficient_envelope(fr)
        if len(env) > 1:
            env = env.sort_values("risk")
            fig.add_trace(go.Scatter(x=env["risk"], y=env["expected_excess_return"],
                                     mode="lines", name="frontier", line=dict(color="black")))
        fig.update_traces(textposition="top center")
        fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="risk (annualised)", yaxis_title="expected excess return",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            fr[["label", "asset_class", "expected_excess_return", "risk", "convexity",
                "sharpe_like", "shape_adj_score", "note"]].style.format({
                "expected_excess_return": "{:.2%}", "risk": "{:.2%}",
                "sharpe_like": "{:.2f}", "shape_adj_score": "{:.2f}",
            }, na_rep="—"),
            use_container_width=True,
        )
    st.warning("Short-vol rows are shape-penalised: |delta|-style risk understates the tail. "
               "This ranks *candidates*; it is not advice.")
