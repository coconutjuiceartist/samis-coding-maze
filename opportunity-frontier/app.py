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

from core import capital as CAP
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


@st.cache_data(ttl=SPOT_TTL, show_spinner=False)
def c_treasury_curve() -> dict:
    return PROVIDER.treasury_curve()


@st.cache_data(ttl=FUND_TTL, show_spinner=False)
def c_market_yields() -> dict:
    """IG/HY credit and equity earnings yields for the cross-asset menu (Tab 3).
    Each leg fails soft to NaN so a dead endpoint never empties the frontier."""
    return {
        "ig": PROVIDER.etf_yield("LQD"),    # iShares IG corporate bond ETF
        "hy": PROVIDER.etf_yield("HYG"),    # iShares HY corporate bond ETF
        "eq_ey": PROVIDER.equity_earnings_yield("SPY"),
    }


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
                        min_oi=0, require_two_sided=True, max_spread=None,
                        min_dte=7, moneyness_pct=30, delta_band=(0.05, 0.55)):
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

        # Skip 0-DTE / near-expiry: pick the first N expiries that clear min_dte
        # (these have real vol content; 0-DTE is degenerate).
        usable = [e for e in c_expiries(t) if np.isfinite(dte_of(e)) and dte_of(e) >= min_dte]
        for exp in usable[:expiries_per_name]:
            dte = dte_of(exp)
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
    diag = {"raw": 0, "kept": 0, "kept_by_ticker": {}, "report": {}}
    if not rows:
        return empty, realized_by_ticker, spot_by_ticker, empty, diag
    raw_df = pd.concat(rows, ignore_index=True)
    diag["raw"] = len(raw_df)
    # Drop illiquid / no-IV junk BEFORE fitting so the surface, frontier and
    # richness are computed only on tradeable, sane contracts.
    band = (1 - moneyness_pct / 100.0, 1 + moneyness_pct / 100.0)
    filt = dict(min_open_int=min_oi, require_two_sided=require_two_sided,
                max_spread_pct=max_spread, min_dte=min_dte, moneyness_range=band,
                delta_band=delta_band)
    diag["report"] = O.liquidity_report(raw_df, **filt)
    df = O.liquidity_filter(raw_df, **filt)
    diag["kept"] = len(df)
    if df.empty:
        return empty, realized_by_ticker, spot_by_ticker, empty, diag
    diag["kept_by_ticker"] = df.groupby("ticker").size().to_dict()
    df = V.fit_skew(df)
    df, frontier = V.yield_risk_frontier(df)
    df = V.add_richness(df)
    df = add_capital_view(df, c_treasury_curve())
    return df, realized_by_ticker, spot_by_ticker, frontier, diag


def add_capital_view(df: pd.DataFrame, curve: dict) -> pd.DataFrame:
    """Attach the capital-/risk-aware columns (CLAUDE.md §1, §6): for each put,
    how much it pays over a Treasury of its OWN maturity once the collateral is
    assumed to earn that risk-free carry, plus an honest tail-loss figure.

    Falls back to the flat 13-week rate when the full curve is unavailable, so
    the columns populate even if the curve endpoints aren't allow-listed.
    """
    out = df.copy()
    for c in ("rf_matched", "premium_yield", "excess_yield", "blended_return",
              "collateral", "breakeven", "tail_loss"):
        out[c] = np.nan
    if out.empty:
        return out
    for i, row in out.iterrows():
        if str(row.get("kind")) != "put":
            continue
        dte = float(row.get("dte", np.nan))
        strike = float(row.get("strike", np.nan))
        mid = float(row.get("mid", np.nan))
        if not (np.isfinite(dte) and np.isfinite(strike) and np.isfinite(mid)):
            continue
        rf_m = CAP.matched_rf(dte / 365.0, curve)
        if not np.isfinite(rf_m):
            rf_m = rf  # flat fallback to the 13-week bill
        m = CAP.csp_metrics(mid, strike, dte, rf_m,
                            spot=row.get("spot"), iv=row.get("iv"), assume_carry=True)
        for c in ("rf_matched", "premium_yield", "excess_yield", "blended_return",
                  "collateral", "breakeven", "tail_loss"):
            out.at[i, c] = m[c]
    return out


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

    with st.expander("Liquidity & sanity filters — defaults target actionable contracts"):
        fc = st.columns(3)
        min_oi = fc[0].number_input("Min open interest", 0, 100000, 50, step=10,
                                    help="Open contracts at this strike. Higher = more liquid.")
        two_sided = fc[1].checkbox("Require two-sided quote (bid & ask > 0)", value=True)
        max_spread = fc[2].slider("Max bid/ask spread %", 0, 200, 25,
                                  help="(ask−bid)/mid. Wide spreads are costly to enter/exit.")
        fc2 = st.columns(2)
        min_dte = fc2[0].number_input("Min days to expiry", 0, 365, 7, step=1,
                                      help="0-DTE / near-expiry options are ~all intrinsic and "
                                           "have no vol content — exclude them.")
        moneyness_pct = fc2[1].slider("Strike within ±% of spot", 5, 100, 30,
                                      help="Keeps near-the-money strikes; drops deep ITM/OTM wings "
                                           "that are untradeable as premium and pollute the surface.")
        delta_band = st.slider(
            "Keep |delta| between (the sellable-premium zone)", 0.0, 1.0, (0.05, 0.55), step=0.05,
            help="Excludes deep-ITM options (|delta|→1) whose IV is meaningless — near-zero vega "
                 "means a few cents of price noise swing IV by 10+ points — and far-OTM lottery "
                 "tickets (|delta|→0) with no real premium.")

    if st.button("Scan options", type="primary"):
        st.session_state["t1_run"] = True

    if st.session_state.get("t1_run"):
        tickers = parse_tickers(tk_text)
        with st.spinner("Pulling chains and solving IV…"):
            df, rv_map, spot_map, yfrontier, diag = build_options_table(
                tickers, int(exp_n), side, rfr_window,
                min_oi=int(min_oi), require_two_sided=two_sided, max_spread=float(max_spread),
                min_dte=int(min_dte), moneyness_pct=int(moneyness_pct),
                delta_band=(float(delta_band[0]), float(delta_band[1])))

        st.session_state["t1_last_df"] = df
        if df.empty:
            if diag["raw"] == 0:
                st.warning("No option data returned. Yahoo may be rate-limiting, the tickers have "
                           "no listed options, or the hosts aren't allow-listed — hit Refresh or try again.")
            else:
                st.warning(f"Pulled **{diag['raw']}** listed contracts but **0 passed the filters**. "
                           "Here's how many each rule removed (independently) so you can see the "
                           "binding constraint:")
                rep = diag.get("report", {})
                if rep:
                    rep_df = (pd.DataFrame(sorted(rep.items(), key=lambda kv: -kv[1]),
                                           columns=["filter", "contracts removed"]))
                    st.dataframe(rep_df, hide_index=True, use_container_width=True)
                    top, top_n = max(rep.items(), key=lambda kv: kv[1])
                    tip = ("If the options market is closed (overnight/pre-market), bids & asks are "
                           "often 0 — untick **Require two-sided quote**." if "one-sided" in top
                           else "Lower **Min open interest**." if "open interest" in top
                           else "Raise **Max bid/ask spread %**." if "spread" in top
                           else "Widen **Strike within ±% of spot**." if "strike outside" in top
                           else "Lower **Min days to expiry**." if "days to expiry" in top
                           else "Loosen the relevant filter above.")
                    st.info(f"Binding constraint: **{top}** removed {top_n} of {diag['raw']}. {tip}")
        else:
            st.caption(f"Realised vol ({rfr_window}d): " +
                       "  ".join(f"{t}={rv*100:.0f}%" for t, rv in rv_map.items() if np.isfinite(rv))
                       + f"  ·  filtered {diag['raw']} → {diag['kept']} tradeable contracts  ·  loaded {stamp()}")

            thin = [t for t, n in diag["kept_by_ticker"].items() if n < 4]
            if thin:
                st.warning("Too few liquid strikes to fit a skew for: **" + ", ".join(thin) +
                           "** (need ≥4). Their points still plot, but the skew signal needs a "
                           "fuller surface — loosen liquidity filters or add more expiries.")
            if len(tickers) < 3:
                st.caption("ℹ️ This is a *relative-value* scanner — it ranks contracts against each "
                           "other. Give it several liquid names (e.g. SPY, QQQ, AAPL, TSLA, NVDA) "
                           "across 2–3 expiries so there's a real cross-section to rank.")

            n_rich = int((df["signal"] == "RICH → sell").sum())
            n_cheap = int((df["signal"] == "CHEAP → buy").sum())
            ranked = df.sort_values("richness", ascending=False)
            richest, cheapest = ranked.iloc[0], ranked.iloc[-1]

            def _tag(row):
                rr = row.get("iv_rv_ratio", np.nan)
                rr_txt = f", IV/RV {rr:.2f}" if np.isfinite(rr) else ""
                return f"{row['ticker']} {row['strike']:.0f}p (richness {row['richness']:+.2f}{rr_txt})"

            if n_rich or n_cheap:
                st.success(f"**{n_rich} strong RICH (sell)** · **{n_cheap} strong CHEAP (buy)** "
                           f"signals (|richness| > 0.7) out of {len(df)} contracts.")
            else:
                st.info("No *strong* signals (|richness| > 0.7) this scan — normal in a quiet tape. "
                        "Showing the most rich- and cheap-leaning contracts so you always have "
                        "candidates to weigh.")
            st.caption(f"Most rich-leaning: **{_tag(richest)}** → sell premium.  ·  "
                       f"Most cheap-leaning: **{_tag(cheapest)}** → buy convexity.")

            # Capital verdict for the top pick: does the premium actually pay you
            # enough OVER the matched Treasury to justify the tail (CLAUDE.md §1)?
            ex = richest.get("excess_yield", np.nan)
            bl = richest.get("blended_return", np.nan)
            tl = richest.get("tail_loss", np.nan)
            if np.isfinite(ex):
                verdict = ("**barely worth it** — it pays almost nothing over cash for real tail risk"
                           if ex < 0.02 else
                           "**a modest premium** over cash for the tail you take" if ex < 0.05 else
                           "**a meaningful premium** over cash — still confirm the tail and liquidity")
                tl_txt = f", and a ~2σ bad move costs **{tl*100:.0f}% of collateral**" if np.isfinite(tl) else ""
                st.info(f"💰 **Capital read — {richest['ticker']} {richest['strike']:.0f}p:** pays "
                        f"**{ex*100:.1f}%/yr over the matched Treasury** "
                        f"(blended {bl*100:.1f}%/yr with carry){tl_txt}. That's {verdict}. "
                        "See Tab 3 to rank it against credit, equities and convex buys.")
            st.caption("⚠️ |delta| is **not** the tail risk of a short option, and 'RICH → sell' "
                       "marks where premium is, not free money. Check spread_% and open_int before acting.")

            # -------- Opportunities (always surface both ends of the ranking) --------
            st.subheader("🎯 Opportunities (ranked by richness)")
            rename = {"kind": "side", "dte": "DTE", "iv": "IV", "vrp": "VRP",
                      "iv_rv_ratio": "IV/RV", "csp_yield": "annual_CSP_yield",
                      "spread_pct": "spread_%", "abs_delta": "|delta|", "signal": "flag",
                      "iv_source": "IV_src", "excess_yield": "excess_vs_UST",
                      "blended_return": "blended", "tail_loss": "tail_loss_2σ",
                      "rf_matched": "matched_UST"}
            full = df.rename(columns=rename).replace([np.inf, -np.inf], np.nan)
            fmt = {"strike": "{:.1f}", "log_moneyness": "{:.3f}", "mid": "{:.2f}",
                   "IV": "{:.1%}", "IV/RV": "{:.2f}", "VRP": "{:.1%}", "annual_CSP_yield": "{:.1%}",
                   "spread_%": "{:.0f}%", "|delta|": "{:.2f}", "skew_resid": "{:.3f}",
                   "z_skew": "{:.2f}", "z_vrp": "{:.2f}", "value_ratio": "{:.2f}",
                   "richness": "{:.2f}", "DTE": "{:.0f}", "excess_vs_UST": "{:.1%}",
                   "blended": "{:.1%}", "tail_loss_2σ": "{:.0%}", "matched_UST": "{:.1%}"}
            compact = ["ticker", "expiry", "strike", "annual_CSP_yield", "excess_vs_UST",
                       "blended", "tail_loss_2σ", "|delta|", "IV/RV", "spread_%", "open_int",
                       "richness", "flag"]

            with st.expander("📖 What these columns mean (plain English)"):
                st.markdown(
                    "- **annual_CSP_yield** — premium income per year as a % of the cash you set "
                    "aside (strike − premium). The raw return from selling the put.\n"
                    "- **matched_UST** — the risk-free Treasury yield for *this option's own "
                    "maturity* (a 2-year put is judged against the 2-year, not the 3-month, bill).\n"
                    "- **excess_vs_UST** — how much **more per year** the put pays than that matched "
                    "Treasury. Assuming the collateral earns the risk-free rate as carry, this *is* "
                    "your pay for taking the short-vol risk. **This is the number that answers "
                    "\"is it worth it?\"** — small here means you're being paid almost nothing extra "
                    "over cash to warehouse tail risk.\n"
                    "- **blended** — total annual return if held in T-bills: risk-free carry **plus** "
                    "the premium.\n"
                    "- **tail_loss_2σ** — Marks' \"how much could I lose\": the % of your collateral "
                    "gone in a ~2-sigma adverse move by expiry. The honest risk |delta| can't see.\n"
                    "- **|delta|** — the *probability* of assignment, not the size of the loss.\n"
                    "- **IV/RV** — implied vol ÷ the stock's realised vol; >1 = the vol you're "
                    "selling is dear vs what the stock actually does (the edge).\n"
                    "- **richness** — composite vol-edge score (skew + IV-vs-realised); the *timing/"
                    "selection* signal, separate from the *capital-quality* columns above.")

            def _view(frame):
                return frame[[c for c in compact if c in frame.columns]]

            cS, cB = st.columns(2)
            with cS:
                st.markdown("**🔴 Richest — premium-selling candidates**")
                st.caption("Vol looks dear here (high IV vs its own smile *and* vs realised). "
                           "Now read **excess_vs_UST**: that's the pay over cash for the tail in "
                           "**tail_loss_2σ** — a rich vol edge with thin excess is still a poor "
                           "use of capital.")
                st.dataframe(_view(full.sort_values("richness", ascending=False).head(8))
                             .style.format(fmt, na_rep="—"), use_container_width=True, height=320)
            with cB:
                st.markdown("**🟢 Cheapest — convexity-buying candidates**")
                st.caption("Vol looks cheap here — *buying* these is long convexity (capped loss, "
                           "fat right tail; Taleb/PTJ). The capital columns are framed for *selling*, "
                           "so read these as buy candidates, not CSP yields.")
                st.dataframe(_view(full.sort_values("richness", ascending=True).head(8))
                             .style.format(fmt, na_rep="—"), use_container_width=True, height=320)

            def _flag_row(row):
                f = str(row.get("flag", ""))
                bg = ("background-color:#ffe3e3" if f.startswith("RICH")
                      else "background-color:#e3ffe6" if f.startswith("CHEAP") else "")
                return [bg] * len(row)

            with st.expander(f"Full ranked chain ({len(df)} contracts) — skew vs realised breakdown"):
                order = ["ticker", "expiry", "strike", "DTE", "log_moneyness", "mid", "IV", "IV/RV",
                         "VRP", "annual_CSP_yield", "matched_UST", "excess_vs_UST", "blended",
                         "tail_loss_2σ", "spread_%", "open_int", "volume", "|delta|",
                         "skew_resid", "z_skew", "z_vrp", "value_ratio", "richness", "flag", "IV_src"]
                disp = full[[c for c in order if c in full.columns]].sort_values(
                    "richness", ascending=False, na_position="last")
                st.dataframe(disp.style.format(fmt, na_rep="—").apply(_flag_row, axis=1),
                             use_container_width=True, height=440)
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
    st.markdown("**Where does this trade rank among *everything*?** One chart, two axes: "
                "**up = how much more you earn per year than cash** (expected excess return), "
                "**right = how much you could lose** (risk). Every alternative use of the same "
                "capital sits here — cash, Treasuries, IG/HY credit, equities, and the options "
                "you screened on Tab 1 — so a single picture answers the governing question. "
                "Convex (long-vol) payoffs are rewarded and concave (short-vol) penalised *beyond* "
                "what a volatility-Sharpe implies, because |delta|-style vol hides a short option's "
                "tail (Taleb / Marks).")

    # ---- The market menu: auto-pulled, editable (graceful fallback to manual) ----
    curve = c_treasury_curve()
    mkt = c_market_yields()
    auto_ok = bool(curve) and any(np.isfinite(v) for v in mkt.values())
    ust3m = CAP.matched_rf(0.25, curve)
    ust10 = CAP.matched_rf(10.0, curve)
    ig_auto, hy_auto, eq_auto = mkt.get("ig"), mkt.get("hy"), mkt.get("eq_ey")

    src = ("✅ Menu auto-pulled from live market data (editable below)." if auto_ok else
           "⚠️ Live market menu unavailable (Yahoo hosts may not be allow-listed) — "
           "using editable manual defaults.")
    st.caption(src + "  These are the *alternatives* every option competes with.")
    with st.expander("Market menu — auto-pulled yields & risks (edit any cell)"):
        g = st.columns(4)
        ust10_in = g[0].number_input("10y UST yield %", 0.0, 15.0,
                                     round((ust10 if np.isfinite(ust10) else 0.043) * 100, 2), 0.1) / 100
        ig_in = g[1].number_input("IG credit yield %", 0.0, 20.0,
                                  round((ig_auto if np.isfinite(ig_auto) else 0.054) * 100, 2), 0.1) / 100
        hy_in = g[2].number_input("HY credit yield %", 0.0, 30.0,
                                  round((hy_auto if np.isfinite(hy_auto) else 0.078) * 100, 2), 0.1) / 100
        eq_ey_in = g[3].number_input("Equity earnings yield %", 0.0, 20.0,
                                     round((eq_auto if np.isfinite(eq_auto) else 0.05) * 100, 2), 0.1) / 100
        eq_vol_default = vix_level / 100.0 if np.isfinite(vix_level) else 0.16
        eq_vol = st.number_input("Equity index risk (annual vol) %", 1.0, 60.0,
                                 round(eq_vol_default * 100, 1), 0.5) / 100
        st.caption("Risks (x-axis): a Treasury/credit's risk is its price volatility; equity's is "
                   "its return vol; a short option's is the vol of the stock you'd be forced to own "
                   "if assigned — a fuller risk than |delta|, which sees only the odds of assignment, "
                   "not the size of the loss.")

    rows = [
        FR.cash_row(rf),
        FR.treasury_row(ust3m if np.isfinite(ust3m) else rf, rf, "3m T-bill", risk=0.005),
        FR.treasury_row(ust10_in, rf, "10y Treasury", risk=0.07),
        FR.credit_row(ig_in, rf, "IG credit (LQD)", risk=0.06),
        FR.credit_row(hy_in, rf, "HY credit (HYG)", risk=0.11),
        FR.equity_row(eq_ey_in, rf, eq_vol, "Equity index (SPY)"),
    ]
    user_labels: set[str] = set()  # rows the user supplied (scan picks + CSV), for highlighting

    # ---- Both sides of the Tab 1 scan: short premium AND convex buys ----
    t1_df = st.session_state.get("t1_last_df")
    if isinstance(t1_df, pd.DataFrame) and not t1_df.empty:
        sells = t1_df[t1_df.get("kind", "") == "put"].dropna(subset=["excess_yield"])
        sells = sells[sells["excess_yield"].replace([np.inf, -np.inf], np.nan).notna()]
        sells = sells.sort_values("richness", ascending=False).head(5)
        for _, r in sells.iterrows():
            lbl = f"SELL {r['ticker']} {r['expiry']} {r['strike']:.0f}p"
            tl = r.get("tail_loss", np.nan)
            tail_txt = f"; 2σ tail loss ≈ {tl*100:.0f}% of collateral" if np.isfinite(tl) else ""
            rows.append(FR.FrontierRow(
                lbl, "short_option", float(r["excess_yield"]),
                CAP.short_put_risk(r.get("abs_delta"), r.get("iv")), -1,
                "short vol: premium over cash for the tail" + tail_txt))
            user_labels.add(lbl)

        buys = t1_df.dropna(subset=["iv", "vrp"])
        buys = buys.sort_values("richness", ascending=True).head(3)  # cheapest vol
        for _, r in buys.iterrows():
            edge = -float(r["vrp"])  # IV below realised ⇒ positive edge
            if not np.isfinite(edge):
                continue
            lbl = f"BUY {r['ticker']} {r['expiry']} {r['strike']:.0f}{r['kind'][0]}"
            rows.append(FR.long_option_row(edge, float(r["iv"]), lbl))
            user_labels.add(lbl)
        st.caption("📍 Your Tab 1 picks are on the chart: **SELL …** = premium-selling candidates "
                   "(plotted at their pay-over-cash and honest risk), **BUY …** = cheap convexity.")
    else:
        st.caption("ℹ️ Run a scan on **Tab 1** to drop your screened options onto this chart.")

    # ---- Score specific trades from a CSV (e.g. a broker export) ----
    with st.expander("⬆️ Score specific trades — upload a CSV"):
        st.markdown(
            "Two formats accepted:\n"
            "1. **Option trades** (recommended): columns `type, strike, mid, dte` "
            "(`type` ∈ short_put / long_put / long_call / short_call). Add `spot` & `iv` "
            "(decimal, e.g. 0.45) to get an honest tail-loss and risk; add `label`/`expiry` "
            "(YYYY-MM-DD, used if `dte` is absent). Each row is priced with the carry assumption "
            "and dropped onto the chart.\n"
            "2. **Pre-computed**: columns `label, expected_excess_return, risk, convexity` "
            "(decimals) — plotted as-is.")
        st.code("type,label,strike,mid,dte,spot,iv\n"
                "short_put,SPCX Dec28 25p,25,2.04,912,24,0.50", language="text")
        up = st.file_uploader("Trades CSV", type=["csv"])

    def _rows_from_csv(hold: pd.DataFrame):
        """Map an uploaded CSV to FrontierRows. Returns (rows, labels, messages)."""
        new_rows, labels, msgs = [], set(), []
        cols = {c.lower(): c for c in hold.columns}
        # Pre-computed schema.
        if {"expected_excess_return", "risk"}.issubset(cols):
            for _, r in hold.iterrows():
                lbl = str(r.get(cols.get("label", "label"), "position"))
                new_rows.append(FR.FrontierRow(lbl, "your_trade",
                                               float(r[cols["expected_excess_return"]]),
                                               float(r[cols["risk"]]),
                                               int(r.get(cols.get("convexity", "convexity"), 0) or 0),
                                               "your trade (pre-computed)"))
                labels.add(lbl)
            return new_rows, labels, msgs
        if "type" not in cols:
            msgs.append(("error", "CSV needs either a `type` column (option trades) or "
                                  "`expected_excess_return` & `risk` (pre-computed)."))
            return new_rows, labels, msgs
        for i, r in hold.iterrows():
            try:
                typ = str(r[cols["type"]]).strip().lower()
                strike = float(r[cols["strike"]])
                mid = float(r[cols.get("mid", cols.get("price", "mid"))])
                dte = (float(r[cols["dte"]]) if "dte" in cols
                       else dte_of(str(r[cols["expiry"]])) if "expiry" in cols else np.nan)
                spot = float(r[cols["spot"]]) if "spot" in cols else None
                iv = float(r[cols["iv"]]) if "iv" in cols else None
                lbl = str(r[cols["label"]]) if "label" in cols else f"{typ} {strike:.0f}"
                if not (np.isfinite(strike) and np.isfinite(mid) and np.isfinite(dte)):
                    msgs.append(("warning", f"Row {i}: need numeric strike, mid and dte/expiry — skipped."))
                    continue
                rf_m = CAP.matched_rf(dte / 365.0, curve)
                rf_m = rf_m if np.isfinite(rf_m) else rf
                if typ == "short_put":
                    m = CAP.csp_metrics(mid, strike, dte, rf_m, spot=spot, iv=iv, assume_carry=True)
                    risk = CAP.short_put_risk(np.nan, iv)
                    if not np.isfinite(risk):
                        msgs.append(("warning", f"Row {i} ({lbl}): add `iv` so the risk axis is "
                                                "honest — skipped."))
                        continue
                    tl = m["tail_loss"]
                    tail_txt = f"; 2σ tail loss ≈ {tl*100:.0f}% of collateral" if np.isfinite(tl) else ""
                    new_rows.append(FR.FrontierRow(
                        lbl, "your_trade", float(m["excess_yield"]), float(risk), -1,
                        f"your short put: {m['excess_yield']*100:.1f}%/yr over matched UST{tail_txt}"))
                    labels.add(lbl)
                elif typ in ("long_put", "long_call"):
                    risk = float(iv) if (iv is not None and np.isfinite(iv) and iv > 0) else 0.20
                    new_rows.append(FR.long_option_row(0.0, risk, lbl))
                    labels.add(lbl)
                    msgs.append(("info", f"Row {i} ({lbl}): long convexity — loss capped at premium; "
                                         "expected excess is view-dependent so plotted at 0 (the "
                                         "convexity reward still lifts its score)."))
                else:
                    msgs.append(("warning", f"Row {i}: unknown type '{typ}' — skipped."))
            except Exception as e:
                msgs.append(("warning", f"Row {i}: could not parse ({e}) — skipped."))
        return new_rows, labels, msgs

    if up is not None:
        try:
            hold = pd.read_csv(up)
            new_rows, labels, msgs = _rows_from_csv(hold)
            rows.extend(new_rows)
            user_labels |= labels
            for level, text in msgs:
                getattr(st, level)(text)
            if new_rows:
                st.success(f"Scored {len(new_rows)} trade(s) and added them to the frontier below.")
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

    # ---- Build, plot, explain ----
    fr = FR.build_frontier(rows)
    if not fr.empty:
        shape_name = {1: "convex (long vol)", -1: "concave (short vol)", 0: "linear"}
        fr = fr.copy()
        fr["payoff"] = fr["convexity"].map(shape_name)
        fr["yours"] = fr["label"].isin(user_labels)

        fig = px.scatter(fr, x="risk", y="expected_excess_return", color="asset_class",
                         symbol="payoff",
                         symbol_map={"convex (long vol)": "triangle-up",
                                     "concave (short vol)": "triangle-down", "linear": "circle"},
                         text="label",
                         hover_data={"label": False, "payoff": True, "sharpe_like": ":.2f",
                                     "shape_adj_score": ":.2f", "note": True})
        env = FR.efficient_envelope(fr)
        if len(env) > 1:
            env = env.sort_values("risk")
            fig.add_trace(go.Scatter(x=env["risk"], y=env["expected_excess_return"],
                                     mode="lines", name="frontier (best return per unit risk)",
                                     line=dict(color="black")))
        # Mark cash (the floor every trade must beat) and ring the user's own trades.
        fig.add_hline(y=0, line_dash="dot", line_color="grey",
                      annotation_text="cash floor (0% over risk-free)")
        yours = fr[fr["yours"]]
        if not yours.empty:
            fig.add_trace(go.Scatter(x=yours["risk"], y=yours["expected_excess_return"],
                                     mode="markers", name="your trades",
                                     marker=dict(size=16, color="rgba(0,0,0,0)",
                                                 line=dict(color="black", width=2))))
        fig.update_traces(textposition="top center", selector=dict(mode="markers+text"))
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="risk → (annualised; how much you could lose)",
                          yaxis_title="excess return ↑ (per year, over risk-free)",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**How to read it:** the black line is the frontier — the best excess return "
                   "available at each level of risk. Points **above** it are unusually good value; "
                   "points **below** it are dominated (something else pays more for the same risk, "
                   "or the same return for less risk). ▲ = convex (capped loss, fat right tail), "
                   "▼ = concave/short-vol (the score docks these for tail risk), ● = linear. "
                   "Black rings are your own trades/picks.")

        # Where do the user's trades land? Plain-English ranking.
        ranked = fr.reset_index(drop=True)
        if user_labels:
            lines = []
            for lbl in user_labels:
                hit = ranked[ranked["label"] == lbl]
                if hit.empty:
                    continue
                pos = int(hit.index[0]) + 1
                row = hit.iloc[0]
                beats_cash = "beats" if row["expected_excess_return"] > 0 else "does **not** beat"
                lines.append(f"- **{lbl}** ranks **#{pos} of {len(ranked)}** by shape-adjusted score "
                             f"(pays {row['expected_excess_return']*100:.1f}%/yr over cash at "
                             f"{row['risk']*100:.0f}% risk — {beats_cash} cash).")
            if lines:
                st.markdown("**Your trades vs the field:**\n" + "\n".join(lines))

        st.markdown("**The full menu, ranked** — `shape_adj_score` is the bottom line: excess "
                    "return per unit of risk, *then* tilted for payoff shape (convex up, concave "
                    "down). Higher = a better use of the same capital and risk.")
        st.dataframe(
            ranked[["label", "asset_class", "payoff", "expected_excess_return", "risk",
                    "sharpe_like", "shape_adj_score", "note"]].rename(columns={
                "expected_excess_return": "excess_vs_cash", "sharpe_like": "return_per_risk",
                "shape_adj_score": "score (shape-adj)"}).style.format({
                "excess_vs_cash": "{:.2%}", "risk": "{:.2%}",
                "return_per_risk": "{:.2f}", "score (shape-adj)": "{:.2f}",
            }, na_rep="—"),
            use_container_width=True,
        )
        st.caption("**Columns:** *excess_vs_cash* = annual return above the risk-free rate; "
                   "*risk* = annualised loss/vol proxy; *return_per_risk* = excess ÷ risk (a "
                   "Sharpe-like ratio); *score (shape-adj)* = that ratio after rewarding convexity "
                   "and penalising short-vol tails.")
    st.warning("⚠️ Short-vol rows are shape-penalised because |delta|-style vol understates the "
               "tail (Taleb/Marks). This ranks *candidates* on honest reward-vs-risk; it is a "
               "screen, never advice. Confirm liquidity and size for the tail before acting.")
