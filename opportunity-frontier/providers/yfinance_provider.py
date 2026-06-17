"""yfinance implementation of ``DataProvider`` (the free default, CLAUDE.md §3).

Every call is wrapped in try/except and returns an empty/NaN result on failure
so a dead ticker or a Yahoo endpoint change never crashes the app. Caching is
applied in the Streamlit layer (app.py), not here, to keep this importable and
testable without Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - yfinance optional at import time
    yf = None


class YFinanceProvider:
    name = "yfinance"

    def _ticker(self, ticker: str):
        if yf is None:
            return None
        try:
            return yf.Ticker(ticker)
        except Exception:
            return None

    def spot(self, ticker: str) -> float:
        t = self._ticker(ticker)
        if t is None:
            return float("nan")
        try:
            fi = getattr(t, "fast_info", None)
            if fi:
                for key in ("last_price", "lastPrice", "regular_market_price"):
                    v = fi.get(key) if hasattr(fi, "get") else getattr(fi, key, None)
                    if v:
                        return float(v)
        except Exception:
            pass
        try:
            hist = t.history(period="5d")
            if not hist.empty:
                return float(hist["Close"].dropna().iloc[-1])
        except Exception:
            pass
        return float("nan")

    def history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        t = self._ticker(ticker)
        if t is None:
            return pd.DataFrame()
        try:
            df = t.history(period=period, interval=interval)
            return df if df is not None else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def option_expiries(self, ticker: str) -> list[str]:
        t = self._ticker(ticker)
        if t is None:
            return []
        try:
            return list(t.options or [])
        except Exception:
            return []

    def chain(self, ticker: str, expiry: str):
        t = self._ticker(ticker)
        if t is None:
            return pd.DataFrame(), pd.DataFrame()
        try:
            oc = t.option_chain(expiry)
            calls = oc.calls.copy() if oc.calls is not None else pd.DataFrame()
            puts = oc.puts.copy() if oc.puts is not None else pd.DataFrame()
            return calls, puts
        except Exception:
            return pd.DataFrame(), pd.DataFrame()

    def fundamentals(self, ticker: str) -> dict:
        t = self._ticker(ticker)
        if t is None:
            return {}
        try:
            info = t.info
            return dict(info) if info else {}
        except Exception:
            return {}

    def financial_statements(self, ticker: str) -> dict:
        t = self._ticker(ticker)
        if t is None:
            return {"income": pd.DataFrame(), "balance": pd.DataFrame(), "cashflow": pd.DataFrame()}
        out = {}
        for key, attr in (("income", "income_stmt"), ("balance", "balance_sheet"), ("cashflow", "cashflow")):
            try:
                df = getattr(t, attr)
                out[key] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
            except Exception:
                out[key] = pd.DataFrame()
        return out

    def risk_free_rate(self) -> float:
        if yf is None:
            return float("nan")
        try:
            irx = yf.Ticker("^IRX").history(period="5d")
            if not irx.empty:
                return float(irx["Close"].dropna().iloc[-1]) / 100.0  # ^IRX quoted in %
        except Exception:
            pass
        return float("nan")

    def vix(self):
        if yf is None:
            return float("nan"), float("nan")
        try:
            hist = yf.Ticker("^VIX").history(period="1y")
            if hist.empty:
                return float("nan"), float("nan")
            closes = hist["Close"].dropna()
            level = float(closes.iloc[-1])
            pct = float((closes < level).mean() * 100.0)
            return level, pct
        except Exception:
            return float("nan"), float("nan")
