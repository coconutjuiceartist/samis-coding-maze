"""The ``DataProvider`` protocol (CLAUDE.md §3).

Extract data access behind this interface so a Tradier / EODHD / Polygon
implementation can be dropped in via an env var or .streamlit/secrets.toml
without touching the analytics core or the UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass
class Quote:
    ticker: str
    spot: float
    currency: str = "USD"


@runtime_checkable
class DataProvider(Protocol):
    """Minimal surface every provider must implement.

    All methods must FAIL SOFT: on any network/parse error return an empty /
    NaN result rather than raising, so one dead ticker never crashes the app
    (CLAUDE.md §8).
    """

    name: str

    def spot(self, ticker: str) -> float:
        """Latest (delayed/EOD ok) spot price, or NaN."""

    def history(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """OHLCV history with a 'Close' column (possibly empty)."""

    def option_expiries(self, ticker: str) -> list[str]:
        """Available expiry strings (YYYY-MM-DD), possibly empty."""

    def chain(self, ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(calls, puts) DataFrames for one expiry; empty frames on failure."""

    def fundamentals(self, ticker: str) -> dict:
        """yfinance-style ``.info`` dict (possibly empty)."""

    def financial_statements(self, ticker: str) -> dict:
        """Raw statements for ROIC / F-score: keys 'income', 'balance',
        'cashflow' -> DataFrames (possibly empty)."""

    def risk_free_rate(self) -> float:
        """Risk-free rate as a decimal (e.g. 0.053), or NaN."""

    def vix(self) -> tuple[float, float]:
        """(VIX level, 1-year percentile 0-100); NaNs if unavailable."""
