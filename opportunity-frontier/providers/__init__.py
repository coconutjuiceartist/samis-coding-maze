"""Swappable data providers (CLAUDE.md §3).

``get_provider()`` selects an implementation by name / env var so a paid feed
(Tradier, EODHD, Polygon) can replace yfinance without touching the app.
"""
from __future__ import annotations

import os

from .base import DataProvider, Quote
from .yfinance_provider import YFinanceProvider

_REGISTRY = {
    "yfinance": YFinanceProvider,
}


def get_provider(name: str | None = None) -> DataProvider:
    """Return a provider instance.

    Resolution order: explicit ``name`` arg, then the OPPORTUNITY_PROVIDER env
    var, then the yfinance default.
    """
    key = (name or os.environ.get("OPPORTUNITY_PROVIDER") or "yfinance").lower()
    cls = _REGISTRY.get(key, YFinanceProvider)
    return cls()


__all__ = ["DataProvider", "Quote", "YFinanceProvider", "get_provider"]
