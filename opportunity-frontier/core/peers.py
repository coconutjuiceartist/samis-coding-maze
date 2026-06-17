"""Sector-aware peer auto-suggestion (CLAUDE.md §5 roadmap 4).

Given one ticker's sector/industry, propose a sensible peer set so the user can
type a single name and still get a relative screen. Uses a small built-in map
of liquid large caps by sector (no network); the provider can extend this.
"""
from __future__ import annotations

# Compact, liquid peer baskets keyed by yfinance ``sector`` strings.
SECTOR_PEERS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "AMD"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "T", "VZ", "TMUS"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX"],
    "Consumer Defensive": ["PG", "KO", "PEP", "COST", "WMT", "MDLZ", "CL"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK"],
    "Healthcare": ["UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC"],
    "Industrials": ["CAT", "HON", "UPS", "BA", "GE", "RTX", "DE", "LMT"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE"],
    "Basic Materials": ["LIN", "SHW", "APD", "ECL", "FCX", "NEM", "DOW"],
    "Real Estate": ["PLD", "AMT", "EQIX", "PSA", "O", "SPG", "CCI"],
}


def suggest_peers(ticker: str, sector: str | None, limit: int = 8) -> list[str]:
    """Return up to ``limit`` peers for ``ticker`` based on its sector.

    The seed ticker is always first; if the sector is unknown the seed is
    returned alone so the caller can prompt for an explicit peer set.
    """
    ticker = ticker.upper()
    basket = SECTOR_PEERS.get(sector or "", [])
    peers = [t for t in basket if t != ticker]
    out = [ticker] + peers
    return out[:limit]
