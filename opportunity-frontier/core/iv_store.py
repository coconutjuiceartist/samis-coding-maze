"""Local SQLite snapshot store for ATM implied vol (CLAUDE.md §4 roadmap).

yfinance gives no history of *implied* vol, so we snapshot ATM IV daily into a
local DB. Over time this enables a true IV Rank / IV Percentile vs trailing
252 trading days — the proper stand-in for the 'vs the 200-day' comparison.

No network here; just sqlite3 + the percentile math.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "iv_snapshots.sqlite"


def _connect(db_path: str | Path | None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS iv_snapshots (
            ticker TEXT NOT NULL,
            asof   TEXT NOT NULL,
            atm_iv REAL NOT NULL,
            PRIMARY KEY (ticker, asof)
        )
        """
    )
    return conn


def atm_iv_from_chain(enriched_puts: pd.DataFrame, spot: float) -> float:
    """ATM IV = IV of the contract whose strike is closest to spot."""
    if enriched_puts is None or enriched_puts.empty or "iv" not in enriched_puts:
        return float("nan")
    df = enriched_puts.dropna(subset=["iv", "strike"])
    if df.empty:
        return float("nan")
    i = (df["strike"] - spot).abs().idxmin()
    return float(df.loc[i, "iv"])


def snapshot(ticker: str, atm_iv: float, asof: dt.date | None = None,
             db_path: str | Path | None = None) -> None:
    """Upsert one ATM IV reading for ``ticker`` on ``asof`` (default today)."""
    if not np.isfinite(atm_iv):
        return
    asof = asof or dt.date.today()
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO iv_snapshots (ticker, asof, atm_iv) VALUES (?, ?, ?) "
            "ON CONFLICT(ticker, asof) DO UPDATE SET atm_iv=excluded.atm_iv",
            (ticker.upper(), asof.isoformat(), float(atm_iv)),
        )
        conn.commit()
    finally:
        conn.close()


def history(ticker: str, lookback_days: int = 365,
            db_path: str | Path | None = None) -> pd.Series:
    """Return the stored ATM IV series for ``ticker`` over the lookback."""
    conn = _connect(db_path)
    try:
        cutoff = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()
        rows = conn.execute(
            "SELECT asof, atm_iv FROM iv_snapshots WHERE ticker=? AND asof>=? ORDER BY asof",
            (ticker.upper(), cutoff),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.Series([r[1] for r in rows], index=idx, name="atm_iv")


def iv_rank(current: float, hist: pd.Series) -> float:
    """IV Rank = (current - min) / (max - min) * 100 over the history window."""
    h = pd.to_numeric(pd.Series(hist), errors="coerce").dropna()
    if len(h) < 2 or not np.isfinite(current):
        return float("nan")
    lo, hi = float(h.min()), float(h.max())
    if hi - lo <= 0:
        return float("nan")
    return float((current - lo) / (hi - lo) * 100.0)


def iv_percentile(current: float, hist: pd.Series) -> float:
    """IV Percentile = share of history below ``current`` (0-100)."""
    h = pd.to_numeric(pd.Series(hist), errors="coerce").dropna()
    if len(h) < 2 or not np.isfinite(current):
        return float("nan")
    return float((h < current).mean() * 100.0)
