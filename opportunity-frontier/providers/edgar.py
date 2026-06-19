"""SEC EDGAR ``companyfacts`` fundamentals (free, no API key, authoritative).

Preferred over yfinance for accuracy (CLAUDE.md §3): used to compute clean
ROIC inputs and a real Piotroski F-score from two years of 10-K XBRL facts.

Network notes:
    - EDGAR requires a descriptive ``User-Agent`` with a contact email.
    - Hosts ``www.sec.gov`` and ``data.sec.gov`` must be in the environment's
      network egress allowlist. If they are not, every call FAILS SOFT
      (returns an empty/unavailable result) so the app falls back to yfinance.

The fact-parsing (``extract_annual_facts``) is a pure function and is
unit-tested against a synthetic companyfacts fixture — no network needed.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TICKER_CACHE = DATA_DIR / "edgar_tickers.json"
TICKER_CACHE_TTL_DAYS = 7

# SEC asks for a real contact; override via the EDGAR_USER_AGENT env var.
USER_AGENT = os.environ.get(
    "EDGAR_USER_AGENT", "OpportunityFrontier/1.0 (contact: set EDGAR_USER_AGENT)"
)

# Candidate us-gaap tags per concept, tried in order (issuers tag differently).
CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "ebit": ["OperatingIncomeLoss"],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "net_income": ["NetIncomeLoss"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "shares": ["WeightedAverageNumberOfSharesOutstandingBasic", "WeightedAverageNumberOfDilutedSharesOutstanding"],
    # instant (balance sheet)
    "assets": ["Assets"],
    "assets_current": ["AssetsCurrent"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "equity": ["StockholdersEquity"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashAndCashEquivalentsAtCarryingValueIncludingDisposalGroupAndDiscontinuedOperations"],
    "lt_debt_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "lt_debt_current": ["LongTermDebtCurrent", "DebtCurrent"],
}

INSTANT_CONCEPTS = {
    "assets", "assets_current", "liabilities_current", "equity", "cash",
    "lt_debt_noncurrent", "lt_debt_current",
}


# ----------------------------------------------------------------------------
# Network (fail-soft)
# ----------------------------------------------------------------------------
def _get(url: str, timeout: int = 20) -> bytes | None:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                data = gzip.decompress(data)
            return data
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return None


def _load_ticker_map() -> dict:
    """Map UPPER ticker -> zero-padded CIK string, cached to disk for a week."""
    if TICKER_CACHE.exists():
        age = dt.datetime.now() - dt.datetime.fromtimestamp(TICKER_CACHE.stat().st_mtime)
        if age.days < TICKER_CACHE_TTL_DAYS:
            try:
                return json.loads(TICKER_CACHE.read_text())
            except Exception:
                pass
    raw = _get("https://www.sec.gov/files/company_tickers.json")
    if raw is None:
        # serve a stale cache if we have one, else empty
        if TICKER_CACHE.exists():
            try:
                return json.loads(TICKER_CACHE.read_text())
            except Exception:
                return {}
        return {}
    try:
        parsed = json.loads(raw)
        mapping = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in parsed.values()}
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        TICKER_CACHE.write_text(json.dumps(mapping))
        return mapping
    except Exception:
        return {}


def cik_for_ticker(ticker: str) -> str | None:
    return _load_ticker_map().get(ticker.upper())


def company_facts(ticker: str) -> dict | None:
    cik = cik_for_ticker(ticker)
    if not cik:
        return None
    raw = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Pure parsing (unit-tested)
# ----------------------------------------------------------------------------
def _days(start: str, end: str) -> int:
    try:
        s = dt.date.fromisoformat(start)
        e = dt.date.fromisoformat(end)
        return (e - s).days
    except Exception:
        return -1


def _annual_by_fy(units: list, instant: bool) -> dict:
    """Reduce a concept's unit entries to {fiscal_year: value} from annual 10-K
    filings. Flow facts must cover ~a full year; instant facts have no start.
    The most recently filed value for each FY wins (handles restatements)."""
    best: dict[int, tuple[str, float]] = {}
    for e in units or []:
        if "val" not in e or "fy" not in e:
            continue
        if not str(e.get("form", "")).startswith("10-K"):
            continue
        has_start = bool(e.get("start"))
        if instant and has_start:
            continue
        if not instant:
            if not has_start:
                continue
            d = _days(e["start"], e["end"])
            if d < 330 or d > 400:  # only full-year durations
                continue
        fy = e["fy"]
        filed = str(e.get("filed", e.get("end", "")))
        if fy not in best or filed > best[fy][0]:
            best[fy] = (filed, float(e["val"]))
    return {fy: v for fy, (_, v) in best.items()}


def _concept_series(gaap: dict, concept_key: str) -> dict:
    """Merge candidate tags for a concept into one {fy: value} series."""
    instant = concept_key in INSTANT_CONCEPTS
    merged: dict[int, float] = {}
    for tag in CONCEPTS[concept_key]:
        node = gaap.get(tag)
        if not node:
            continue
        units = node.get("units", {}).get("USD") or next(iter(node.get("units", {}).values()), [])
        series = _annual_by_fy(units, instant)
        for fy, val in series.items():
            merged.setdefault(fy, val)  # earlier (preferred) tag wins
    return merged


def extract_annual_facts(facts: dict) -> dict:
    """Turn a companyfacts JSON into a tidy two-year fundamentals bundle.

    Returns ``{"available": False}`` if the latest two fiscal years cannot be
    assembled. Otherwise returns ROIC inputs for the latest year and Piotroski
    F-score component dicts for the current and prior years.
    """
    try:
        gaap = facts["facts"]["us-gaap"]
    except (KeyError, TypeError):
        return {"available": False}

    series = {k: _concept_series(gaap, k) for k in CONCEPTS}

    # Fiscal years where we at least know net income (the anchor concept).
    anchor_years = sorted(series["net_income"].keys(), reverse=True)
    if len(anchor_years) < 2:
        return {"available": False}
    cy, py = anchor_years[0], anchor_years[1]

    def v(key, fy):
        return series.get(key, {}).get(fy, float("nan"))

    def fscore_row(fy):
        rev = v("revenue", fy)
        cost = v("cost_of_revenue", fy)
        assets = v("assets", fy)
        ni = v("net_income", fy)
        ac = v("assets_current", fy)
        lc = v("liabilities_current", fy)
        gm = (rev - cost) / rev if rev and rev == rev and cost == cost else float("nan")
        return {
            "net_income": ni,
            "op_cash_flow": v("op_cash_flow", fy),
            "roa": (ni / assets) if assets else float("nan"),
            "total_assets": assets,
            "long_term_debt": v("lt_debt_noncurrent", fy),
            "current_ratio": (ac / lc) if lc else float("nan"),
            "shares": v("shares", fy),
            "gross_margin": gm,
            "asset_turnover": (rev / assets) if assets else float("nan"),
        }

    ebit = v("ebit", cy)
    pretax = v("pretax_income", cy)
    tax = v("tax_expense", cy)
    tax_rate = (tax / pretax) if (pretax and pretax == pretax and tax == tax) else 0.21
    if not (0.0 <= tax_rate <= 0.6):  # guard against odd ratios
        tax_rate = 0.21
    ltd = v("lt_debt_noncurrent", cy)
    ltc = v("lt_debt_current", cy)
    total_debt = sum(x for x in (ltd, ltc) if x == x)  # NaN-skipping sum

    return {
        "available": True,
        "fiscal_years": [cy, py],
        "roic_inputs": {
            "ebit": ebit,
            "tax_rate": tax_rate,
            "total_debt": total_debt if total_debt else float("nan"),
            "book_equity": v("equity", cy),
            "cash": v("cash", cy),
        },
        "fscore_curr": fscore_row(cy),
        "fscore_prev": fscore_row(py),
    }


def fundamentals_bundle(ticker: str) -> dict:
    """Fetch + parse EDGAR facts for ``ticker``; ``{"available": False}`` on any
    failure (host blocked, unknown ticker, parse error)."""
    facts = company_facts(ticker)
    if facts is None:
        return {"available": False}
    return extract_annual_facts(facts)
