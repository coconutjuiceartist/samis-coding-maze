"""Tests for the pure EDGAR companyfacts parser (no network)."""
import pytest

from providers import edgar
from core import fundamentals as F


def _flow(fy, val):
    return {"start": f"{fy}-01-01", "end": f"{fy}-12-31", "val": val,
            "fy": fy, "fp": "FY", "form": "10-K", "filed": f"{fy + 1}-02-01"}


def _inst(fy, val):
    return {"end": f"{fy}-12-31", "val": val, "fy": fy, "fp": "FY",
            "form": "10-K", "filed": f"{fy + 1}-02-01"}


def _facts():
    def usd(rows):
        return {"units": {"USD": rows}}
    def shares(rows):
        return {"units": {"shares": rows}}
    gaap = {
        "NetIncomeLoss": usd([_flow(2023, 800), _flow(2024, 1000)]),
        "NetCashProvidedByUsedInOperatingActivities": usd([_flow(2023, 1100), _flow(2024, 1500)]),
        "Revenues": usd([_flow(2023, 9000), _flow(2024, 10000)]),
        "CostOfRevenue": usd([_flow(2023, 5400), _flow(2024, 5000)]),
        "OperatingIncomeLoss": usd([_flow(2023, 1200), _flow(2024, 1400)]),
        "IncomeTaxExpenseBenefit": usd([_flow(2023, 250), _flow(2024, 300)]),
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest":
            usd([_flow(2023, 1050), _flow(2024, 1200)]),
        "WeightedAverageNumberOfSharesOutstandingBasic": shares([_flow(2023, 510), _flow(2024, 500)]),
        "Assets": usd([_inst(2023, 8000), _inst(2024, 9000)]),
        "AssetsCurrent": usd([_inst(2023, 3000), _inst(2024, 4000)]),
        "LiabilitiesCurrent": usd([_inst(2023, 2000), _inst(2024, 2000)]),
        "StockholdersEquity": usd([_inst(2023, 4000), _inst(2024, 5000)]),
        "CashAndCashEquivalentsAtCarryingValue": usd([_inst(2023, 1000), _inst(2024, 1200)]),
        "LongTermDebtNoncurrent": usd([_inst(2023, 2200), _inst(2024, 2000)]),
        "LongTermDebtCurrent": usd([_inst(2023, 300), _inst(2024, 500)]),
    }
    return {"facts": {"us-gaap": gaap}}


def test_extract_annual_facts():
    b = edgar.extract_annual_facts(_facts())
    assert b["available"]
    assert b["fiscal_years"] == [2024, 2023]

    ri = b["roic_inputs"]
    assert ri["ebit"] == 1400
    assert ri["tax_rate"] == pytest.approx(300 / 1200)
    assert ri["total_debt"] == pytest.approx(2000 + 500)
    assert ri["book_equity"] == 5000
    assert ri["cash"] == 1200

    # ROIC computes cleanly from the EDGAR inputs.
    roic = F.compute_roic(ri["ebit"], ri["tax_rate"], ri["total_debt"], ri["book_equity"], ri["cash"])
    assert roic == pytest.approx(1400 * (1 - 0.25) / (2500 + 5000 - 1200))

    cur, prev = b["fscore_curr"], b["fscore_prev"]
    assert cur["gross_margin"] == pytest.approx((10000 - 5000) / 10000)
    assert cur["current_ratio"] == pytest.approx(4000 / 2000)
    assert cur["roa"] == pytest.approx(1000 / 9000)
    # A strong-improvement year should score high; just assert it runs and ranks well.
    score = F.piotroski_fscore(cur, prev)
    assert 6 <= score <= 9


def test_extract_annual_facts_insufficient_history():
    facts = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [_flow(2024, 1000)]}}}}}
    assert edgar.extract_annual_facts(facts) == {"available": False}
    assert edgar.extract_annual_facts({}) == {"available": False}


def test_fundamentals_bundle_failsoft(monkeypatch):
    # When the network/host is blocked, company_facts returns None -> unavailable.
    monkeypatch.setattr(edgar, "company_facts", lambda t: None)
    assert edgar.fundamentals_bundle("AAPL") == {"available": False}
