# Opportunity Frontier

A relative-value scanner that ranks opportunities **against each other** rather
than judging them in isolation — *a trade is only as good as the best
alternative use of the same capital and risk.* See [`CLAUDE.md`](CLAUDE.md) for
the full methodology and the practitioner lineage behind each feature.

## What it does

Three surfaces, three tabs:

1. **Volatility & Options Frontier** — where option premium is rich (sell) vs
   cheap (buy), relative to each name's own skew, its trailing realised vol, and
   the VIX regime. Surfaces the skew frontier, the IV-vs-RV frontier, and the
   yield-vs-|delta| capital frontier, with a composite *richness* score. Greeks
   and IV are computed locally via Black-Scholes (Yahoo's IV field is only a
   fallback).
2. **Fundamentals Frontier** — which companies are outliers (high quality *and*
   cheap) vs a peer group. Peer-relative z-scored Quality / Value / Growth /
   Safety composites, the Magic-Formula rank, plus best-effort **ROIC−WACC**
   (Damodaran) and a **Piotroski F-score**.
3. **Available Risk Frontier** — every candidate use of capital (cash,
   Treasuries, IG/HY credit, equities, screened options, your own holdings) on
   one expected-excess-return vs risk chart. Convex payoffs are rewarded and
   concave (short-vol) ones penalised beyond plain Sharpe (Taleb / Marks).

## Layout

```
opportunity-frontier/
├── app.py                  # Streamlit UI (3 tabs, caching + freshness here)
├── core/                   # pure, network-free analytics (unit-tested)
│   ├── options.py          # Black-Scholes, Greeks, IV solver, chain enrichment
│   ├── vol.py              # realised vol, skew fit, frontiers, richness, regime
│   ├── fundamentals.py     # metrics, composites, ROIC/WACC, Piotroski F-score
│   ├── frontier.py         # cross-tab unified risk frontier
│   ├── iv_store.py         # local SQLite ATM-IV snapshots -> IV Rank/Percentile
│   ├── peers.py            # sector-aware peer auto-suggestion
│   └── stats.py            # z-scores, ranks, percentiles
├── providers/              # swappable data sources behind a DataProvider protocol
│   ├── base.py             # the protocol (spot/history/chain/fundamentals/…)
│   └── yfinance_provider.py# free default
└── tests/                  # pytest for the pure functions (no network needed)
```

## Run

```bash
cd opportunity-frontier
pip install -r requirements.txt
streamlit run app.py
```

## Test

```bash
cd opportunity-frontier
pip install -r requirements-dev.txt
pytest
```

## Swapping data providers

Data access lives behind the `DataProvider` protocol in `providers/base.py`. The
free default is yfinance. To add Tradier / EODHD / Polygon, implement the
protocol, register it in `providers/__init__.py`, and select it via the
`OPPORTUNITY_PROVIDER` env var or `.streamlit/secrets.toml`
(see `.streamlit/secrets.toml.example`).

## Caveats (kept visible in the UI)

- `|delta|` is **not** the real risk of a short option — the tail is. "RICH →
  sell" means *where the premium is*, not free money. Confirm liquidity (spread
  %, open interest) before acting.
- yfinance is unofficial and can rate-limit; everything fails soft and caches.
- This is a **screen** that produces candidates + risk framing. Never advice.
