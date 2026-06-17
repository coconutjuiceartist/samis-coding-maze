# Opportunity Frontier

A trader/investor **relative-value scanner** built with Streamlit. It ranks
opportunities against each other rather than judging them in isolation,
building "frontiers" — the best return available at each level of risk — and
flagging the points that sit above the curve (value) or below it (avoid).

Two tabs:

- **⚡ Volatility & Options Frontier** — pulls VIX and option chains via
  `yfinance`, computes Greeks (Black-Scholes), re-solves implied vol from mid
  prices, fits each expiry's volatility skew, compares implied vol to trailing
  realised vol, and flags contracts that are RICH (sell) or CHEAP (buy)
  relative to their own surface and history.
- **🏛️ Fundamentals Frontier** — pulls fundamentals for a peer group, z-scores
  every metric within the group, builds Quality / Value / Growth / Safety
  composites, and plots a Quality-vs-Value frontier that surfaces the outliers
  (high quality **and** cheap).

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Data is free via `yfinance` (delayed/EOD). Put any paid API keys in
`.streamlit/secrets.toml` — never hard-code them.

See [`CLAUDE.md`](CLAUDE.md) for the full methodology, design philosophy, data
sources, and roadmap.

> ⚠️ **Educational tool, not investment advice.** Volatility understates tail
> risk — read the caveats in the app.

## License

MIT — see [`LICENSE`](LICENSE).
