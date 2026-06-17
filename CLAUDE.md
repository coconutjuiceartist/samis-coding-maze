# CLAUDE.md — Opportunity Frontier

**You are extending this project with Claude Code.** This file is the spec. It explains *what to build, why, and how* — the methodology and the practitioner principles are deliberately encoded here so you (Claude Code) can extend the tool without re-deriving them. `app.py` is a working starter that implements the core; treat it as the reference implementation and grow it along the roadmap below.

-----

## 1. Mission

A relative-value scanner that helps a trader/investor **rank opportunities against each other**, not judge them in isolation. The governing idea, learned the hard way: *a trade is only as good as the best alternative use of the same capital and risk.* The tool builds “frontiers” — the best return available at each level of risk — and flags the points that sit above the curve (value) or below it (avoid).

Two surfaces, two tabs:

- **Tab 1 — Volatility & Options Frontier.** Where is option premium rich (sell) vs cheap (buy), relative to (a) each name’s own volatility skew, (b) its trailing realised volatility, and (c) the VIX regime?
- **Tab 2 — Fundamentals Frontier.** Which companies are outliers — high business quality *and* cheap — relative to a peer group, on the metrics a principal investor actually underwrites (margins, cash generation, returns on capital, leverage)?

It must stay **simple to use** (type tickers, get answers), with data **fresh enough to act on** (delayed/EOD is fine; intraday-real-time is not required).

-----

## 2. Design philosophy — the practitioner lineage

Each principle below maps to a concrete feature. Paraphrased; attributions are to orient the design, not to quote.

**Lloyd Blankfein (Goldman / J. Aron → FICC, and the principal-investing side).** Two lessons. (1) *You get paid to take the other side of risk others want to shed* — selling options is intermediation, and the premium is your spread; but a desk only warehouses risk it can hedge and exit, so **liquidity (open interest, bid/ask) is a first-class column, not an afterthought.** (2) *Sell insurance when it’s dear, not cheap* — so the tool foregrounds the VIX regime and implied-vs-realised vol, because *when* you sell premium matters as much as *what* you sell. On the investment side, Goldman’s merchant/principal bets keyed on cash generation and balance-sheet strength → Tab 2’s Quality and Safety composites.

**Nassim Taleb.** Selling premium is a negatively-skewed payoff (“pennies in front of a steamroller”): frequent small gains, rare large losses. Volatility *understates* the risk of a short option. → The yield-vs-|delta| frontier carries a permanent caveat that |delta| is not the real risk; the real risk is the tail. Encourage buying convexity when it’s cheap as much as selling it when it’s rich.

**Paul Tudor Jones.** Hunt asymmetry — risk 1 to make 5; play defense first. → Surface *both* sides: a “buy convexity” view (cheap long options when IV-rank is low) alongside the premium-selling view. Always show the downside/max-loss framing.

**Howard Marks.** Risk is the probability of *permanent loss of capital*, not volatility; the key question is “how much could I lose if I’m wrong?” → Show max-loss and assignment scenarios, not just yield. In Tab 2, value without quality is a trap.

**Joel Greenblatt (Magic Formula).** Rank on the *combination* of return on capital and earnings yield — good business, cheap price. → Tab 2’s `MF_rank` and the Quality-vs-Value frontier are this idea.

**Aswath Damodaran.** A company creates value only when **ROIC > cost of capital**; growth without that is value-destroying. → Roadmap: compute true ROIC and ROIC−WACC (the starter uses ROE/ROA as a robust proxy; upgrade this).

**Buffett / Graham.** Quality + durable margins + margin of safety. → Quality composite (margins, FCF) and the insistence on cheapness alongside it.

**Joseph Piotroski.** A 9-point fundamental-health checklist (profitability, leverage, efficiency). → Roadmap: add an F-score column.

**Ed Seykota / Mark Douglas.** Think in probabilities over a series; define and accept risk before entering; cut losses. → The tool is a *screen* that produces candidates and risk framing, never a “do this” button.

**Ed Thorp / Kelly.** Size by edge, and never enough to risk ruin. → Roadmap: position-sizing and a portfolio-level frontier with correlation.

-----

## 3. Data sources

**Free (default, implemented): `yfinance`.**

- Chains: `yf.Ticker(t).options` → expiries; `.option_chain(exp)` → `.calls` / `.puts` DataFrames with `strike, bid, ask, lastPrice, volume, openInterest, impliedVolatility, inTheMoney`. **Greeks are NOT provided → compute via Black-Scholes** (done in `app.py`). Yahoo’s `impliedVolatility` field is often stale/garbage for illiquid strikes → the app **re-solves IV from the mid price** and only falls back to Yahoo’s field.
- Spot/history: `.history(period=...)`, `.fast_info`. VIX: `^VIX`. Risk-free: `^IRX` (13-wk T-bill, in %).
- Fundamentals: `.info` (ratios), `.income_stmt`, `.balance_sheet`, `.cashflow`.
- Caveat: unofficial, can rate-limit or break when Yahoo changes endpoints. Cache aggressively (already done) and handle failures gracefully.

**Paid / more reliable upgrades (wire behind a `DataProvider` interface so they’re swappable):**

- **Tradier** (developer API, delayed quotes free-ish, **includes Greeks & IV**) — good first upgrade for options.
- **EODHD** (~$100/mo, 30y history, Greeks/IV) and **Databento** (usage-based, institutional-grade) for serious work.
- **Alpha Vantage** (`HISTORICAL_OPTIONS` with Greeks/IV; free tier limited) and **Polygon.io** (free tier, options + reference).
- **SEC EDGAR `companyfacts`** (free, no key, authoritative) for fundamentals — preferred over yfinance for accuracy; use to compute clean ROIC/WACC.

> **First refactor for Claude Code:** extract data access into `providers/` with a `DataProvider` protocol (`spot`, `history`, `chain`, `fundamentals`) and a yfinance implementation, so a Tradier/EODHD provider can be dropped in via an env var / `.streamlit/secrets.toml`.

-----

## 4. Tab 1 — Volatility & Options Frontier (methodology)

**Inputs:** tickers (comma-separated), expiries-per-name, side (puts/calls/both). Show the risk-free rate and VIX (level + 1-year percentile) up top, with a one-line regime read (low VIX ⇒ optionality cheap ⇒ favour buying convexity; high ⇒ favour selling premium).

**Per-contract enrichment** (implemented): mid, **own-solved IV**, BS Greeks (delta/gamma/theta/vega), DTE, %OTM, log-moneyness, **annualised cash-secured-put yield** = `mid/(strike−mid) × 365/DTE`, bid/ask spread %, open interest, volume, and `VRP = IV − trailing realised vol`.

**Three “frontiers” / signals:**

1. **Skew frontier (today, relative value within the surface).** Per (ticker, expiry), fit IV vs log-moneyness (deg-2). The fitted curve *is* the local frontier; the **residual** (actual IV − fitted) flags a contract as rich (+) or cheap (−) vs its own smile.
1. **Realised-vol frontier (the “200-day” comparison).** Compare each contract’s IV to the underlying’s **trailing realised volatility** (the starter uses 60d; expose 20/60/200d). `IV ≫ RV` ⇒ rich (sell); `IV ≪ RV` ⇒ cheap (buy). This is the implementable stand-in for “vs the 200-day”: realised vol has a real 200-day series, whereas historical *implied* vol does not come from yfinance. **Roadmap:** snapshot ATM IV daily into a local SQLite so a true **IV Rank/Percentile vs trailing 252d** becomes available over time.
1. **Yield-vs-risk frontier (the capital view).** Scatter annualised CSP yield (y) vs |delta| (x) for all puts; the upper envelope is the frontier; `value_ratio = yield / frontier_yield_at_that_delta` (>1 = pays more than peers for the same risk). Draw the risk-free line as the floor.

**Composite `richness`** = mean of standardised skew-residual and standardised VRP → `RICH → sell`, `CHEAP → buy`, or `fair`. The opportunities table sorts by richness and **always shows `spread_%` and `open_int`** so thin/wide contracts are visible.

**Hard caveat (must remain in the UI):** |delta| is a poor proxy for the tail risk of a short option; “RICH → sell” means “where the premium is,” not “free money.” Confirm liquidity before acting.

-----

## 5. Tab 2 — Fundamentals Frontier (methodology)

**Inputs:** a peer group (comma-separated). Everything is **z-scored within the peer set**, so the comparison is explicitly relative — changing the peers reframes the screen.

**Metric panel** (implemented, from yfinance `.info`): revenue growth; gross/operating/net margins; FCF margin; ROE/ROA; FCF yield; earnings yield; EV/EBITDA, P/E, P/B; net-debt/EBITDA, debt/equity, current ratio.

**Composites (peer-relative z):**

- **Quality** = margins + FCF margin + ROE + ROA (Buffett/Blankfein: good businesses generate cash and high returns).
- **Value** = FCF yield + earnings yield − EV/EBITDA − P/E − P/B (Greenblatt/Graham: cheap on what the business throws off).
- **Growth** = revenue growth (extend with earnings/FCF growth and durability).
- **Safety** = current ratio − net-debt/EBITDA − debt/equity (balance-sheet strength).
- **Composite** = Quality + Value + ½Growth + ½Safety.
- **`MF_rank`** = rank(ROE) + rank(earnings yield) — Greenblatt’s Magic Formula, approximated with `.info` fields.

**Frontier viz:** Quality (x) vs Value (y) scatter; bubble = market cap; colour = growth. **Top-right = high quality AND cheap = the outliers** (flagged with ★ when both z-scores > 0.5). This is the literal “companies that emerge as outliers relative to peers.”

**Roadmap upgrades (high value, in priority order):**

1. **True ROIC and ROIC−WACC** (Damodaran): NOPAT = EBIT×(1−tax); invested capital = debt + equity − cash; WACC from beta/cost-of-debt/weights. Plot ROIC−WACC as the real value-creation axis.
1. **Piotroski F-score** column.
1. **Multi-year trends** (3–5y CAGRs, margin trajectory, capital-cycle/ROIC trend à la Marathon) from `.income_stmt`/`.cashflow`, ideally via **SEC EDGAR** for accuracy.
1. **Sector-aware peer auto-suggestion** (so the user types one ticker and gets a sensible peer set).

-----

## 6. Cross-tab “available risk frontier” (north-star feature, not yet built)

The most powerful version unifies both tabs onto one **expected-excess-return vs risk** chart spanning cash, Treasuries, IG/HY credit (pull yields), equities (earnings yield), the user’s actual option positions (parse a CSV of holdings), and screened candidates — so a single picture answers “where does this trade rank among everything?” Penalise concave/short-vol payoffs and reward convex ones beyond what volatility-based Sharpe implies (Taleb/Marks). This is the synthesis the whole tool is building toward.

-----

## 7. Build & run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Put any paid API keys in `.streamlit/secrets.toml` (never hard-code). Keep caching on; add a `data_timestamp` to every panel so freshness is visible.

-----

## 8. Engineering conventions for Claude Code

- **Math is sacred:** `bs()` and `implied_vol()` are validated; don’t alter the formulas without a numeric test (ATM call S=K=100,T=1,r=0,σ=0.2 ⇒ price≈7.97, delta≈0.540).
- **Fail soft:** every network call wrapped in try/except; a dead ticker must never crash the app.
- **Cache with TTLs:** options/spot ~15 min, fundamentals ~6 h; expose a Refresh button.
- **Provider abstraction** before adding any new data source (see §3).
- **Tests:** add `pytest` for the pure functions (Greeks, IV solver, z-scores, frontier binning, composites) using synthetic data — these need no network.
- **Never present an output as advice.** Surface candidates + risk framing; keep the tail-risk and liquidity caveats visible.

-----

## 9. Definition of done (per feature)

A feature is done when: it pulls fresh data and degrades gracefully on failure; it produces a ranked/plotted output a trader can act on; it shows the *risk* alongside the *reward* (max loss, liquidity, or peer context); and the relevant practitioner caveat is visible in the UI. Simple in, insight out.