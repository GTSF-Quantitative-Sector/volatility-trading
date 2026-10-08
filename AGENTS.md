# Volatility Trading

This repository studies whether the VIX term structure can time SPY exposure and, later, volatility carry. The source brief is the shared outline, [Volatility Trading Outline](https://docs.google.com/document/d/1QibTIo-fmcqWgU5nyo8870P_eWuorh1aW3qIFOUpCh4/edit). Read this file before changing strategy, data, or backtest code.

## Current milestone

Data ingestion only. Download and store daily bars for the core series:

- `SPY` equity bars
- `I:VIX` CBOE Volatility Index
- `I:VIX3M` CBOE 3-Month Volatility Index

Do not add signals, backtests, portfolio rules, or a trading integration until this dataset is reliable. `polygon-api-demo.ipynb` is a reference for Massive endpoints. Production ingestion lives in `data.py` and `main.py`.

## Repository layout

- `main.py` runs the daily core-series download.
- `data.py` is the Massive HTTP client, bar normalizer, and CSV writer.
- `.env` holds the local API key and is never committed. Copy `.env.example`.
- `data/raw/` holds downloaded CSVs and is never committed.
- `.massive_cache/` is the local HTTP cache and is never committed.

## Secrets and API access

Put the key in `.env` as `MASSIVE_API_KEY`. `POLYGON_API_KEY` is accepted only as a fallback because Massive was formerly Polygon. The client sends the key in the `Authorization: Bearer` header. It strips `apiKey` query parameters and does not cache them.

Never print, log, commit, or paste the key or licensed response bodies into docs, notebooks, issues, or chat. Indices require a Massive entitlement. A `403` means the key cannot read that endpoint; do not work around it by embedding another credential.

Useful overrides, all optional:

- `MASSIVE_BASE_URL` defaults to `https://api.massive.com`.
- `MASSIVE_CACHE_DIR` defaults to `.massive_cache`.
- `MASSIVE_CACHE_MODE` is `auto`, `live`, or `cache`. The CLI default is `auto`.

## Data conventions

The default window starts on `2023-02-14`, the first date Massive documents for index aggregates, and ends on the current America/New_York date. SPY history on Massive is longer, but the aligned core download uses the shared start so VIX and VIX3M are not silently empty. The outline's 2010-onward study needs an additional historical source later. Do not backfill or fabricate that history from the Massive responses.

Daily bars come from `GET /v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}`. SPY uses `adjusted=true`, which adjusts splits, not dividends. It is not a total-return series. Index requests omit `adjusted` because that parameter is not part of the index aggregate endpoint. Timestamps are Unix milliseconds converted to America/New_York. Missing index volume stays missing; do not fill it with zero.

Each series is written to its own CSV under the output directory:

- `SPY.csv`
- `I_VIX.csv`
- `I_VIX3M.csv`

Columns, in order: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `vwap`, `transactions`, `ticker`. Rows are sorted by timestamp, duplicate timestamps keep the last row, and timestamps are ISO 8601 with the Eastern offset. An empty API result still writes the header.

Cache successful GET responses only. Snapshot and market-status paths expire after 30 seconds, reference and corporate-action paths after 12 hours, and other paths, including aggregates, after 5 minutes. `cache_mode=live` refreshes. `cache_mode=cache` fails when no cached response exists.

## Run ingestion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
python main.py --start 2024-01-01 --end 2024-12-31 --output-dir data/raw
```

## Strategy hypothesis

The primary signals in the outline are `VIX3M / VIX` and the front-month versus second-month VIX futures slope.

- Contango, ratio above 1 or an upward slope, means near-term implied volatility is cheaper than longer-term volatility and usually coincides with calm markets.
- A flattening curve means near-term risk is rising.
- Backwardation, ratio below 1 or an inverted slope, means near-term fear exceeds longer-term fear and often coincides with selloffs.

The economic claim is twofold. Persistent demand for downside protection creates a variance risk premium, and the term structure says when that risk is concentrated. The first implementation should change SPY exposure, adding equity in favorable regimes and moving toward cash or Treasuries as the curve flattens or inverts. A direct short-volatility futures carry book comes only after the equity-timing signal is robust.

## Research phases

1. Literature and market structure: VIX term structure, variance risk premium, volatility carry, futures pricing, expiration, convergence, and roll mechanics.
2. Data and signals: collect SPY total return, VIX, VIX3M, VIX futures, risk-free rates, and benchmark assets. Compare `VIX3M/VIX`, a normalized spread, futures slope, rolling z-scores, and smoothed variants.
3. Statistical tests: relate today's signal to later SPY returns, realized volatility, drawdowns, and downside tails across horizons. Compare against the VIX level and realized volatility.
4. Baseline backtests: long SPY when `VIX3M/VIX > 1`, otherwise cash; z-score allocation; multi-level exposure. Benchmarks are SPY buy-and-hold, 60/40, a simple VIX-level filter, and volatility targeting.
5. Refinement, only if the baseline is promising: thresholds, smoothing, hysteresis, continuous sizing, daily versus weekly versus monthly rebalancing, cash versus Treasury defense, and combining the VIX level with term structure.
6. Robustness: parameter sensitivity, subperiod stability, crisis behavior, costs, turnover, and walk-forward performance. Select on robustness, not the best in-sample Sharpe.

Portfolio variants to test later are binary risk-on/risk-off, multi-regime weights, continuous sizing, alternate rebalance frequencies, and hysteresis. A required timing check is whether the signal changes before or early in stress, and whether it re-enters in time for recoveries.

## Success metrics

The outline's goal is a defensible systematic strategy and an Investment Committee pitch. Absolute return does not need to beat SPY when risk-adjusted return or drawdown protection is materially better.

Report CAGR, Sharpe, Sortino, maximum drawdown, Calmar, annualized volatility, turnover, transaction costs, and performance in major stress periods. The finished strategy needs a clear economic rationale, explicit rules, realistic execution, competitive risk-adjusted performance, better drawdowns, stability across reasonable parameters, and credible out-of-sample results.

## Guardrails

- Backwardation can arrive after equities have already fallen. Measure lead time instead of assuming the signal is predictive.
- Reduced exposure can miss fast recoveries. Measure re-entry delay.
- Thresholds can whipsaw. Report turnover after costs.
- Many windows and thresholds are available. Keep specifications simple and economically motivated, and limit tuning.
- Crisis samples are small. Do not treat one path as a large independent sample.
- Option-market structure, 0DTE flow, and volatility products can break older relationships.
- Avoid look-ahead bias, incorrect futures stitching, mismatched calendars, and unrealistic fills. Use `filing_date` when statements eventually enter the research. Do not treat option last trades or fair-market-value marks as executable prices.
- Open interest is prior-session data. Leave missing Greeks, implied volatility, volume, and open interest missing.
- Do not use the club key or cached licensed data outside this project.

## Later data, not this milestone

Phase 2 still needs SPY total return, VIX futures, a risk-free series, and benchmark assets such as the 60/40 mix. VIX futures use Massive's futures aggregate endpoint and need explicit roll rules before any continuous series is built. Flat files, not repeated REST pagination, are the intended route for large intraday backtests. Portfolio-manager and shared-backtester integration waits until the signal and execution assumptions are specified.
