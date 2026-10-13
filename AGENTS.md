# Volatility Trading

This repository studies whether the VIX term structure can time SPY exposure and, later, volatility carry. The source brief is the shared outline, [Volatility Trading Outline](https://docs.google.com/document/d/1QibTIo-fmcqWgU5nyo8870P_eWuorh1aW3qIFOUpCh4/edit). Read this file before changing strategy, data, or backtest code.

## Current milestone

Baseline research, phases 3 and 4 of the outline: monthly SPY allocation backtests driven by Cboe volatility-index ratios from 2010, the benchmarks, predictive regressions, and robustness checks. Refinement (phase 5) and any trading integration wait until these baselines are reviewed. VIX futures are not in the dataset yet.

## Repository layout

All pipeline code lives in the `vol_trading/` package and uses relative imports. Run it from the repository root with `python -m vol_trading`.

- `vol_trading/main.py` has two commands: `ingest` downloads data, `backtest` builds signals, runs every backtest and statistic, and writes `results/`.
- `vol_trading/public_data.py` downloads the free Cboe, FRED, and Yahoo history.
- `vol_trading/data.py` is the Massive HTTP client, bar normalizer, and CSV writer. It now fetches only `SPY`.
- `vol_trading/signals.py` builds the daily panel and the tenor-pair ratios and z-scores.
- `vol_trading/backtest.py` is the monthly rebalance engine, the strategy rules, the benchmarks, and the metrics.
- `vol_trading/stats.py` holds the predictive regressions, out-of-sample R-squared, stress lead/lag table, subperiods, and walk-forward.
- `notebooks/term_structure_backtest.ipynb` reads `results/` and draws the P&L, drawdown, tenor heatmap, and regression charts. Do not commit it with outputs.
- `tests/` holds pytest checks for trade timing, z-score windows, costs, metrics, and parsers. Run `python -m pytest`.
- `polygon-api-demo.ipynb` is a reference for Massive endpoints, not part of the pipeline.
- `.env` holds the local API key and is never committed. Copy `.env.example`.
- `data/raw/`, `results/`, and `.massive_cache/` are generated and never committed.

## Secrets and API access

Put the key in `.env` as `MASSIVE_API_KEY`. `POLYGON_API_KEY` is accepted only as a fallback because Massive was formerly Polygon. The client sends the key in the `Authorization: Bearer` header. It strips `apiKey` query parameters and does not cache them.

Never print, log, commit, or paste the key or licensed response bodies into docs, notebooks, issues, or chat. Indices require a Massive entitlement, and the current key gets `403` for `I:VIX` and `I:VIX3M`. Do not work around a `403` by embedding another credential; the public Cboe files below need none.

Useful overrides, all optional:

- `MASSIVE_BASE_URL` defaults to `https://api.massive.com`.
- `MASSIVE_CACHE_DIR` defaults to `.massive_cache`.
- `MASSIVE_CACHE_MODE` is `auto`, `live`, or `cache`. The CLI default is `auto`.

## Data sources

The backtest uses only free public files, saved to `data/raw/` and reused unless `ingest --refresh` is passed:

| File | Source | History |
| --- | --- | --- |
| `cboe_VIX9D.csv` | Cboe `VIX9D_History.csv` | 2011-01-04 |
| `cboe_VIX.csv` | Cboe `VIX_History.csv` | 1990 |
| `cboe_VIX3M.csv` | Cboe `VIX3M_History.csv` | 2009-09-18 |
| `cboe_VIX6M.csv` | Cboe `VIX6M_History.csv` | 2008 |
| `cboe_VIX1Y.csv` | Cboe `VIX1Y_History.csv` | 2007 |
| `fred_DTB3.csv` | FRED 3-month T-bill yield | 1954 |
| `yahoo_SPY.csv`, `yahoo_IEF.csv` | `yfinance` close and dividend-adjusted close | 2005 |

Cboe URLs follow `https://cdn.cboe.com/api/global/us_indices/daily_prices/{SYMBOL}_History.csv`. Zero or negative Cboe prints are placeholders and are dropped. FRED drops connections for some custom `User-Agent` strings, so `public_data.py` keeps the `requests` default and retries. Yahoo rows dated today in New York are dropped because that session is unfinished. Yahoo is an unofficial source; if it breaks, replace it with another total-return series rather than Massive's split-only SPY.

Massive `SPY.csv` is still downloaded by `ingest` for later use, starting `2023-02-14` by default. It is not used by the backtest. Massive bars use `adjusted=true`, which adjusts splits, not dividends. Columns, in order: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `vwap`, `transactions`, `ticker`, with ISO 8601 Eastern timestamps. Cache successful GET responses only; aggregates expire after 5 minutes, reference paths after 12 hours, and snapshots after 30 seconds.

## Backtest conventions

- The panel is indexed by SPY trading days. Volatility indexes, IEF, and the T-bill yield are forward-filled by at most one day. Cboe holiday prints on days SPY did not trade are dropped.
- Tenors are `9D`, `1M` (VIX), `3M`, `6M`, and `12M`. A pair `long/short` is the longer index over the shorter, for all 10 combinations. The main pair is `3M/1M`, the outline's `VIX3M / VIX`.
- Z-scores use a trailing 252-day window ending on the signal day.
- Signals are read at the last trading day of each completed month and traded at the next session's close. VIX closes at 4:15 pm and SPY at 4:00 pm, so same-close trading would be look-ahead. A trade day's return accrues to the old weights.
- Cash earns the prior day's DTB3 yield, accrued per calendar day on a 360-day year. Costs are 5 basis points per unit of SPY plus IEF turnover by default, with 0, 5, and 10 reported.
- Rules: `binary` holds SPY when the ratio is above 1; `bands` holds 0, 0.5, or 1 below 0.95, between 0.95 and 1.05, and above 1.05; `zscore` holds 1 at z of 0 or higher, 0.5 from -1 to 0, and 0 below -1.
- Benchmarks: SPY buy-and-hold, 60/40 SPY/IEF rebalanced monthly, SPY when VIX is below 20, and SPY sized to 15% volatility from 21-day realized volatility, capped at 1.
- The headline run starts at the first month end with the `3M/1M` z-score, 2010-09-30. The tenor grid starts at the first month end with every pair's z-score, 2012-01-31. Do not compare Sharpe ratios across those two windows.
- Regressions use daily overlapping forward windows, standardized regressors, and Newey-West errors with lags equal to the horizon. Out-of-sample forecasts at month end train only on days whose forward window had closed.

## Run

```bash
conda activate vol-trading
pip install -r requirements.txt
cp .env.example .env
python -m vol_trading ingest
python -m vol_trading backtest
python -m pytest
```

`ingest --skip-massive` skips the Massive download. If the Massive key is missing or rejected, `ingest` prints a warning and still succeeds because the backtest does not need it.

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

VIX futures are still missing. They use Massive's futures aggregate endpoint and need explicit roll rules before any continuous series is built.

Rebuilding the curve from Massive SPY options is not planned. Options day aggregates start 2014-06-02, so they cannot cover 2010 to 2013. They hold closing trades rather than bid-ask midpoints, and quotes start only in 2022. Per-contract REST would take millions of requests, so any rebuild would need flat files and separate S3 credentials. SPY options are American and carry dividends, so the result would not match the published indexes. Treat such a rebuild only as a cross-check.

Portfolio-manager and shared-backtester integration waits until the signal and execution assumptions are specified.
