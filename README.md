# volatility-trading

Research project testing whether the VIX term structure can time SPY exposure. It downloads free Cboe volatility-index history, backtests monthly SPY allocation rules from 2010, and runs predictive regressions. Strategy context and conventions for contributors and agents are in [AGENTS.md](AGENTS.md).

## Setup

```bash
conda create -n vol-trading python=3.12 -y
conda activate vol-trading
pip install -r requirements.txt
```

The backtest needs no API key. To also download Massive SPY bars, create `.env` and set the key:

```bash
cp .env.example .env
```

```bash
MASSIVE_API_KEY=your_key_here
```

`POLYGON_API_KEY` still works as a fallback. Do not commit the key, `data/`, or `results/`.

## File structure

```text
volatility-trading/
├── vol_trading/                 Python package with all pipeline code
│   ├── __init__.py
│   ├── __main__.py              lets `python -m vol_trading` run the command line
│   ├── main.py                  command line: `ingest` and `backtest`
│   ├── public_data.py           downloads free Cboe, FRED, and Yahoo history
│   ├── data.py                  Massive API client and SPY bar download
│   ├── signals.py               daily panel, term-structure ratios, z-scores
│   ├── backtest.py              monthly rebalance engine, rules, benchmarks, metrics
│   └── stats.py                 regressions and robustness checks
├── tests/
│   └── test_backtest.py         checks for trade timing, look-ahead, costs, metrics, parsers
├── notebooks/
│   └── term_structure_backtest.ipynb   charts and tables built from results/
├── polygon-api-demo.ipynb       reference notebook for Massive endpoints
├── data/raw/                    downloaded CSVs, generated, not committed
├── results/                     backtest output tables, generated, not committed
├── AGENTS.md                    strategy context and conventions for agents
├── requirements.txt
├── pytest.ini
└── .env.example                 template for the Massive API key
```

What each Python file does:

- `main.py` parses the command line. `ingest` calls the downloaders and saves CSVs to `data/raw/`. `backtest` builds the signals, runs every strategy and statistic, prints the headline metrics, and writes the tables to `results/`.
- `public_data.py` fetches the five Cboe volatility indexes (VIX9D, VIX, VIX3M, VIX6M, VIX1Y), the FRED 3-month T-bill yield, and Yahoo dividend-adjusted closes for SPY and IEF. It cleans dates and placeholder prints and reuses saved files unless asked to refresh.
- `data.py` is the Massive REST client: it loads the key from `.env`, caches responses without storing the key, follows pagination, and writes daily SPY bars. The backtest does not depend on it.
- `signals.py` aligns everything to SPY trading days, computes daily SPY, IEF, and cash returns, and builds the ratio and 252-day z-score for every tenor pair, such as `3M/1M` for VIX3M / VIX. It also finds each completed month's last trading day.
- `backtest.py` turns month-end signals into SPY/IEF/cash weights traded one session later, simulates daily returns with weight drift and turnover costs, and defines the binary, band, and z-score rules plus the SPY, 60/40, VIX-filter, and volatility-target benchmarks. It also computes CAGR, Sharpe, Sortino, drawdown, Calmar, turnover, and stress-window returns.
- `stats.py` runs predictive regressions of forward SPY returns and realized volatility on the signals with Newey-West errors, out-of-sample R-squared, the stress-period lead and re-entry table, subperiod metrics, and the walk-forward pair selection.

## Run

Run commands from the repository root:

```bash
python -m vol_trading ingest
python -m vol_trading backtest
```

`ingest` writes these files to `data/raw/` and reuses them on later runs; add `--refresh` to download again:

- Cboe `VIX9D`, `VIX`, `VIX3M`, `VIX6M`, and `VIX1Y` daily closes
- FRED `DTB3`, the 3-month T-bill yield used as the cash return
- Yahoo dividend-adjusted closes for `SPY` and `IEF`
- Massive `SPY.csv`, skipped with a warning if the key is missing or rejected, or with `--skip-massive`

Massive returns `403` for its VIX and VIX3M indexes on the current key, which is why volatility data comes from Cboe.

`backtest` prints the headline metrics and writes every table to `results/`. Change the cost assumption with `--cost-bps 10`.

Open [notebooks/term_structure_backtest.ipynb](notebooks/term_structure_backtest.ipynb) with the `vol-trading` kernel for the P&L and drawdown chart, the Sharpe heatmap across tenor pairs, the walk-forward comparison, the regression coefficients, and the stress-period timing tables.

Run the tests with:

```bash
python -m pytest
```

`polygon-api-demo.ipynb` remains a reference for the broader Massive endpoints.
