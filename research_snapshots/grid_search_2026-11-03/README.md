# VIX3M/VIX Grid Search — Results Snapshot

Run captured: 2026-11-03

This folder records actual outputs from the local `run_grid_search_experiment.py` run. It is a fixed research snapshot, separate from the generated `results/` directory, which remains gitignored.

## Run specification

- Signal: VIX3M / VIX (`3M/1M`)
- Rebalance: monthly signal, traded one session later
- Transaction cost: 5 bps
- Binary candidates: 54
- Bands candidates: 192
- Walk-forward minimum training history: 3 years
- Walk-forward parameter reselection: annually
- Walk-forward selection objective: net-of-cost Sharpe ratio
- Full-grid sample shown in the top-20 CSVs: 2010-10-01 through 2026-11-02
- Walk-forward OOS selection years: 2014 through 2026

## Files

- `walk_forward_comparison.csv`: common-window OOS strategy/benchmark metrics printed by the runner
- `binary_walk_forward_selections.csv`: Binary parameters selected for each OOS year
- `bands_walk_forward_selections.csv`: Bands parameters selected for each OOS year
- `binary_top20.csv`: top 20 Binary full-sample configurations by Sharpe
- `bands_top20.csv`: top 20 Bands full-sample configurations by Sharpe

The complete full grids and daily return series remain generated locally by `run_grid_search_experiment.py` under `results/grid_search/`.
