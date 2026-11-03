# VIX3M/VIX Grid Search and Walk-Forward Experiment

This experiment is intentionally isolated from the existing analysts' pipeline. It adds new files only and reuses the existing data, signal, simulation, benchmark, and metric helpers without editing them.

## What it tests

### Binary family

For `R = VIX3M / VIX`:

```text
R > threshold   -> high SPY weight
R <= threshold  -> low SPY weight
```

Grid:

- Threshold: 0.900 to 1.100 in 0.025 steps
- Low SPY weight: 0%, 25%, 50%
- High SPY weight: 75%, 100%
- Constraint: low weight < high weight

Total: 54 configurations.

### Bands family

```text
R > high threshold                      -> 100% SPY
low threshold <= R <= high threshold    -> middle SPY weight
R < low threshold                       -> low SPY weight
```

Grid:

- Low threshold: 0.900 to 1.000 in 0.025 steps
- High threshold: 1.000 to 1.100 in 0.025 steps
- Low SPY weight: 0%, 25%, 50%
- Middle SPY weight: 25%, 50%, 75%
- High SPY weight: 100%
- Constraints: low threshold < high threshold and low weight <= middle weight

Total: 192 configurations.

The unallocated portfolio weight remains in cash, matching the existing strategy convention.

## Walk-forward design

The full-sample grid is diagnostic only. It is useful for seeing whether a broad region of parameter choices works, but its best configuration is in-sample and should not be treated as evidence of a tradable edge.

The walk-forward test is the primary evaluation:

1. Start with at least three years of prior data.
2. At each year-end, score every candidate using only returns available through that year-end.
3. Select the candidate with the highest net-of-cost Sharpe ratio. Lower turnover breaks ties.
4. Use those parameters for all trades occurring in the next calendar year.
5. Repeat with an expanding training window.
6. Stitch the selected target weights together and pass them through one call to the existing simulator so turnover and transaction costs are handled consistently across year boundaries.

The December month-end signal is assigned to the next year when its next-session trade occurs in January.

## Run

First make sure the existing raw data are present. If necessary:

```bash
python -m vol_trading ingest
```

Then run:

```bash
python run_grid_search_experiment.py
```

Optional arguments:

```bash
python run_grid_search_experiment.py --cost-bps 10 --min-train-years 4
```

Results go to:

```text
results/grid_search/
```

This does not overwrite the existing `results/` tables.

## Main output files

- `binary_full_grid.csv`: every Binary configuration and full-sample metrics
- `bands_full_grid.csv`: every Bands configuration and full-sample metrics
- `binary_top20.csv`, `bands_top20.csv`: convenient diagnostic shortlists
- `binary_walk_forward_selections.csv`: parameters selected for each OOS year
- `bands_walk_forward_selections.csv`: parameters selected for each OOS year
- `walk_forward_comparison.csv`: the main honest comparison against fixed rules and benchmarks
- `walk_forward_returns.csv`: daily returns on the common OOS window
- `walk_forward_spy_weights.csv`: SPY exposure of the two walk-forward strategies
- `full_sample_comparison.csv`: includes best full-sample configurations, clearly in-sample
- `run_info.json`: run settings and sample dates

Interactive HTML diagnostics:

- `binary_sharpe_surface.html`
- `bands_sharpe_surface.html`
- `walk_forward_growth.html`
- `walk_forward_parameter_stability.html`

## How to judge the result

Focus on `walk_forward_comparison.csv`, not the best full-sample Sharpe.

Compare:

- Walk-forward Binary
- Walk-forward Bands
- Fixed Binary at 1.00 with 0%/100% weights
- Fixed Bands at 0.95/1.05 with 0%/50%/100% weights
- SPY buy-and-hold
- 60/40 SPY/IEF
- VIX < 20 filter
- 15% volatility target

All strategies in that table are evaluated on the exact same return window.

Key questions:

1. Does walk-forward tuning improve Sharpe or Calmar versus the fixed strategy?
2. Does it preserve or improve maximum drawdown?
3. Are the selected thresholds and weights stable across years?
4. Is there a broad region of good parameters, or one isolated full-sample winner?
5. Does any additional return justify the extra turnover and complexity?

A stable plateau of good parameters plus strong walk-forward results is much more convincing than a single exceptional in-sample configuration.
