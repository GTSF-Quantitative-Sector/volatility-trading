"""Command line for data ingestion and the VIX term-structure backtest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from .backtest import (
    COST_SENSITIVITY_BPS,
    DEFAULT_COST_BPS,
    RULES,
    first_complete_signal,
    headline_strategies,
    metrics_table,
    pair_grid_strategies,
    performance_metrics,
    returns_frame,
    run_suite,
    spy_weight_frame,
    stress_table,
    strategy_name,
)
from .data import INDEX_HISTORY_START, MassiveAPIError, MassiveConfigError, ingest_core_daily, market_today
from .public_data import PublicDataError, ingest_public
from .signals import MAIN_PAIR, build_panel, build_signals, tenor_pairs
from .stats import lead_lag_table, oos_r2, predictive_regressions, subperiod_table, walk_forward


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use YYYY-MM-DD") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VIX term-structure research pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser(
        "ingest", help="Download Cboe, FRED, and Yahoo history, plus Massive SPY bars."
    )
    ingest.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Default: data/raw")
    ingest.add_argument("--refresh", action="store_true", help="Re-download public files that already exist.")
    ingest.add_argument("--skip-massive", action="store_true", help="Skip the Massive SPY download.")
    ingest.add_argument(
        "--start",
        type=_parse_date,
        default=INDEX_HISTORY_START,
        help=f"Massive SPY start date. Default: {INDEX_HISTORY_START.isoformat()}",
    )
    ingest.add_argument(
        "--end", type=_parse_date, default=None, help="Massive SPY end date. Default: today in New York"
    )

    run = commands.add_parser("backtest", help="Build signals, run backtests and statistics, save results.")
    run.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Default: data/raw")
    run.add_argument("--results-dir", type=Path, default=Path("results"), help="Default: results")
    run.add_argument(
        "--cost-bps",
        type=float,
        default=DEFAULT_COST_BPS,
        help=f"Cost per unit of SPY/IEF turnover, in basis points. Default: {DEFAULT_COST_BPS:g}",
    )
    return parser.parse_args(argv)


def run_ingest(args: argparse.Namespace) -> int:
    try:
        results = ingest_public(args.output_dir, refresh=args.refresh)
    except (PublicDataError, OSError) as exc:
        print(f"public data download failed: {exc}", file=sys.stderr)
        return 1
    for result in results:
        status = "downloaded" if result.downloaded else "cached"
        print(f"{result.name:<12} {result.rows:>6} rows  {result.first} to {result.last}  {status}  {result.path}")

    if args.skip_massive:
        return 0
    end = args.end or market_today()
    if args.start > end:
        print("start date must be on or before end date", file=sys.stderr)
        return 2
    try:
        massive = ingest_core_daily(args.output_dir, args.start, end)
    except (MassiveConfigError, MassiveAPIError) as exc:
        print(f"Massive SPY skipped, the backtest does not need it: {exc}", file=sys.stderr)
        return 0
    for result in massive:
        print(f"massive:{result.ticker:<5} {result.rows:>6} rows  {result.path}")
    return 0


def _write_csv(frame: pd.DataFrame, path: Path, index: bool = True) -> None:
    frame.to_csv(path, index=index, lineterminator="\n", float_format="%.10g")


def run_backtest(args: argparse.Namespace) -> int:
    try:
        panel = build_panel(args.raw_dir)
    except FileNotFoundError as exc:
        print(f"missing raw data, run `python -m vol_trading ingest` first: {exc}", file=sys.stderr)
        return 1
    signals = build_signals(panel)
    cash = panel["cash_ret"]
    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    headline_start = first_complete_signal(
        signals, [f"ratio:{MAIN_PAIR}", f"z:{MAIN_PAIR}", "vix", "rv21"]
    )
    headline = run_suite(panel, signals, headline_strategies(), headline_start, args.cost_bps)
    headline_metrics = metrics_table(headline, cash)

    sensitivity = []
    for bps in COST_SENSITIVITY_BPS:
        table = metrics_table(run_suite(panel, signals, headline_strategies(), headline_start, bps), cash)
        sensitivity.append(table.assign(cost_bps=bps).reset_index())

    grid_start = first_complete_signal(
        signals, [f"{kind}:{pair}" for pair in tenor_pairs() for kind in ("ratio", "z")] + ["vix", "rv21"]
    )
    grid = run_suite(panel, signals, pair_grid_strategies(), grid_start, args.cost_bps)
    binary_candidates = {name: grid[name] for name in (strategy_name("binary", pair) for pair in tenor_pairs())}
    wf_returns, wf_selections = walk_forward(binary_candidates, cash, args.cost_bps)

    pair_rows = []
    for name, result in grid.items():
        rule, _, pair = name.partition(" ")
        is_rule = rule in RULES
        metrics = performance_metrics(result.returns, cash, result.turnover, result.costs)
        pair_rows.append({"strategy": name, "rule": rule if is_rule else "benchmark", "pair": pair if is_rule else "", **metrics})
    pair_rows.append({"strategy": wf_returns.name, "rule": "walk-forward", "pair": "", **performance_metrics(wf_returns, cash)})
    grid_returns = returns_frame(grid).join(wf_returns, how="left")

    regressions = predictive_regressions(panel, signals)
    oos = oos_r2(panel, signals)
    lead_lag = lead_lag_table(panel, signals, headline[strategy_name("binary", MAIN_PAIR)])
    subperiods = subperiod_table(headline, cash)
    stress = stress_table(headline)

    _write_csv(signals, out / "signals.csv")
    _write_csv(returns_frame(headline), out / "headline_returns.csv")
    _write_csv(spy_weight_frame(headline), out / "headline_spy_weights.csv")
    _write_csv(headline_metrics, out / "headline_metrics.csv")
    _write_csv(pd.concat(sensitivity, ignore_index=True), out / "cost_sensitivity.csv", index=False)
    _write_csv(stress, out / "stress.csv")
    _write_csv(subperiods, out / "subperiods.csv", index=False)
    _write_csv(lead_lag, out / "lead_lag.csv", index=False)
    _write_csv(grid_returns, out / "pair_returns.csv")
    _write_csv(pd.DataFrame(pair_rows), out / "pair_metrics.csv", index=False)
    _write_csv(wf_selections, out / "walk_forward_selections.csv", index=False)
    _write_csv(regressions, out / "regressions.csv", index=False)
    _write_csv(oos, out / "oos_r2.csv", index=False)
    info = {
        "main_pair": MAIN_PAIR,
        "cost_bps": args.cost_bps,
        "headline_first_signal": headline_start.strftime("%Y-%m-%d"),
        "pair_grid_first_signal": grid_start.strftime("%Y-%m-%d"),
        "data_end": panel.index[-1].strftime("%Y-%m-%d"),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")

    columns = ["cagr", "ann_vol", "sharpe", "max_drawdown", "calmar", "turnover_per_year"]
    print(f"Headline backtest, first signal {info['headline_first_signal']}, data through {info['data_end']}, {args.cost_bps:g} bps costs")
    print(headline_metrics[columns].round(3).to_string())
    print(f"\nResults written to {out}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "ingest":
        return run_ingest(args)
    return run_backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
