"""Run the isolated VIX3M/VIX grid-search and walk-forward experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vol_trading.grid_search_experiment import (
    DEFAULT_MIN_TRAIN_YEARS,
    DEFAULT_START,
    default_band_config,
    default_binary_config,
    comparison_metrics,
    common_returns,
    evaluate_grid,
    first_ratio_signal,
    generate_band_grid,
    generate_binary_grid,
    run_existing_benchmarks,
    walk_forward_search,
)
from vol_trading.signals import MAIN_PAIR, build_panel, build_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search Binary/Bands VIX3M/VIX rules and run annual walk-forward selection."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/grid_search"))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--min-train-years", type=int, default=DEFAULT_MIN_TRAIN_YEARS)
    return parser.parse_args()


def _write_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    frame.to_csv(path, index=index, lineterminator="\n", float_format="%.10g")


def _binary_heatmap(grid: pd.DataFrame, path: Path) -> None:
    high_weights = sorted(grid["high_weight"].unique())
    fig = make_subplots(
        rows=1,
        cols=len(high_weights),
        subplot_titles=[f"high weight = {weight:.0%}" for weight in high_weights],
        horizontal_spacing=0.08,
    )
    for col, high_weight in enumerate(high_weights, start=1):
        subset = grid[grid["high_weight"] == high_weight]
        table = (
            subset.pivot(index="low_weight", columns="threshold", values="sharpe")
            .sort_index()
            .sort_index(axis=1)
        )
        fig.add_trace(
            go.Heatmap(
                z=table.values,
                x=table.columns,
                y=[f"{value:.0%}" for value in table.index],
                coloraxis="coloraxis",
                text=table.round(3).values,
                texttemplate="%{text}",
                hovertemplate="threshold=%{x}<br>low weight=%{y}<br>Sharpe=%{z:.3f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="ratio threshold", row=1, col=col)
    fig.update_yaxes(title_text="SPY weight below threshold", row=1, col=1)
    fig.update_layout(
        height=450,
        title="Binary full-sample Sharpe surface (diagnostic only; not OOS evidence)",
        coloraxis=dict(colorscale="RdBu"),
    )
    fig.write_html(path, include_plotlyjs="cdn")


def _bands_heatmap(grid: pd.DataFrame, path: Path) -> None:
    fixed = grid[
        (grid["low_weight"] == 0.0)
        & (grid["mid_weight"] == 0.5)
        & (grid["high_weight"] == 1.0)
    ]
    table = (
        fixed.pivot(index="low_threshold", columns="high_threshold", values="sharpe")
        .sort_index()
        .sort_index(axis=1)
    )
    fig = go.Figure(
        go.Heatmap(
            z=table.values,
            x=table.columns,
            y=table.index,
            colorscale="RdBu",
            text=table.round(3).values,
            texttemplate="%{text}",
            hovertemplate="high=%{x}<br>low=%{y}<br>Sharpe=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=500,
        title="Bands full-sample Sharpe surface at 0% / 50% / 100% weights",
        xaxis_title="high ratio threshold",
        yaxis_title="low ratio threshold",
    )
    fig.write_html(path, include_plotlyjs="cdn")


def _growth_plot(returns: pd.DataFrame, path: Path) -> None:
    equity = (1.0 + returns).cumprod()
    fig = go.Figure()
    for column in equity:
        width = 2.6 if column.startswith("walk-forward") else 1.3
        fig.add_trace(
            go.Scatter(x=equity.index, y=equity[column], name=column, line=dict(width=width))
        )
    fig.update_yaxes(type="log")
    fig.update_layout(
        height=550,
        title="Walk-forward strategies vs fixed rules and benchmarks — growth of $1",
        legend=dict(orientation="h"),
    )
    fig.write_html(path, include_plotlyjs="cdn")


def _selection_plot(binary: pd.DataFrame, bands: pd.DataFrame, path: Path) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Binary selected parameters", "Bands selected thresholds"],
        vertical_spacing=0.16,
    )
    fig.add_trace(
        go.Scatter(
            x=binary["oos_year"],
            y=binary["threshold"],
            mode="lines+markers",
            name="binary threshold",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=binary["oos_year"],
            y=binary["low_weight"],
            mode="lines+markers",
            name="binary low weight",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=binary["oos_year"],
            y=binary["high_weight"],
            mode="lines+markers",
            name="binary high weight",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bands["oos_year"],
            y=bands["low_threshold"],
            mode="lines+markers",
            name="bands low threshold",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bands["oos_year"],
            y=bands["high_threshold"],
            mode="lines+markers",
            name="bands high threshold",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="out-of-sample year", row=2, col=1)
    fig.update_layout(height=700, title="Walk-forward parameter stability")
    fig.write_html(path, include_plotlyjs="cdn")


def main() -> int:
    args = parse_args()
    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)

    panel = build_panel(args.raw_dir)
    signals = build_signals(panel)
    first_signal = first_ratio_signal(signals, MAIN_PAIR, args.start)

    binary_configs = generate_binary_grid()
    band_configs = generate_band_grid()

    binary_grid, binary_results = evaluate_grid(
        panel, signals, binary_configs, first_signal, args.cost_bps, MAIN_PAIR
    )
    band_grid, band_results = evaluate_grid(
        panel, signals, band_configs, first_signal, args.cost_bps, MAIN_PAIR
    )

    wf_binary, binary_selections = walk_forward_search(
        panel,
        signals,
        binary_configs,
        binary_results,
        first_signal,
        args.cost_bps,
        MAIN_PAIR,
        args.min_train_years,
        name="walk-forward binary",
    )
    wf_bands, band_selections = walk_forward_search(
        panel,
        signals,
        band_configs,
        band_results,
        first_signal,
        args.cost_bps,
        MAIN_PAIR,
        args.min_train_years,
        name="walk-forward bands",
    )

    benchmarks = run_existing_benchmarks(panel, signals, first_signal, args.cost_bps)
    default_binary = binary_results[default_binary_config()]
    default_bands = band_results[default_band_config()]

    oos_results = {
        "walk-forward binary": wf_binary,
        "walk-forward bands": wf_bands,
        "fixed binary 1.00 (0/100)": default_binary,
        "fixed bands 0.95/1.05 (0/50/100)": default_bands,
        **benchmarks,
    }
    oos_metrics = comparison_metrics(oos_results, panel["cash_ret"])
    oos_returns = common_returns(oos_results)

    best_binary_config = next(
        config for config in binary_configs if config.name == binary_grid.iloc[0]["strategy"]
    )
    best_band_config = next(
        config for config in band_configs if config.name == band_grid.iloc[0]["strategy"]
    )
    full_results = {
        "best full-sample binary (in-sample)": binary_results[best_binary_config],
        "best full-sample bands (in-sample)": band_results[best_band_config],
        "fixed binary 1.00 (0/100)": default_binary,
        "fixed bands 0.95/1.05 (0/50/100)": default_bands,
        **benchmarks,
    }
    full_metrics = comparison_metrics(full_results, panel["cash_ret"])

    _write_csv(binary_grid, out / "binary_full_grid.csv")
    _write_csv(band_grid, out / "bands_full_grid.csv")
    _write_csv(binary_grid.head(20), out / "binary_top20.csv")
    _write_csv(band_grid.head(20), out / "bands_top20.csv")
    _write_csv(binary_selections, out / "binary_walk_forward_selections.csv")
    _write_csv(band_selections, out / "bands_walk_forward_selections.csv")
    _write_csv(oos_metrics, out / "walk_forward_comparison.csv", index=True)
    _write_csv(full_metrics, out / "full_sample_comparison.csv", index=True)
    _write_csv(oos_returns, out / "walk_forward_returns.csv", index=True)

    weights = pd.DataFrame(
        {
            "walk-forward binary": wf_binary.weights["SPY"],
            "walk-forward bands": wf_bands.weights["SPY"],
        }
    )
    _write_csv(weights, out / "walk_forward_spy_weights.csv", index=True)

    info = {
        "pair": MAIN_PAIR,
        "first_signal": first_signal.strftime("%Y-%m-%d"),
        "data_end": panel.index[-1].strftime("%Y-%m-%d"),
        "cost_bps": args.cost_bps,
        "min_train_years": args.min_train_years,
        "binary_candidates": len(binary_configs),
        "bands_candidates": len(band_configs),
        "selection_metric": "net-of-cost Sharpe",
        "important_note": (
            "Full-sample winners are diagnostic/in-sample. "
            "Use walk_forward_comparison.csv for honest strategy comparison."
        ),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")

    _binary_heatmap(binary_grid, out / "binary_sharpe_surface.html")
    _bands_heatmap(band_grid, out / "bands_sharpe_surface.html")
    _growth_plot(oos_returns, out / "walk_forward_growth.html")
    _selection_plot(
        binary_selections,
        band_selections,
        out / "walk_forward_parameter_stability.html",
    )

    columns = ["cagr", "ann_vol", "sharpe", "max_drawdown", "calmar", "turnover_per_year"]
    print(
        f"Grid search complete: {len(binary_configs)} Binary and "
        f"{len(band_configs)} Bands configurations."
    )
    print(
        f"First signal: {first_signal.date()} | data through {panel.index[-1].date()} | "
        f"costs: {args.cost_bps:g} bps"
    )
    print("\nWalk-forward / benchmark comparison (common OOS window):")
    print(oos_metrics[columns].round(3).to_string())
    print(f"\nResults written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
