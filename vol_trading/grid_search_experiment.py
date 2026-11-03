"""Standalone grid-search and walk-forward experiments for VIX3M/VIX.

This module intentionally does not modify or replace the existing backtest engine.
It reuses the repository's public helpers for scheduling, simulation, benchmarks,
and performance metrics, while keeping all parameter-search logic isolated here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .backtest import (
    DEFAULT_COST_BPS,
    BacktestResult,
    asset_returns,
    benchmark_targets,
    performance_metrics,
    schedule,
    simulate,
    spy_targets,
)
from .signals import MAIN_PAIR, month_end_dates

BINARY_THRESHOLDS = (0.900, 0.925, 0.950, 0.975, 1.000, 1.025, 1.050, 1.075, 1.100)
BINARY_LOW_WEIGHTS = (0.00, 0.25, 0.50)
BINARY_HIGH_WEIGHTS = (0.75, 1.00)

BAND_LOW_THRESHOLDS = (0.900, 0.925, 0.950, 0.975, 1.000)
BAND_HIGH_THRESHOLDS = (1.000, 1.025, 1.050, 1.075, 1.100)
BAND_LOW_WEIGHTS = (0.00, 0.25, 0.50)
BAND_MID_WEIGHTS = (0.25, 0.50, 0.75)
BAND_HIGH_WEIGHT = 1.00

DEFAULT_START = "2010-09-30"
DEFAULT_MIN_TRAIN_YEARS = 3
DEFAULT_SELECTION_METRIC = "sharpe"


@dataclass(frozen=True)
class BinaryConfig:
    threshold: float
    low_weight: float
    high_weight: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_weight <= 1.0:
            raise ValueError("low_weight must be between 0 and 1")
        if not 0.0 <= self.high_weight <= 1.0:
            raise ValueError("high_weight must be between 0 and 1")
        if self.low_weight >= self.high_weight:
            raise ValueError("binary strategy requires low_weight < high_weight")

    @property
    def name(self) -> str:
        return (
            f"binary threshold={self.threshold:.3f} "
            f"weights={self.low_weight:.2f}/{self.high_weight:.2f}"
        )


@dataclass(frozen=True)
class BandConfig:
    low_threshold: float
    high_threshold: float
    low_weight: float
    mid_weight: float
    high_weight: float = BAND_HIGH_WEIGHT

    def __post_init__(self) -> None:
        if self.low_threshold >= self.high_threshold:
            raise ValueError("band strategy requires low_threshold < high_threshold")
        for label, value in (
            ("low_weight", self.low_weight),
            ("mid_weight", self.mid_weight),
            ("high_weight", self.high_weight),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be between 0 and 1")
        if not self.low_weight <= self.mid_weight <= self.high_weight:
            raise ValueError("band weights must satisfy low <= mid <= high")

    @property
    def name(self) -> str:
        return (
            f"bands thresholds={self.low_threshold:.3f}/{self.high_threshold:.3f} "
            f"weights={self.low_weight:.2f}/{self.mid_weight:.2f}/{self.high_weight:.2f}"
        )


Config = BinaryConfig | BandConfig


def generate_binary_grid() -> list[BinaryConfig]:
    return [
        BinaryConfig(threshold, low_weight, high_weight)
        for threshold in BINARY_THRESHOLDS
        for low_weight in BINARY_LOW_WEIGHTS
        for high_weight in BINARY_HIGH_WEIGHTS
        if low_weight < high_weight
    ]


def generate_band_grid() -> list[BandConfig]:
    configs: list[BandConfig] = []
    for low_threshold in BAND_LOW_THRESHOLDS:
        for high_threshold in BAND_HIGH_THRESHOLDS:
            if low_threshold >= high_threshold:
                continue
            for low_weight in BAND_LOW_WEIGHTS:
                for mid_weight in BAND_MID_WEIGHTS:
                    if low_weight <= mid_weight <= BAND_HIGH_WEIGHT:
                        configs.append(
                            BandConfig(
                                low_threshold=low_threshold,
                                high_threshold=high_threshold,
                                low_weight=low_weight,
                                mid_weight=mid_weight,
                            )
                        )
    return configs


def binary_weights(ratio: pd.Series, config: BinaryConfig) -> pd.Series:
    weight = np.where(ratio > config.threshold, config.high_weight, config.low_weight)
    return pd.Series(weight, index=ratio.index, dtype=float).where(ratio.notna())


def band_weights(ratio: pd.Series, config: BandConfig) -> pd.Series:
    weight = np.select(
        [ratio > config.high_threshold, ratio >= config.low_threshold],
        [config.high_weight, config.mid_weight],
        default=config.low_weight,
    )
    return pd.Series(weight, index=ratio.index, dtype=float).where(ratio.notna())


def config_weights(ratio: pd.Series, config: Config) -> pd.Series:
    if isinstance(config, BinaryConfig):
        return binary_weights(ratio, config)
    return band_weights(ratio, config)


def first_ratio_signal(
    signals: pd.DataFrame,
    pair: str = MAIN_PAIR,
    start: str | pd.Timestamp = DEFAULT_START,
) -> pd.Timestamp:
    dates = month_end_dates(signals.index)
    dates = dates[dates >= pd.Timestamp(start)]
    ratio = signals.loc[dates, f"ratio:{pair}"]
    valid = ratio.notna()
    if not valid.any():
        raise ValueError(f"no usable {pair} ratio exists at or after {start}")
    return valid[valid].index[0]


def _signal_dates(
    signals: pd.DataFrame,
    first_signal: pd.Timestamp,
    pair: str,
) -> pd.DatetimeIndex:
    dates = month_end_dates(signals.index)
    dates = dates[dates >= first_signal]
    usable = signals.loc[dates, f"ratio:{pair}"].notna()
    return pd.DatetimeIndex(usable[usable].index)


def targets_for_config(
    signals: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    config: Config,
    pair: str = MAIN_PAIR,
) -> pd.DataFrame:
    ratio = signals.loc[signal_dates, f"ratio:{pair}"]
    return spy_targets(config_weights(ratio, config))


def run_config(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    config: Config,
    first_signal: pd.Timestamp,
    cost_bps: float = DEFAULT_COST_BPS,
    pair: str = MAIN_PAIR,
) -> BacktestResult:
    signal_dates = _signal_dates(signals, first_signal, pair)
    targets = targets_for_config(signals, signal_dates, config, pair)
    trades = schedule(targets, panel.index)
    return simulate(config.name, asset_returns(panel), trades, cost_bps)


def evaluate_grid(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    configs: Iterable[Config],
    first_signal: pd.Timestamp,
    cost_bps: float = DEFAULT_COST_BPS,
    pair: str = MAIN_PAIR,
) -> tuple[pd.DataFrame, dict[Config, BacktestResult]]:
    cash = panel["cash_ret"]
    rows: list[dict] = []
    results: dict[Config, BacktestResult] = {}
    for config in configs:
        result = run_config(panel, signals, config, first_signal, cost_bps, pair)
        results[config] = result
        metrics = performance_metrics(result.returns, cash, result.turnover, result.costs)
        row = {
            "strategy": config.name,
            "family": "binary" if isinstance(config, BinaryConfig) else "bands",
        }
        row.update(asdict(config))
        row.update(metrics)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["sharpe", "turnover_per_year"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
    return frame, results


def run_existing_benchmarks(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    first_signal: pd.Timestamp,
    cost_bps: float = DEFAULT_COST_BPS,
) -> dict[str, BacktestResult]:
    dates = month_end_dates(signals.index)
    dates = dates[dates >= first_signal]
    targets = benchmark_targets(signals, dates)
    returns = asset_returns(panel)
    return {
        name: simulate(name, returns, schedule(target, panel.index), cost_bps)
        for name, target in targets.items()
    }


def _slice_result_metrics(
    result: BacktestResult,
    cash: pd.Series,
    end: pd.Timestamp,
) -> dict:
    mask = result.returns.index <= end
    returns = result.returns.loc[mask]
    turnover = result.turnover.reindex(returns.index)
    costs = result.costs.reindex(returns.index)
    return performance_metrics(returns, cash, turnover, costs)


def select_training_winner(
    configs: Iterable[Config],
    results: dict[Config, BacktestResult],
    cash: pd.Series,
    train_end: pd.Timestamp,
    metric: str = DEFAULT_SELECTION_METRIC,
) -> tuple[Config, dict]:
    scored: list[tuple[float, float, str, Config, dict]] = []
    for config in configs:
        metrics = _slice_result_metrics(results[config], cash, train_end)
        score = float(metrics.get(metric, np.nan))
        turnover = float(metrics.get("turnover_per_year", np.inf))
        if np.isfinite(score):
            scored.append((score, -turnover, config.name, config, metrics))
    if not scored:
        raise ValueError(f"no candidate has a finite {metric} through {train_end.date()}")

    _, _, _, config, metrics = max(scored, key=lambda item: (item[0], item[1], item[2]))
    return config, metrics


def _trade_dates_for_signals(
    signal_dates: pd.DatetimeIndex,
    trading_days: pd.DatetimeIndex,
) -> pd.Series:
    positions = trading_days.get_indexer(signal_dates)
    if (positions < 0).any():
        raise ValueError("all signal dates must be trading days")
    next_positions = positions + 1
    valid = next_positions < len(trading_days)
    return pd.Series(
        trading_days[next_positions[valid]],
        index=signal_dates[valid],
        name="trade_date",
    )


def first_oos_year(first_signal: pd.Timestamp, min_train_years: int) -> int:
    return first_signal.year + min_train_years + (0 if first_signal.month == 1 else 1)


def walk_forward_search(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    configs: list[Config],
    candidate_results: dict[Config, BacktestResult],
    first_signal: pd.Timestamp,
    cost_bps: float = DEFAULT_COST_BPS,
    pair: str = MAIN_PAIR,
    min_train_years: int = DEFAULT_MIN_TRAIN_YEARS,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    name: str = "walk-forward",
) -> tuple[BacktestResult, pd.DataFrame]:
    """Re-select parameters each January using only information through prior year-end.

    December month-end signals are assigned to the following year when their next-session
    trade occurs in January. The selected target weights for every OOS year are stitched
    first, then passed through one call to simulate, so year-boundary turnover and costs
    are handled by the same engine as every other strategy.
    """
    cash = panel["cash_ret"]
    signal_dates = _signal_dates(signals, first_signal, pair)
    trade_dates = _trade_dates_for_signals(signal_dates, panel.index)
    start_year = first_oos_year(first_signal, min_train_years)

    targets: list[pd.DataFrame] = []
    selections: list[dict] = []

    for year in range(start_year, panel.index[-1].year + 1):
        prior = panel.index[panel.index <= pd.Timestamp(f"{year - 1}-12-31")]
        if prior.empty:
            continue
        train_end = prior[-1]
        chosen, train_metrics = select_training_winner(
            configs,
            candidate_results,
            cash,
            train_end,
            selection_metric,
        )

        year_signal_dates = trade_dates.index[trade_dates.dt.year == year]
        if len(year_signal_dates) == 0:
            continue

        targets.append(targets_for_config(signals, pd.DatetimeIndex(year_signal_dates), chosen, pair))
        row = {
            "oos_year": year,
            "train_end": train_end.strftime("%Y-%m-%d"),
            "selected": chosen.name,
            "train_selection_metric": float(train_metrics[selection_metric]),
            "train_cagr": float(train_metrics["cagr"]),
            "train_max_drawdown": float(train_metrics["max_drawdown"]),
            "train_turnover_per_year": float(train_metrics.get("turnover_per_year", np.nan)),
        }
        row.update(asdict(chosen))
        selections.append(row)

    if not targets:
        raise ValueError("walk-forward search produced no out-of-sample targets")

    stitched_targets = pd.concat(targets).sort_index()
    stitched_targets = stitched_targets[~stitched_targets.index.duplicated(keep="last")]
    trades = schedule(stitched_targets, panel.index)
    result = simulate(name, asset_returns(panel), trades, cost_bps)
    return result, pd.DataFrame(selections)


def comparison_metrics(
    results: dict[str, BacktestResult],
    cash: pd.Series,
) -> pd.DataFrame:
    """Compare all results on the exact common return window."""
    if not results:
        return pd.DataFrame()
    common_start = max(result.returns.index[0] for result in results.values())
    common_end = min(result.returns.index[-1] for result in results.values())
    rows: dict[str, dict] = {}
    for name, result in results.items():
        returns = result.returns.loc[common_start:common_end]
        turnover = result.turnover.reindex(returns.index)
        costs = result.costs.reindex(returns.index)
        rows[name] = performance_metrics(returns, cash, turnover, costs)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "strategy"
    return table


def common_returns(results: dict[str, BacktestResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    common_start = max(result.returns.index[0] for result in results.values())
    common_end = min(result.returns.index[-1] for result in results.values())
    return pd.DataFrame(
        {
            name: result.returns.loc[common_start:common_end]
            for name, result in results.items()
        }
    )


def default_binary_config() -> BinaryConfig:
    return BinaryConfig(threshold=1.0, low_weight=0.0, high_weight=1.0)


def default_band_config() -> BandConfig:
    return BandConfig(
        low_threshold=0.95,
        high_threshold=1.05,
        low_weight=0.0,
        mid_weight=0.5,
        high_weight=1.0,
    )
