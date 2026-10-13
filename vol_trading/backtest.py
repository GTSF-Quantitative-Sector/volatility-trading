"""Monthly-rebalanced SPY allocation backtests, benchmarks, and performance metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .signals import MAIN_PAIR, TRADING_DAYS, month_end_dates, tenor_pairs

ASSETS = ("SPY", "IEF", "CASH")
RISKY_ASSETS = ("SPY", "IEF")
DEFAULT_COST_BPS = 5.0
COST_SENSITIVITY_BPS = (0.0, 5.0, 10.0)
TRADE_LAG_DAYS = 1
VIX_FILTER_LEVEL = 20.0
VOL_TARGET = 0.15
RATIO_BANDS = (0.95, 1.05)

# Peak-to-trough SPY windows for the named stress periods.
STRESS_WINDOWS = (
    ("Aug 2011 debt ceiling", "2011-07-07", "2011-10-03"),
    ("Aug 2015 China devaluation", "2015-08-17", "2015-08-25"),
    ("Feb 2018 Volmageddon", "2018-01-26", "2018-02-08"),
    ("Q4 2018 selloff", "2018-09-20", "2018-12-24"),
    ("Mar 2020 COVID crash", "2020-02-19", "2020-03-23"),
    ("2022 bear market", "2022-01-03", "2022-10-12"),
    ("Aug 2024 yen carry unwind", "2024-07-16", "2024-08-05"),
    ("Apr 2025 tariff shock", "2025-02-19", "2025-04-08"),
)


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series


def asset_returns(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"SPY": panel["spy_ret"], "IEF": panel["ief_ret"], "CASH": panel["cash_ret"]},
        index=panel.index,
    )


def spy_targets(spy_weight: pd.Series) -> pd.DataFrame:
    """Turn a SPY weight per signal date into SPY/IEF/CASH targets, cash taking the rest."""
    weight = spy_weight.astype(float)
    return pd.DataFrame({"SPY": weight, "IEF": 0.0, "CASH": 1.0 - weight}, index=weight.index)


def schedule(targets: pd.DataFrame, trading_days: pd.DatetimeIndex, lag: int = TRADE_LAG_DAYS) -> pd.DataFrame:
    """Move each signal-date target to the close `lag` sessions later, when it is traded."""
    positions = trading_days.get_indexer(targets.index)
    if (positions < 0).any():
        raise ValueError("signal dates must be trading days")
    trade_positions = positions + lag
    keep = trade_positions < len(trading_days)
    scheduled = targets.iloc[keep].copy()
    scheduled.index = trading_days[trade_positions[keep]]
    return scheduled


def simulate(
    name: str,
    returns: pd.DataFrame,
    trades: pd.DataFrame,
    cost_bps: float = DEFAULT_COST_BPS,
) -> BacktestResult:
    """Daily close-to-close simulation with weight drift between rebalances.

    On a trade date the day's return accrues to the old weights, then the book is
    rebalanced at that close and pays `cost_bps` on the SPY and IEF notional traded.
    Rows of `trades` containing NaN are skipped, which leaves the position unchanged.
    """
    trades = trades.reindex(columns=list(ASSETS)).dropna(how="any")
    if trades.empty:
        raise ValueError(f"{name} has no tradable targets")
    days = returns.loc[trades.index[0]:].index
    asset_rets = returns.loc[days, list(ASSETS)].fillna(0.0).to_numpy()
    trade_lookup = {day: row.to_numpy(dtype=float) for day, row in trades.iterrows()}
    risky = np.array([asset in RISKY_ASSETS for asset in ASSETS])

    weights = np.array([0.0, 0.0, 1.0])
    daily_returns = np.zeros(len(days))
    end_weights = np.zeros((len(days), len(ASSETS)))
    turnover = np.zeros(len(days))
    costs = np.zeros(len(days))
    for i, day in enumerate(days):
        growth = 1.0
        if i > 0:
            gross = weights * (1.0 + asset_rets[i])
            growth = gross.sum()
            weights = gross / growth
        target = trade_lookup.get(day)
        if target is not None:
            traded = np.abs(target - weights)[risky].sum()
            cost = traded * cost_bps / 10_000.0
            turnover[i] = traded
            costs[i] = cost
            growth *= 1.0 - cost
            weights = target
        daily_returns[i] = growth - 1.0
        end_weights[i] = weights

    return BacktestResult(
        name=name,
        returns=pd.Series(daily_returns, index=days, name=name),
        weights=pd.DataFrame(end_weights, index=days, columns=list(ASSETS)),
        turnover=pd.Series(turnover, index=days, name=name),
        costs=pd.Series(costs, index=days, name=name),
    )


def binary_rule(ratio: pd.Series) -> pd.Series:
    return (ratio > 1.0).astype(float).where(ratio.notna())


def zscore_rule(z: pd.Series) -> pd.Series:
    weight = np.select([z >= 0.0, z >= -1.0], [1.0, 0.5], default=0.0)
    return pd.Series(weight, index=z.index).where(z.notna())


def band_rule(ratio: pd.Series, bands: tuple[float, float] = RATIO_BANDS) -> pd.Series:
    low, high = bands
    weight = np.select([ratio > high, ratio >= low], [1.0, 0.5], default=0.0)
    return pd.Series(weight, index=ratio.index).where(ratio.notna())


def vix_filter_rule(vix: pd.Series, level: float = VIX_FILTER_LEVEL) -> pd.Series:
    return (vix < level).astype(float).where(vix.notna())


def vol_target_rule(rv21: pd.Series, target: float = VOL_TARGET) -> pd.Series:
    return (target / rv21).clip(upper=1.0).where(rv21 > 0)


RULES: dict[str, Callable[[pd.Series], pd.Series]] = {
    "binary": binary_rule,
    "zscore": zscore_rule,
    "bands": band_rule,
}
RULE_COLUMN = {"binary": "ratio", "zscore": "z", "bands": "ratio"}


def strategy_name(rule: str, pair: str) -> str:
    return f"{rule} {pair}"


def benchmark_targets(signals: pd.DataFrame, signal_dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    first = signal_dates[:1]
    sixty_forty = pd.DataFrame({"SPY": 0.6, "IEF": 0.4, "CASH": 0.0}, index=signal_dates)
    return {
        "SPY buy-and-hold": pd.DataFrame({"SPY": 1.0, "IEF": 0.0, "CASH": 0.0}, index=first),
        "60/40 SPY/IEF": sixty_forty,
        f"VIX < {VIX_FILTER_LEVEL:g} filter": spy_targets(vix_filter_rule(signals.loc[signal_dates, "vix"])),
        f"Vol target {VOL_TARGET:.0%}": spy_targets(vol_target_rule(signals.loc[signal_dates, "rv21"])),
    }


def rule_targets(signals: pd.DataFrame, signal_dates: pd.DatetimeIndex, rule: str, pair: str) -> pd.DataFrame:
    column = f"{RULE_COLUMN[rule]}:{pair}"
    return spy_targets(RULES[rule](signals.loc[signal_dates, column]))


def first_complete_signal(signals: pd.DataFrame, columns: list[str], start: str | None = None) -> pd.Timestamp:
    dates = month_end_dates(signals.index)
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    complete = signals.loc[dates, columns].notna().all(axis=1)
    if not complete.any():
        raise ValueError("no month-end date has every required signal")
    return complete[complete].index[0]


def run_suite(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    strategies: list[tuple[str, str]],
    first_signal: pd.Timestamp,
    cost_bps: float = DEFAULT_COST_BPS,
    include_benchmarks: bool = True,
) -> dict[str, BacktestResult]:
    """Run `(rule, pair)` strategies and the benchmarks from the same first signal date."""
    signal_dates = month_end_dates(signals.index)
    signal_dates = signal_dates[signal_dates >= first_signal]
    returns = asset_returns(panel)
    targets: dict[str, pd.DataFrame] = {
        strategy_name(rule, pair): rule_targets(signals, signal_dates, rule, pair) for rule, pair in strategies
    }
    if include_benchmarks:
        targets.update(benchmark_targets(signals, signal_dates))
    return {
        name: simulate(name, returns, schedule(target, panel.index), cost_bps)
        for name, target in targets.items()
    }


def headline_strategies() -> list[tuple[str, str]]:
    return [(rule, MAIN_PAIR) for rule in RULES]


def pair_grid_strategies() -> list[tuple[str, str]]:
    return [(rule, pair) for rule in RULES for pair in tenor_pairs()]


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def performance_metrics(
    returns: pd.Series,
    cash: pd.Series,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
) -> dict[str, float]:
    returns = returns.dropna()
    excess = returns - cash.reindex(returns.index).fillna(0.0)
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    total = float((1.0 + returns).prod())
    cagr = total ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = float(returns.std() * np.sqrt(TRADING_DAYS))
    excess_std = float(excess.std())
    downside = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2)))
    mdd = max_drawdown(returns)
    metrics = {
        "start": returns.index[0].strftime("%Y-%m-%d"),
        "end": returns.index[-1].strftime("%Y-%m-%d"),
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": float(excess.mean() / excess_std * np.sqrt(TRADING_DAYS)) if excess_std > 0 else np.nan,
        "sortino": float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "total_return": total - 1.0,
    }
    if turnover is not None:
        metrics["turnover_per_year"] = float(turnover.loc[returns.index].sum() / years) if years > 0 else np.nan
    if costs is not None:
        metrics["total_cost"] = float(costs.loc[returns.index].sum())
    return metrics


def metrics_table(results: dict[str, BacktestResult], cash: pd.Series) -> pd.DataFrame:
    rows = {
        name: performance_metrics(result.returns, cash, result.turnover, result.costs)
        for name, result in results.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "strategy"
    return table


def stress_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Compounded return from the close on `start` to the close on `end` for each strategy."""
    rows = []
    for label, start, end in STRESS_WINDOWS:
        row = {"window": label, "start": start, "end": end}
        for name, result in results.items():
            index = result.returns.index
            window = result.returns[(index > pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
            covered = len(window) > 0 and index[0] <= pd.Timestamp(start)
            row[name] = float((1.0 + window).prod() - 1.0) if covered else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("window")


def returns_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame({name: result.returns for name, result in results.items()})


def spy_weight_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame({name: result.weights["SPY"] for name, result in results.items()})
