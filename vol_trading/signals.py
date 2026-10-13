"""Daily research panel and VIX term-structure signals."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .public_data import cboe_path, fred_path, read_source, yahoo_path

# Tenor label -> (Cboe symbol, calendar days to the constant maturity).
TENORS = {
    "9D": ("VIX9D", 9),
    "1M": ("VIX", 30),
    "3M": ("VIX3M", 93),
    "6M": ("VIX6M", 182),
    "12M": ("VIX1Y", 365),
}
MAIN_PAIR = "3M/1M"
ZSCORE_WINDOW = 252
TRADING_DAYS = 252
MAX_FILL_DAYS = 1


def tenor_pairs() -> list[str]:
    """Every `long/short` label where the long tenor has the longer maturity."""
    ordered = sorted(TENORS, key=lambda label: TENORS[label][1])
    return [f"{long}/{short}" for short, long in combinations(ordered, 2)]


def split_pair(pair: str) -> tuple[str, str]:
    long, short = pair.split("/")
    return long, short


def build_panel(raw_dir: Path) -> pd.DataFrame:
    """One row per SPY trading day with returns, T-bill cash return, and volatility levels."""
    spy = read_source(yahoo_path(raw_dir, "SPY")).set_index("date")["adj_close"]
    ief = read_source(yahoo_path(raw_dir, "IEF")).set_index("date")["adj_close"]
    days = spy.index

    panel = pd.DataFrame(index=days)
    panel.index.name = "date"
    panel["spy"] = spy
    panel["spy_ret"] = spy.pct_change()
    panel["ief_ret"] = ief.reindex(days).ffill(limit=MAX_FILL_DAYS).pct_change(fill_method=None)

    tbill = read_source(fred_path(raw_dir, "DTB3")).set_index("date")["value"]
    tbill = tbill.reindex(days.union(tbill.index)).ffill(limit=MAX_FILL_DAYS).reindex(days)
    panel["tbill_yield"] = tbill
    # Cash earns yesterday's quoted yield, accrued per calendar day between sessions.
    gap_days = days.to_series().diff().dt.days.to_numpy()
    panel["cash_ret"] = tbill.shift(1).to_numpy() / 100.0 * gap_days / 360.0

    for label, (symbol, _) in TENORS.items():
        close = read_source(cboe_path(raw_dir, symbol)).set_index("date")["close"]
        panel[label] = close.reindex(days).ffill(limit=MAX_FILL_DAYS)

    panel["rv21"] = panel["spy_ret"].rolling(21, min_periods=21).std() * np.sqrt(TRADING_DAYS)
    return panel


def rolling_zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Z-score of each day against the trailing window ending that day; no future data."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def build_signals(panel: pd.DataFrame, window: int = ZSCORE_WINDOW) -> pd.DataFrame:
    """Ratio and rolling z-score for every tenor pair, plus the VIX level and realized vol."""
    columns: dict[str, pd.Series] = {}
    for pair in tenor_pairs():
        long, short = split_pair(pair)
        ratio = panel[long] / panel[short]
        columns[f"ratio:{pair}"] = ratio
        columns[f"z:{pair}"] = rolling_zscore(ratio, window)
    signals = pd.DataFrame(columns, index=panel.index)
    signals["vix"] = panel["1M"]
    signals["rv21"] = panel["rv21"]
    return signals


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each completed calendar month in `index`."""
    frame = pd.Series(index, index=index)
    ends = pd.DatetimeIndex(frame.groupby(index.to_period("M")).max().to_numpy())
    if len(ends) and ends[-1] < ends[-1] + pd.offsets.BMonthEnd(0):
        ends = ends[:-1]
    return ends
