"""Predictive regressions and robustness checks for the term-structure signal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .backtest import STRESS_WINDOWS, BacktestResult, performance_metrics
from .signals import MAIN_PAIR, TRADING_DAYS, month_end_dates, tenor_pairs

HORIZONS = (21, 63)
SIGNAL_KINDS = ("ratio", "z")
TARGETS = ("excess_return", "realized_vol")
SUBPERIODS = (
    ("2010-2014", "2010-01-01", "2014-12-31"),
    ("2015-2019", "2015-01-01", "2019-12-31"),
    ("2020-2026", "2020-01-01", "2026-12-31"),
)
REGRESSION_START = "2010-01-01"
OOS_MIN_TRAIN_YEARS = 3
WALK_FORWARD_MIN_YEARS = 2
LEAD_LOOKBACK_DAYS = 63


def forward_targets(panel: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Targets measured strictly after day t: days t+1 through t+h."""
    log_excess = np.log1p(panel["spy_ret"]) - np.log1p(panel["cash_ret"].fillna(0.0))
    columns = {}
    for h in horizons:
        columns[f"excess_return_{h}"] = log_excess.rolling(h, min_periods=h).sum().shift(-h)
        columns[f"realized_vol_{h}"] = (
            panel["spy_ret"].rolling(h, min_periods=h).std().shift(-h) * np.sqrt(TRADING_DAYS)
        )
    return pd.DataFrame(columns, index=panel.index)


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame - frame.mean()) / frame.std()


def _hac_fit(y: pd.Series, x: pd.DataFrame, lags: int):
    return sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def predictive_regressions(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    pairs: list[str] | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    start: str = REGRESSION_START,
) -> pd.DataFrame:
    """Daily overlapping regressions with Newey-West errors; regressors are standardized,
    so each coefficient is the effect of a one-standard-deviation move in the signal."""
    targets = forward_targets(panel, horizons)
    pairs = pairs or tenor_pairs()
    rows = []
    for target in TARGETS:
        for h in horizons:
            y_all = targets[f"{target}_{h}"]
            specs: list[tuple[str, str, str, list[str]]] = [("VIX level", "vix", "VIX only", ["vix"])]
            for pair in pairs:
                for kind in SIGNAL_KINDS:
                    column = f"{kind}:{pair}"
                    specs += [
                        (pair, kind, "signal", [column]),
                        (pair, kind, "signal + VIX", [column, "vix"]),
                        (pair, kind, "signal + VIX + RV21", [column, "vix", "rv21"]),
                    ]
            for pair, kind, spec, columns in specs:
                data = pd.concat([y_all.rename("y"), signals[columns]], axis=1).loc[start:].dropna()
                if len(data) < 252:
                    continue
                x = _standardize(data[columns])
                fit = _hac_fit(data["y"], x, lags=h)
                name = columns[0]
                ci = fit.conf_int().loc[name]
                rows.append(
                    {
                        "target": target,
                        "horizon": h,
                        "pair": pair,
                        "signal": kind,
                        "spec": spec,
                        "coef": fit.params[name],
                        "se": fit.bse[name],
                        "t": fit.tvalues[name],
                        "p": fit.pvalues[name],
                        "ci_low": ci.iloc[0],
                        "ci_high": ci.iloc[1],
                        "vix_coef": fit.params.get("vix", np.nan) if name != "vix" else np.nan,
                        "r2": fit.rsquared,
                        "nobs": int(fit.nobs),
                        "start": data.index[0].strftime("%Y-%m-%d"),
                        "end": data.index[-1].strftime("%Y-%m-%d"),
                    }
                )
    return pd.DataFrame(rows)


def oos_r2(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    pairs: list[str] | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    start: str = REGRESSION_START,
    min_train_years: int = OOS_MIN_TRAIN_YEARS,
) -> pd.DataFrame:
    """Expanding-window out-of-sample R-squared against the historical mean.

    A forecast made at month end t trains only on days s whose forward window
    closed by t (s + h <= t), so no target overlapping the forecast date is used.
    """
    targets = forward_targets(panel, horizons)
    pairs = pairs or tenor_pairs()
    candidates = [("VIX level", "vix", "vix")] + [
        (pair, kind, f"{kind}:{pair}") for pair in pairs for kind in SIGNAL_KINDS
    ]
    position = pd.Series(np.arange(len(panel.index)), index=panel.index)
    rows = []
    for target in TARGETS:
        for h in horizons:
            for pair, kind, column in candidates:
                data = pd.concat(
                    [targets[f"{target}_{h}"].rename("y"), signals[column].rename("x")], axis=1
                ).loc[start:].dropna()
                if data.empty:
                    continue
                pos = position.loc[data.index].to_numpy()
                y = data["y"].to_numpy()
                x = data["x"].to_numpy()
                first_forecast = data.index[0] + pd.DateOffset(years=min_train_years)
                forecast_dates = [d for d in month_end_dates(panel.index) if d >= first_forecast and d in data.index]
                errors_model, errors_mean = [], []
                for date in forecast_dates:
                    cutoff = position[date] - h
                    train = pos <= cutoff
                    if train.sum() < 252:
                        continue
                    design = np.column_stack([np.ones(train.sum()), x[train]])
                    beta, *_ = np.linalg.lstsq(design, y[train], rcond=None)
                    i = data.index.get_loc(date)
                    errors_model.append(y[i] - (beta[0] + beta[1] * x[i]))
                    errors_mean.append(y[i] - y[train].mean())
                if not errors_model:
                    continue
                sse_model = float(np.sum(np.square(errors_model)))
                sse_mean = float(np.sum(np.square(errors_mean)))
                rows.append(
                    {
                        "target": target,
                        "horizon": h,
                        "pair": pair,
                        "signal": kind,
                        "oos_r2": 1.0 - sse_model / sse_mean,
                        "forecasts": len(errors_model),
                        "first_forecast": forecast_dates[0].strftime("%Y-%m-%d"),
                    }
                )
    return pd.DataFrame(rows)


def _trading_gap(index: pd.DatetimeIndex, a: pd.Timestamp, b: pd.Timestamp) -> int:
    return int(index.get_loc(b) - index.get_loc(a))


def _exit_trade(weight: pd.Series, search_start: pd.Timestamp, trough: pd.Timestamp) -> pd.Timestamp:
    """Trade date that began the first below-full-exposure spell overlapping the window."""
    held = weight.loc[search_start:trough]
    out = held[held < 1.0]
    if out.empty:
        return pd.NaT
    fully_in = weight.loc[: out.index[0]]
    fully_in = fully_in[fully_in >= 1.0]
    if fully_in.empty:
        return weight.index[0]
    return weight.index[weight.index.get_loc(fully_in.index[-1]) + 1]


def lead_lag_table(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    strategy: BacktestResult,
    pair: str = MAIN_PAIR,
    lookback: int = LEAD_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """When the daily ratio inverted, when the monthly strategy exited and re-entered,
    measured against each stress window's SPY peak and trough.

    Negative `reentry_days_after_trough` means the strategy was fully invested again
    before the trough and rode the rest of the decline.
    """
    ratio = signals[f"ratio:{pair}"]
    spy = panel["spy"]
    weight = strategy.weights["SPY"]
    index = panel.index
    rows = []
    for label, start, end in STRESS_WINDOWS:
        peak, trough = pd.Timestamp(start), pd.Timestamp(end)
        if peak < weight.index[0]:
            continue
        search_start = index[max(index.get_loc(peak) - lookback, 0)]
        before_trough = ratio.loc[search_start:trough]
        inverted = before_trough[before_trough < 1.0]
        first_inversion = inverted.index[0] if len(inverted) else pd.NaT

        exit_date = _exit_trade(weight, search_start, trough)

        after_trough = ratio.loc[trough:]
        recovered = after_trough[after_trough > 1.0]
        reentry_signal = recovered.index[0] if len(recovered) else pd.NaT

        reentry = pd.NaT
        if pd.notna(exit_date):
            back_in = weight.loc[exit_date:]
            back_in = back_in[back_in >= 1.0]
            reentry = back_in.index[0] if len(back_in) else pd.NaT

        rows.append(
            {
                "window": label,
                "peak": peak.date(),
                "trough": trough.date(),
                "spy_drawdown": spy[trough] / spy[peak] - 1.0,
                "first_inversion": first_inversion.date() if pd.notna(first_inversion) else None,
                "inversion_days_before_peak": _trading_gap(index, first_inversion, peak) if pd.notna(first_inversion) else np.nan,
                "spy_move_peak_to_inversion": spy[first_inversion] / spy[peak] - 1.0 if pd.notna(first_inversion) else np.nan,
                "strategy_exit": exit_date.date() if pd.notna(exit_date) else None,
                "spy_move_peak_to_exit": spy[exit_date] / spy[peak] - 1.0 if pd.notna(exit_date) else np.nan,
                "reentry_signal": reentry_signal.date() if pd.notna(reentry_signal) else None,
                "strategy_reentry": reentry.date() if pd.notna(reentry) else None,
                "reentry_days_after_trough": _trading_gap(index, trough, reentry) if pd.notna(reentry) else np.nan,
                "spy_gain_missed_trough_to_reentry": (
                    spy[reentry] / spy[trough] - 1.0 if pd.notna(reentry) and reentry > trough else np.nan
                ),
                "spy_loss_ridden_reentry_to_trough": (
                    spy[trough] / spy[reentry] - 1.0 if pd.notna(reentry) and reentry < trough else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def subperiod_table(results: dict[str, BacktestResult], cash: pd.Series) -> pd.DataFrame:
    rows = []
    for label, start, end in SUBPERIODS:
        for name, result in results.items():
            window = result.returns.loc[start:end]
            if len(window) < 21:
                continue
            metrics = performance_metrics(window, cash, result.turnover, result.costs)
            rows.append({"period": label, "strategy": name, **metrics})
    return pd.DataFrame(rows)


def _sharpe(returns: pd.Series, cash: pd.Series) -> float:
    excess = returns - cash.reindex(returns.index).fillna(0.0)
    std = excess.std()
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else np.nan


def walk_forward(
    candidates: dict[str, BacktestResult],
    cash: pd.Series,
    cost_bps: float,
    min_years: int = WALK_FORWARD_MIN_YEARS,
) -> tuple[pd.Series, pd.DataFrame]:
    """Each January, hold the candidate with the best Sharpe on all earlier data.

    Switching candidates charges `cost_bps` on the change in SPY weight at the switch.
    """
    returns = pd.DataFrame({name: result.returns for name, result in candidates.items()})
    weights = pd.DataFrame({name: result.weights["SPY"] for name, result in candidates.items()})
    first_year = returns.index[0].year + min_years + (0 if returns.index[0].month == 1 else 1)
    pieces, selections = [], []
    previous: str | None = None
    for year in range(first_year, returns.index[-1].year + 1):
        history = returns.loc[: f"{year - 1}-12-31"]
        scores = {name: _sharpe(history[name].dropna(), cash) for name in returns}
        chosen = max(scores, key=lambda name: scores[name])
        year_returns = returns.loc[f"{year}-01-01":f"{year}-12-31", chosen].copy()
        if year_returns.empty:
            break
        if previous is not None and previous != chosen:
            last_day = history.index[-1]
            switch = abs(weights.at[last_day, chosen] - weights.at[last_day, previous])
            year_returns.iloc[0] = (1.0 + year_returns.iloc[0]) * (1.0 - switch * cost_bps / 10_000.0) - 1.0
        pieces.append(year_returns)
        selections.append({"year": year, "selected": chosen, "trailing_sharpe": scores[chosen]})
        previous = chosen
    series = pd.concat(pieces).rename("walk-forward binary") if pieces else pd.Series(dtype=float)
    return series, pd.DataFrame(selections)
