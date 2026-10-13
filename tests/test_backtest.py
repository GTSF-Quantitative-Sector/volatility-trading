import numpy as np
import pandas as pd
import pytest

from vol_trading.backtest import performance_metrics, run_suite, simulate, max_drawdown
from vol_trading.public_data import parse_cboe_csv, parse_fred_csv
from vol_trading.signals import month_end_dates, rolling_zscore
from vol_trading.stats import forward_targets


def _panel(days: pd.DatetimeIndex, spy_ret: float = 0.01) -> pd.DataFrame:
    return pd.DataFrame(
        {"spy_ret": spy_ret, "ief_ret": 0.0, "cash_ret": 0.0, "spy": 100.0},
        index=days,
    )


def test_month_end_signal_trades_at_next_close_and_earns_from_the_day_after():
    days = pd.bdate_range("2024-01-02", "2024-03-29")
    ratio = pd.Series(1.2, index=days)
    ratio.loc["2024-01-31"] = 0.9
    signals = pd.DataFrame({"ratio:3M/1M": ratio})

    result = run_suite(
        _panel(days), signals, [("binary", "3M/1M")], pd.Timestamp("2024-01-31"),
        cost_bps=5.0, include_benchmarks=False,
    )["binary 3M/1M"]

    assert result.returns.index[0] == pd.Timestamp("2024-02-01")
    assert (result.returns.loc["2024-02-01":"2024-02-29"] == 0.0).all()
    # The Feb 29 signal trades at the Mar 1 close: that day pays cost but earns nothing.
    assert result.returns.loc["2024-03-01"] == pytest.approx(-0.0005)
    assert result.turnover.loc["2024-03-01"] == pytest.approx(1.0)
    assert result.returns.loc["2024-03-04"] == pytest.approx(0.01)


def test_zscore_ignores_future_values():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(size=120))
    changed = series.copy()
    changed.iloc[60:] += 50.0
    original = rolling_zscore(series, window=20)
    assert original.iloc[:19].isna().all()
    pd.testing.assert_series_equal(original.iloc[:60], rolling_zscore(changed, window=20).iloc[:60])


def test_month_end_dates_drop_the_unfinished_month():
    days = pd.bdate_range("2024-01-02", "2024-03-15")
    assert list(month_end_dates(days)) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]


def test_rebalance_charges_cost_on_drifted_weights():
    days = pd.bdate_range("2024-01-02", periods=3)
    returns = pd.DataFrame({"SPY": [0.0, 0.10, 0.0], "IEF": 0.0, "CASH": 0.0}, index=days)
    trades = pd.DataFrame({"SPY": 0.6, "IEF": 0.4, "CASH": 0.0}, index=days[[0, 1]])

    result = simulate("60/40", returns, trades, cost_bps=10.0)

    drifted_spy = 0.6 * 1.1 / 1.06
    traded = abs(0.6 - drifted_spy) + abs(0.4 - (1.0 - drifted_spy))
    assert result.turnover.iloc[1] == pytest.approx(traded)
    assert result.returns.iloc[1] == pytest.approx(1.06 * (1 - traded * 0.001) - 1.0)
    assert result.weights.iloc[1]["SPY"] == pytest.approx(0.6)


def test_metrics_match_hand_calculation():
    days = pd.DatetimeIndex(["2020-01-02", "2020-07-01", "2021-01-04"])
    returns = pd.Series([0.1, -0.5, 0.2], index=days)
    cash = pd.Series(0.0, index=days)

    metrics = performance_metrics(returns, cash)

    years = (days[-1] - days[0]).days / 365.25
    assert metrics["max_drawdown"] == pytest.approx(-0.5)
    assert max_drawdown(returns) == pytest.approx(-0.5)
    assert metrics["cagr"] == pytest.approx((1.1 * 0.5 * 1.2) ** (1 / years) - 1.0)
    assert metrics["sharpe"] == pytest.approx(returns.mean() / returns.std() * np.sqrt(252))


def test_forward_targets_use_only_later_days():
    days = pd.bdate_range("2024-01-02", periods=5)
    panel = pd.DataFrame({"spy_ret": [0.01, 0.02, -0.01, 0.03, 0.0], "cash_ret": 0.0}, index=days)

    targets = forward_targets(panel, horizons=(2,))

    assert targets["excess_return_2"].iloc[0] == pytest.approx(np.log1p(0.02) + np.log1p(-0.01))
    assert targets["excess_return_2"].iloc[-2:].isna().all()


def test_public_parsers_drop_placeholders():
    cboe = parse_cboe_csv("DATE,OPEN,HIGH,LOW,CLOSE\n01/03/2024,0,0,0,0\n01/02/2024,13,14,12,13.5\n")
    assert list(cboe["close"]) == [13.5]
    fred = parse_fred_csv("observation_date,DTB3\n2024-01-01,.\n2024-01-02,5.2\n", "DTB3")
    assert list(fred["value"]) == [5.2]
