"""Free daily history for the term-structure study: Cboe indexes, FRED T-bills, Yahoo ETFs."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
CBOE_SYMBOLS = ("VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y")
FRED_SERIES = ("DTB3",)
YAHOO_TICKERS = ("SPY", "IEF")
YAHOO_START = "2005-01-01"
REQUEST_TIMEOUT = 30


class PublicDataError(RuntimeError):
    """Raised when a public source returns no usable rows."""


@dataclass(frozen=True)
class SourceResult:
    name: str
    rows: int
    first: str
    last: str
    path: Path
    downloaded: bool


def cboe_path(raw_dir: Path, symbol: str) -> Path:
    return raw_dir / f"cboe_{symbol}.csv"


def fred_path(raw_dir: Path, series: str) -> Path:
    return raw_dir / f"fred_{series}.csv"


def yahoo_path(raw_dir: Path, ticker: str) -> Path:
    return raw_dir / f"yahoo_{ticker}.csv"


def _get_text(url: str, attempts: int = 3) -> str:
    # FRED drops connections for some custom User-Agent strings, so keep the requests default.
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except (requests.ConnectionError, requests.Timeout):
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def parse_cboe_csv(text: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [column.strip().lower() for column in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Cboe history contains a few zero or negative placeholder prints; they are not index levels.
    frame = frame[frame["close"] > 0]
    return _clean_dates(frame[["date", "open", "high", "low", "close"]])


def parse_fred_csv(text: str, series: str) -> pd.DataFrame:
    frame = pd.read_csv(io.StringIO(text), na_values=["."])
    date_column = "observation_date" if "observation_date" in frame else frame.columns[0]
    frame = frame.rename(columns={date_column: "date", series: "value"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return _clean_dates(frame.dropna(subset=["value"])[["date", "value"]])


def _clean_dates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(subset=["date"]).sort_values("date")
    return frame.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def fetch_cboe(symbol: str) -> pd.DataFrame:
    frame = parse_cboe_csv(_get_text(CBOE_URL.format(symbol=symbol)))
    if frame.empty:
        raise PublicDataError(f"Cboe returned no rows for {symbol}")
    return frame


def fetch_fred(series: str) -> pd.DataFrame:
    frame = parse_fred_csv(_get_text(FRED_URL.format(series=series)), series)
    if frame.empty:
        raise PublicDataError(f"FRED returned no rows for {series}")
    return frame


def fetch_yahoo(ticker: str, start: str = YAHOO_START) -> pd.DataFrame:
    """Daily close and dividend-adjusted close, which is the total-return price series."""
    import yfinance as yf

    history = yf.Ticker(ticker).history(start=start, auto_adjust=False, actions=False)
    if history.empty or "Adj Close" not in history:
        raise PublicDataError(f"Yahoo returned no adjusted closes for {ticker}")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(history.index.tz_localize(None).date),
            "close": history["Close"].to_numpy(),
            "adj_close": history["Adj Close"].to_numpy(),
        }
    )
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    frame = frame[(frame["adj_close"] > 0) & (frame["date"] < today)]
    return _clean_dates(frame)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    output.to_csv(path, index=False, lineterminator="\n")


def read_source(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def _result(name: str, frame: pd.DataFrame, path: Path, downloaded: bool) -> SourceResult:
    return SourceResult(
        name=name,
        rows=len(frame),
        first=frame["date"].min().strftime("%Y-%m-%d"),
        last=frame["date"].max().strftime("%Y-%m-%d"),
        path=path,
        downloaded=downloaded,
    )


def ingest_public(raw_dir: Path, refresh: bool = False) -> list[SourceResult]:
    """Download every free source, reusing saved CSVs unless `refresh` is set."""
    jobs = (
        [(f"cboe:{s}", cboe_path(raw_dir, s), lambda s=s: fetch_cboe(s)) for s in CBOE_SYMBOLS]
        + [(f"fred:{s}", fred_path(raw_dir, s), lambda s=s: fetch_fred(s)) for s in FRED_SERIES]
        + [(f"yahoo:{t}", yahoo_path(raw_dir, t), lambda t=t: fetch_yahoo(t)) for t in YAHOO_TICKERS]
    )
    results: list[SourceResult] = []
    for name, path, fetch in jobs:
        if path.exists() and not refresh:
            results.append(_result(name, read_source(path), path, downloaded=False))
            continue
        frame = fetch()
        _write(frame, path)
        results.append(_result(name, frame, path, downloaded=True))
    return results
