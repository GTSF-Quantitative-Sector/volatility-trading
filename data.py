"""Massive REST ingestion for the core daily volatility-trading series.

The client follows the notebook's direct HTTP pattern: header authentication,
credential-safe caching, pagination, and normalized aggregate bars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import requests_cache
from dotenv import load_dotenv

INDEX_HISTORY_START = date(2023, 2, 14)
CORE_TICKERS = ("SPY", "I:VIX", "I:VIX3M")
EASTERN = ZoneInfo("America/New_York")
REQUEST_TIMEOUT = 25
MAX_PAGES = 100
BAR_COLUMNS = {
    "t": "timestamp",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vw": "vwap",
    "n": "transactions",
    "T": "ticker",
}
OUTPUT_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
    "ticker",
]
_SECRET_QUERY_KEYS = {"apikey", "api_key"}


class MassiveConfigError(RuntimeError):
    """Raised when local API configuration is missing or invalid."""


class MassiveAPIError(RuntimeError):
    """An API error that intentionally omits credentials and request URLs."""


@dataclass(frozen=True)
class IngestResult:
    ticker: str
    rows: int
    path: Path


def project_root() -> Path:
    return Path(__file__).resolve().parent


def load_project_env() -> None:
    """Load the repository `.env` without overriding variables already set."""
    load_dotenv(project_root() / ".env", override=False)
    load_dotenv(override=False)


def load_api_key() -> str:
    load_project_env()
    api_key = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise MassiveConfigError(
            "MASSIVE_API_KEY is missing. Copy .env.example to .env and put the key there."
        )
    return api_key


def market_today() -> date:
    return datetime.now(EASTERN).date()


def output_path_for_ticker(output_dir: Path, ticker: str) -> Path:
    return output_dir / f"{ticker.replace(':', '_')}.csv"


def _cache_ttl(path: str) -> int:
    """Choose freshness by data type, matching the notebook's cache policy."""
    if "snapshot" in path or "marketstatus/now" in path:
        return 30
    if any(part in path for part in ("financials", "reference", "dividends", "splits", "holidays")):
        return 43_200
    return 300


def _strip_secret_params(url: str) -> str:
    parsed = urlparse(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _SECRET_QUERY_KEYS
    ]
    return urlunparse(parsed._replace(query=urlencode(query)))


def _redact(detail: str, api_key: str) -> str:
    if api_key and api_key in detail:
        return detail.replace(api_key, "[REDACTED]")
    return detail


def _log_target(path: str) -> str:
    if path.startswith("http"):
        return urlparse(path).path or path
    return path


def _build_session(cache_dir: Path, api_key: str) -> requests_cache.CachedSession:
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests_cache.CachedSession(
        cache_name=str(cache_dir / "massive_http_cache"),
        backend="sqlite",
        expire_after=timedelta(seconds=300),
        allowable_methods=("GET",),
        allowable_codes=(200,),
        ignored_parameters=["apiKey", "api_key", "apikey"],
    )
    session.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "volatility-trading/0.1",
        }
    )
    return session


class MassiveClient:
    """GET client for Massive aggregate and reference endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        cache_dir: Path | None = None,
        cache_mode: str | None = None,
        timeout: int = REQUEST_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        load_project_env()
        self.api_key = api_key or load_api_key()
        self.base_url = (
            base_url or os.getenv("MASSIVE_BASE_URL") or "https://api.massive.com"
        ).rstrip("/")
        self.cache_mode = _validate_cache_mode(
            cache_mode or os.getenv("MASSIVE_CACHE_MODE") or "auto"
        )
        self.timeout = timeout
        self.session = session or _build_session(
            cache_dir or Path(os.getenv("MASSIVE_CACHE_DIR", ".massive_cache")),
            self.api_key,
        )

    def get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        cache_mode: str | None = None,
    ) -> dict | list:
        mode = _validate_cache_mode(cache_mode or self.cache_mode)
        clean_params = {
            key: value
            for key, value in (params or {}).items()
            if value is not None and key.lower() not in _SECRET_QUERY_KEYS
        }
        url = path if path.startswith("http") else f"{self.base_url}/{path.lstrip('/')}"
        url = _strip_secret_params(url)
        request_kwargs: dict[str, Any] = {"params": clean_params, "timeout": self.timeout}
        if isinstance(self.session, requests_cache.CachedSession):
            request_kwargs.update(
                {
                    "expire_after": _cache_ttl(path),
                    "force_refresh": mode == "live",
                    "only_if_cached": mode == "cache",
                }
            )
        response = self.session.get(url, **request_kwargs)

        if response.status_code == 504 and mode == "cache":
            raise MassiveAPIError(f"No cached response exists for {_log_target(path)}")
        if response.status_code != 200:
            detail = response.reason or "request failed"
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                if isinstance(body, dict):
                    detail = str(body.get("error") or body.get("message") or detail)
            raise MassiveAPIError(
                f"{_log_target(path)} returned HTTP {response.status_code}: "
                f"{_redact(detail, self.api_key)}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise MassiveAPIError(f"{_log_target(path)} returned a non-JSON response") from exc
        if isinstance(payload, dict) and str(payload.get("status", "")).upper() == "ERROR":
            detail = str(payload.get("error") or payload.get("message") or "request failed")
            raise MassiveAPIError(
                f"{_log_target(path)} returned an API error: {_redact(detail, self.api_key)}"
            )
        return payload

    def paginate(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        max_pages: int | None = None,
        cache_mode: str | None = None,
    ) -> list[dict]:
        """Follow Massive `next_url` pagination and return one list of records."""
        records: list[dict] = []
        page = 0
        next_path: str | None = path
        next_params = params
        seen: set[str] = set()
        page_limit = MAX_PAGES if max_pages is None else max_pages
        while next_path and page < page_limit:
            request_path = _strip_secret_params(next_path)
            if request_path in seen:
                break
            seen.add(request_path)
            payload = self.get(request_path, next_params, cache_mode)
            page_results = payload.get("results", []) if isinstance(payload, dict) else payload
            if isinstance(page_results, list):
                records.extend(item for item in page_results if isinstance(item, dict))
            elif isinstance(page_results, dict):
                records.append(page_results)
            next_path = payload.get("next_url") if isinstance(payload, dict) else None
            next_params = None
            page += 1
        if next_path and max_pages is None:
            raise MassiveAPIError(f"Pagination stopped after {page_limit} pages for {_log_target(path)}")
        return records

    def get_aggregate_bars(
        self,
        ticker: str,
        start: date | str,
        end: date | str,
        multiplier: int = 1,
        timespan: str = "day",
        adjusted: bool = True,
        limit: int = 50_000,
    ) -> pd.DataFrame:
        symbol = ticker.upper()
        start_text = _date_text(start)
        end_text = _date_text(end)
        if start_text > end_text:
            raise ValueError("start date must be on or before end date")
        path = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_text}/{end_text}"
        params: dict[str, Any] = {"sort": "asc", "limit": limit}
        if not symbol.startswith("I:"):
            params["adjusted"] = str(adjusted).lower()
        rows = self.paginate(path, params)
        return normalize_aggregate_bars(rows, symbol)


def _validate_cache_mode(cache_mode: str) -> str:
    mode = cache_mode.lower()
    if mode not in {"auto", "live", "cache"}:
        raise ValueError("cache_mode must be 'auto', 'live', or 'cache'")
    return mode


def _date_text(value: date | str) -> str:
    if isinstance(value, date):
        return value.isoformat()
    parsed = date.fromisoformat(value)
    return parsed.isoformat()


def unix_to_eastern(values: Iterable) -> pd.Series:
    parsed = pd.to_datetime(values, unit="ms", utc=True, errors="coerce")
    return parsed.dt.tz_convert(EASTERN)


def normalize_aggregate_bars(records: list[dict] | dict | list, ticker: str) -> pd.DataFrame:
    """Map Massive aggregate fields onto the stable CSV schema."""
    if isinstance(records, dict):
        source = records.get("results", records)
        rows = source if isinstance(source, list) else [source]
    else:
        rows = list(records)
    frame = pd.json_normalize(rows).rename(columns=BAR_COLUMNS)
    for column in OUTPUT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    if frame.empty or frame["timestamp"].isna().all():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    frame["timestamp"] = unix_to_eastern(frame["timestamp"])
    frame["ticker"] = ticker
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    return frame.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)


def write_bars(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.reindex(columns=OUTPUT_COLUMNS).copy()
    timestamps = pd.to_datetime(output["timestamp"], utc=True, errors="coerce").dt.tz_convert(EASTERN)
    output["timestamp"] = timestamps.map(lambda value: value.isoformat() if pd.notna(value) else "")
    output.to_csv(path, index=False, lineterminator="\n")
    return path


def ingest_core_daily(
    output_dir: Path,
    start: date | str = INDEX_HISTORY_START,
    end: date | str | None = None,
    client: MassiveClient | None = None,
) -> list[IngestResult]:
    """Download daily SPY, VIX, and VIX3M bars and write one CSV per series."""
    end_value = market_today() if end is None else date.fromisoformat(_date_text(end))
    start_value = date.fromisoformat(_date_text(start))
    if start_value > end_value:
        raise ValueError("start date must be on or before end date")

    api = client or MassiveClient()
    results: list[IngestResult] = []
    for ticker in CORE_TICKERS:
        frame = api.get_aggregate_bars(ticker, start_value, end_value)
        path = write_bars(frame, output_path_for_ticker(output_dir, ticker))
        results.append(IngestResult(ticker=ticker, rows=len(frame), path=path))
    return results
