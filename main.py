"""Download the daily SPY, VIX, and VIX3M series used by the research project."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from data import INDEX_HISTORY_START, MassiveAPIError, MassiveConfigError, ingest_core_daily, market_today


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use YYYY-MM-DD") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download daily SPY, VIX, and VIX3M bars from the Massive API."
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=INDEX_HISTORY_START,
        help=f"Inclusive start date (YYYY-MM-DD). Default: {INDEX_HISTORY_START.isoformat()}",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        help="Inclusive end date (YYYY-MM-DD). Default: today in America/New_York",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for the CSV files. Default: data/raw",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    end = args.end or market_today()
    if args.start > end:
        print("start date must be on or before end date", file=sys.stderr)
        return 2
    try:
        results = ingest_core_daily(args.output_dir, args.start, end)
    except (MassiveConfigError, MassiveAPIError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for result in results:
        print(f"{result.ticker:<8} {result.rows:>6} rows  {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
