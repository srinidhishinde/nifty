"""Yahoo Finance historical market-data adapter.

This module deliberately uses Yahoo's chart endpoint instead of silently
falling back to synthetic data. Intraday retention is provider-controlled,
so the adapter reports the returned coverage and rejects empty/incomplete
responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import pandas as pd
import requests


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Yahoo commonly limits intraday history. Keep the guard conservative so the
# dashboard explains the limitation rather than returning misleading data.
INTRADAY_MAX_DAYS = 59


@dataclass(frozen=True)
class YahooOHLCVResult:
    data: pd.DataFrame
    symbol: str
    interval: str
    requested_start: date
    requested_end: date
    provider_start: date | None
    provider_end: date | None
    status: str
    message: str


def _epoch_seconds(value: date | datetime, end_of_day: bool = False) -> int:
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = datetime.combine(
            value,
            time.max if end_of_day else time.min,
            tzinfo=timezone.utc,
        )
    return int(dt.timestamp())


def _normalise_interval(interval: str) -> str:
    allowed = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1d"}
    value = str(interval).strip()
    if value not in allowed:
        raise ValueError(f"Unsupported Yahoo interval: {value}")
    return value


def fetch_yahoo_ohlcv(
    start: date | datetime,
    end: date | datetime,
    *,
    symbol: str = "^NSEI",
    interval: str = "5m",
    timeout: int = 20,
) -> YahooOHLCVResult:
    """Fetch OHLCV from Yahoo Finance without synthetic fallback."""

    interval = _normalise_interval(interval)

    start_date = start.date() if isinstance(start, datetime) else start
    end_date = end.date() if isinstance(end, datetime) else end

    if end_date < start_date:
        raise ValueError("Yahoo backtest end date must be on/after start date.")

    if interval != "1d":
        requested_days = (end_date - start_date).days + 1
        if requested_days > INTRADAY_MAX_DAYS:
            return YahooOHLCVResult(
                pd.DataFrame(),
                symbol,
                interval,
                start_date,
                end_date,
                None,
                None,
                "LIMITED",
                (
                    f"Yahoo intraday interval {interval} is requested for "
                    f"{requested_days} days. The dashboard limits intraday "
                    f"requests to {INTRADAY_MAX_DAYS} days to avoid relying "
                    "on unavailable provider history."
                ),
            )

    url = YAHOO_CHART_URL.format(symbol=symbol.replace("^", "%5E"))
    params = {
        "period1": _epoch_seconds(start),
        "period2": _epoch_seconds(end, end_of_day=True) + 1,
        "interval": interval,
        "events": "history",
        "includeAdjustedClose": "true",
        "includePrePost": "false",
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; NiftyResearchTerminal/1.0)"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except requests.RequestException as exc:
        return YahooOHLCVResult(
            pd.DataFrame(),
            symbol,
            interval,
            start_date,
            end_date,
            None,
            None,
            "ERROR",
            f"Yahoo request failed: {exc}",
        )
    except ValueError as exc:
        return YahooOHLCVResult(
            pd.DataFrame(),
            symbol,
            interval,
            start_date,
            end_date,
            None,
            None,
            "ERROR",
            f"Yahoo returned invalid JSON: {exc}",
        )

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        description = error.get("description") or str(error)
        return YahooOHLCVResult(
            pd.DataFrame(),
            symbol,
            interval,
            start_date,
            end_date,
            None,
            None,
            "ERROR",
            f"Yahoo rejected the request: {description}",
        )

    results = chart.get("result") or []
    if not results:
        return YahooOHLCVResult(
            pd.DataFrame(),
            symbol,
            interval,
            start_date,
            end_date,
            None,
            None,
            "NO_DATA",
            "Yahoo returned no historical candles for the requested range.",
        )

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]

    if not timestamps:
        return YahooOHLCVResult(
            pd.DataFrame(),
            symbol,
            interval,
            start_date,
            end_date,
            None,
            None,
            "NO_DATA",
            "Yahoo returned no timestamps for the requested range.",
        )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )

    frame = frame.dropna(
        subset=["timestamp", "open", "high", "low", "close", "volume"]
    ).copy()

    if frame.empty:
        return YahooOHLCVResult(
            pd.DataFrame(),
            symbol,
            interval,
            start_date,
            end_date,
            None,
            None,
            "NO_DATA",
            "Yahoo returned timestamps but no complete OHLCV candles.",
        )

    # The strategy/backtest engine operates in exchange-local time.
    frame["timestamp"] = frame["timestamp"].dt.tz_convert("Asia/Kolkata")
    frame = (
        frame.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    numeric = ["open", "high", "low", "close", "volume"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric).reset_index(drop=True)

    provider_start = frame["timestamp"].min().date()
    provider_end = frame["timestamp"].max().date()

    coverage_message = (
        f"Yahoo Finance {symbol} | interval {interval} | "
        f"{len(frame):,} candles | returned {provider_start} to {provider_end}"
    )

    return YahooOHLCVResult(
        frame,
        symbol,
        interval,
        start_date,
        end_date,
        provider_start,
        provider_end,
        "OK",
        coverage_message,
    )
