"""Pre-candle forecast schedule for one-minute NIFTY options research.

This is an advisory forecast generator, not an order execution engine. A
forecast for the next minute is emitted using only candles completed before
that minute. Exact highs/lows and profitable fills cannot be guaranteed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NextCandleForecast:
    generated_at: str
    target_candle_time: str
    underlying: str
    reference_price: float
    predicted_low: float
    predicted_high: float
    direction: str
    confidence_label: str
    method: str
    history_bars: int
    disclaimer: str = "Estimate only; candle extremes and fills are not guaranteed."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def forecast_next_minute(
    completed_candles: pd.DataFrame,
    *,
    underlying: str = "NIFTY",
    generated_at: str | pd.Timestamp | None = None,
    lookback: int = 30,
    quantile: float = 0.80,
) -> NextCandleForecast:
    """Forecast the next minute before it begins.

    Input columns: timestamp, open, high, low, close, volume. The final input
    row must be the latest fully completed candle. The target timestamp is
    one minute after its timestamp. If a market is closed or a minute is
    missing, the caller should align the target timestamp to the next actual
    trading minute using the exchange calendar.
    """
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(completed_candles.columns))
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    if completed_candles.empty:
        raise ValueError("At least one completed candle is required")
    if lookback < 3:
        raise ValueError("lookback must be at least 3")
    if not 0.5 < quantile < 1:
        raise ValueError("quantile must be between 0.5 and 1.0")

    df = completed_candles.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[list(required)].isna().any().any():
        raise ValueError("OHLCV input contains missing or invalid values")
    df = df.sort_values("timestamp").reset_index(drop=True)
    last = df.iloc[-1]
    history = df.tail(lookback)
    reference = float(last["close"])

    # Use the distribution of recent intraminute extremes, scaled to the
    # reference price. This is a transparent baseline to replace after
    # walk-forward validation, not a claim of high predictive accuracy.
    down = (history["open"] - history["low"]).clip(lower=0).quantile(quantile)
    up = (history["high"] - history["open"]).clip(lower=0).quantile(quantile)
    floor = max(reference * 0.0005, 0.01)
    down = max(float(down), floor)
    up = max(float(up), floor)

    # Direction uses only completed closes. No current/future candle data.
    closes = history["close"].astype(float)
    delta = float(closes.iloc[-1] - closes.iloc[0])
    direction = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"

    # Confidence is deliberately qualitative until calibrated on unseen data.
    if len(history) >= lookback:
        confidence_label = "BASELINE"
    else:
        confidence_label = "LOW_HISTORY"

    generated = pd.Timestamp(generated_at) if generated_at is not None else pd.Timestamp.now()
    target_time = pd.Timestamp(last["timestamp"]) + pd.Timedelta(minutes=1)
    return NextCandleForecast(
        generated_at=generated.isoformat(),
        target_candle_time=target_time.isoformat(),
        underlying=underlying,
        reference_price=round(reference, 2),
        predicted_low=round(max(0.01, reference - down), 2),
        predicted_high=round(reference + up, 2),
        direction=direction,
        confidence_label=confidence_label,
        method=f"rolling_{lookback}_bar_open_extreme_q{int(quantile * 100)}",
        history_bars=len(history),
    )
