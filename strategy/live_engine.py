from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from strategy.market_hours import market_is_open
from strategy.rules import StrategyConfig, StrategySignal, generate_signal

@dataclass(frozen=True)
class LiveSignalEvent:
    timestamp: datetime
    instrument: str
    signal: StrategySignal

class LiveRuleEngine:
    """Evaluate completed candles supplied by a live feed; never places orders."""

    def __init__(self, instrument: str, config: StrategyConfig | None = None, timeframe_minutes: int = 5):
        self.instrument = instrument.upper()
        self.config = config or StrategyConfig()
        if timeframe_minutes not in {1, 5, 10, 15}:
            raise ValueError("timeframe_minutes must be one of 1, 5, 10 or 15")
        self.timeframe_minutes = timeframe_minutes

    def evaluate(self, candles: pd.DataFrame) -> LiveSignalEvent | None:
        if candles.empty or "timestamp" not in candles:
            return None
        frame = candles.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
        if frame.empty:
            return None
        timestamp = frame.iloc[-1]["timestamp"]
        now = pd.Timestamp.now(tz="Asia/Kolkata")
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("Asia/Kolkata")
        else:
            timestamp = timestamp.tz_convert("Asia/Kolkata")
        # Candle timestamps represent interval starts. Never evaluate the live,
        # still-forming candle; this keeps live behavior aligned with backtests.
        if now < timestamp + pd.Timedelta(minutes=self.timeframe_minutes):
            frame = frame.iloc[:-1]
        if frame.empty:
            return None
        timestamp = frame.iloc[-1]["timestamp"]
        if not market_is_open(timestamp, self.instrument):
            return None
        return LiveSignalEvent(
            timestamp=timestamp.to_pydatetime(),
            instrument=self.instrument,
            signal=generate_signal(frame, self.config),
        )
