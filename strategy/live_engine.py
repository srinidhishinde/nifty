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

    def __init__(self, instrument: str, config: StrategyConfig | None = None):
        self.instrument = instrument.upper()
        self.config = config or StrategyConfig()

    def evaluate(self, candles: pd.DataFrame) -> LiveSignalEvent | None:
        if candles.empty or "timestamp" not in candles:
            return None
        timestamp = pd.to_datetime(candles.iloc[-1]["timestamp"])
        if not market_is_open(timestamp, self.instrument):
            return None
        return LiveSignalEvent(
            timestamp=timestamp.to_pydatetime(),
            instrument=self.instrument,
            signal=generate_signal(candles, self.config),
        )
