from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import monotonic, sleep
from typing import Callable

import pandas as pd

from features.option_signal_engine import generate_option_chain_signal
from news.global_news import fetch_global_news
from strategy.rules import generate_signal


@dataclass
class LiveSignalLoop:
    """Broker-agnostic live polling loop.

    The application supplies candle and option-chain callbacks. The loop
    deliberately does not place orders; it produces auditable signals that can
    be consumed by paper trading. A callback may persist each signal.
    """
    fetch_candles: Callable[[], pd.DataFrame]
    fetch_options: Callable[[], list]
    spot_provider: Callable[[], float]
    interval_seconds: int = 30
    on_signal: Callable[[dict], None] | None = None
    running: bool = field(default=False, init=False)

    def run_once(self) -> dict:
        candles = self.fetch_candles()
        spot = float(self.spot_provider())
        news = fetch_global_news()
        technical = generate_signal(candles)
        option_signal, _ = generate_option_chain_signal(
            self.fetch_options(),
            spot=spot,
            global_news_score=news.sentiment,
        )
        result = {
            "timestamp": datetime.now(),
            "technical_direction": technical.direction,
            "technical_stop_loss": technical.stop_loss,
            "technical_target": technical.target,
            "option_direction": option_signal.direction,
            "option_confidence": option_signal.confidence,
            "option_entry": option_signal.entry_price,
            "option_stop_loss": option_signal.stop_loss,
            "option_take_profit": option_signal.take_profit,
            "global_news": news.sentiment,
        }
        if self.on_signal:
            self.on_signal(result)
        return result

    def run(self, max_cycles: int | None = None) -> None:
        self.running = True
        cycles = 0
        while self.running and (max_cycles is None or cycles < max_cycles):
            started = monotonic()
            self.run_once()
            cycles += 1
            sleep(max(0.0, self.interval_seconds - (monotonic() - started)))

    def stop(self) -> None:
        self.running = False
