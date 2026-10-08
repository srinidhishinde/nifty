from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import monotonic, sleep
from typing import Callable

import pandas as pd

from features.option_signal_engine import generate_option_chain_signal
from news.global_news import fetch_global_news
from strategy.ai_signal import generate_ai_signal


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
    capital: float | None = None
    _last_signal_key: tuple | None = field(default=None, init=False)
    on_signal: Callable[[dict], None] | None = None
    running: bool = field(default=False, init=False)

    def run_once(self) -> dict:
        if self.capital is None or float(self.capital) <= 0:
            raise ValueError(
                "LiveSignalLoop requires explicit account capital; "
                "no default capital is permitted."
            )
        candles = self.fetch_candles()
        spot = float(self.spot_provider())
        news = fetch_global_news()
        ai = generate_ai_signal(
            candles,
            capital=float(self.capital),
            global_news_score=news.sentiment,
        )
        option_signal, _ = generate_option_chain_signal(
            self.fetch_options(),
            spot=spot,
            global_news_score=news.sentiment,
        )
        result = {
            "timestamp": datetime.now(),
            "signal_candle_time": candles.iloc[-1]["timestamp"] if not candles.empty else None,
            "technical_direction": ai.direction,
            "technical_stop_loss": ai.stop_loss,
            "technical_target": ai.take_profit,
            "ai_confidence": ai.confidence,
            "ai_abstain": ai.abstain,
            "position_quantity": ai.quantity,
            "ai_reasons": ai.reasons,
            "option_direction": option_signal.direction,
            "option_confidence": option_signal.confidence,
            "option_entry": option_signal.entry_price,
            "option_stop_loss": option_signal.stop_loss,
            "option_take_profit": option_signal.take_profit,
            "global_news": news.sentiment,
        }
        signal_key = (result["signal_candle_time"], result["technical_direction"], ai.confidence, ai.stop_loss, ai.take_profit)
        if signal_key == self._last_signal_key:
            result["duplicate_signal"] = True
        else:
            result["duplicate_signal"] = False
            self._last_signal_key = signal_key
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
