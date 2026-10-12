"""Read-only Kotak SFeed capture and OHLC candle aggregation for a selected NSE option."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from marketdata.kotak_live import LiveTick, normalize_sfeed_message

IST = ZoneInfo("Asia/Kolkata")


class OptionCandleBuilder:
    """Aggregate broker ticks into completed candles at the configured minute interval."""

    def __init__(self, interval_minutes: int = 15) -> None:
        if interval_minutes not in {1, 3, 5, 15, 30}:
            raise ValueError("interval_minutes must be one of 1, 3, 5, 15, or 30.")
        self.interval_minutes = interval_minutes
        self._frequency = f"{interval_minutes}min"
        self._states: dict[str, dict[str, Any]] = {}
        self._last_cumulative_volume: dict[str, float] = {}

    def update(self, tick: LiveTick) -> list[dict[str, Any]]:
        key = f"{tick.exchange_segment}|{tick.instrument_token}"
        bucket = pd.Timestamp(tick.timestamp).floor(self._frequency)
        volume_delta = 0.0
        if tick.volume is not None:
            current = max(float(tick.volume), 0.0)
            previous = self._last_cumulative_volume.get(key)
            self._last_cumulative_volume[key] = current
            volume_delta = 0.0 if previous is None else (current if current < previous else current - previous)

        state = self._states.get(key)
        completed: list[dict[str, Any]] = []
        if state is not None and bucket > state["bucket"]:
            completed.append(self._row(state))
            state = None
            del self._states[key]

        if state is None:
            self._states[key] = {
                "bucket": bucket, "instrument": tick.trading_symbol or tick.instrument,
                "exchange_segment": tick.exchange_segment, "instrument_token": tick.instrument_token,
                "open": tick.ltp, "high": tick.ltp, "low": tick.ltp, "close": tick.ltp,
                "volume": volume_delta, "open_interest": tick.open_interest, "tick_count": 1,
            }
        else:
            state["high"] = max(state["high"], tick.ltp)
            state["low"] = min(state["low"], tick.ltp)
            state["close"] = tick.ltp
            state["volume"] += volume_delta
            state["open_interest"] = tick.open_interest
            state["tick_count"] += 1
        return completed

    def flush_completed(self, now: datetime) -> list[dict[str, Any]]:
        current_bucket = pd.Timestamp(now).floor(self._frequency)
        rows = []
        for key, state in list(self._states.items()):
            if state["bucket"] < current_bucket:
                rows.append(self._row(state))
                del self._states[key]
        return rows

    def _row(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": state["bucket"].to_pydatetime(),
            "instrument": state["instrument"],
            "exchange_segment": state["exchange_segment"],
            "instrument_token": state["instrument_token"],
            "open": state["open"], "high": state["high"], "low": state["low"],
            "close": state["close"], "volume": state["volume"],
            "open_interest": state["open_interest"], "tick_count": state["tick_count"],
            "timeframe": f"{self.interval_minutes}min",
            "data_source": f"KOTAK_OPTION_SFEED_{self.interval_minutes}M",
        }


async def stream_option_sfeed(
    client: Any,
    instrument_token: str,
    seconds: int,
    on_tick: Callable[[LiveTick], None],
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> dict[str, int]:
    """Stream only the selected NSE-F&O option token; no order APIs are called."""
    if not 5 <= seconds <= 86_400:
        raise ValueError("seconds must be between 5 and 86400.")
    token = str(instrument_token).strip()
    if not token:
        raise ValueError("Selected option has no Kotak instrument token.")

    from neo_api_client.websocket.feed import WsToken

    counts = {"ticks": 0, "invalid": 0}
    subscription = [WsToken("nse_fo", token)]
    print("[INFO] Opening Kotak SFeed WebSocket for selected NSE option...", flush=True)
    async with client.create_websocket() as ws:
        print("[INFO] WebSocket opened; subscribing to selected option...", flush=True)
        await ws.subscribe_scrips(subscription)
        print("[INFO] Selected option subscribed; enabling exchange feed...", flush=True)
        await ws.subscribe_exchange()
        print(f"[INFO] Option feed active; capture window={seconds}s. Heartbeat every 15s.", flush=True)

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(15)
                if progress_callback:
                    progress_callback(dict(counts))

        task = asyncio.create_task(heartbeat())
        try:
            async with asyncio.timeout(seconds):
                async for message in ws:
                    tick = normalize_sfeed_message(message)
                    if tick is None:
                        counts["invalid"] += 1
                        continue
                    if tick.exchange_segment != "nse_fo" or tick.instrument_token != token:
                        continue
                    on_tick(tick)
                    counts["ticks"] += 1
        except TimeoutError:
            pass
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            try:
                await ws.unsubscribe_exchange()
            finally:
                await ws.unsubscribe_scrips(subscription)
    return counts


class OneMinuteCandleBuilder(OptionCandleBuilder):
    """Backward-compatible one-minute builder."""

    def __init__(self) -> None:
        super().__init__(interval_minutes=1)


class FifteenMinuteCandleBuilder(OptionCandleBuilder):
    """Build completed 15-minute candles from selected-option live ticks."""

    def __init__(self) -> None:
        super().__init__(interval_minutes=15)
