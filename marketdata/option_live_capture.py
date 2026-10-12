"""Read-only Kotak SFeed capture for a selected NSE option contract at 1-minute resolution."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from marketdata.kotak_live import LiveTick, normalize_sfeed_message

IST = ZoneInfo("Asia/Kolkata")


class OneMinuteCandleBuilder:
    """Aggregate broker ticks into completed one-minute candles."""

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}
        self._last_cumulative_volume: dict[str, float] = {}

    def update(self, tick: LiveTick) -> list[dict[str, Any]]:
        key = f"{tick.exchange_segment}|{tick.instrument_token}"
        bucket = pd.Timestamp(tick.timestamp).floor("1min")
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
        current_bucket = pd.Timestamp(now).floor("1min")
        rows = []
        for key, state in list(self._states.items()):
            if state["bucket"] < current_bucket:
                rows.append(self._row(state))
                del self._states[key]
        return rows

    @staticmethod
    def _row(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": state["bucket"].to_pydatetime(),
            "instrument": state["instrument"],
            "exchange_segment": state["exchange_segment"],
            "instrument_token": state["instrument_token"],
            "open": state["open"], "high": state["high"], "low": state["low"],
            "close": state["close"], "volume": state["volume"],
            "open_interest": state["open_interest"], "tick_count": state["tick_count"],
            "data_source": "KOTAK_OPTION_SFEED_1M",
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
