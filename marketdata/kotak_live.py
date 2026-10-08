from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class LiveTick:
    """Normalized broker-originated market-data tick."""

    instrument: str
    exchange_segment: str
    instrument_token: str
    trading_symbol: str
    timestamp: datetime
    ltp: float
    volume: float | None = None
    open_interest: float | None = None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(IST) if value.tzinfo else value.replace(tzinfo=IST)
    if isinstance(value, (int, float)):
        try:
            seconds = float(value)
            if seconds > 10_000_000_000:
                seconds /= 1000.0
            return datetime.fromtimestamp(seconds, tz=IST)
        except (OverflowError, OSError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=IST)
            except ValueError:
                continue
        return None
    return parsed.astimezone(IST) if parsed.tzinfo else parsed.replace(tzinfo=IST)


def normalize_sfeed_message(message: Any) -> LiveTick | None:
    """Convert a Kotak SFeed message into the application's tick contract.

    Invalid broker messages are rejected. No price or timestamp is fabricated.
    """

    segment = str(getattr(message, "exchange_segment", "") or "").strip().lower()
    token = str(getattr(message, "instrument_token", "") or "").strip()
    symbol = str(getattr(message, "trading_symbol", "") or "").strip()
    ltp = _number(getattr(message, "last_traded_price", None))
    timestamp = _timestamp(getattr(message, "last_update_time", None) or getattr(message, "last_trade_time", None) or getattr(message, "timestamp", None))

    if not segment or not token or ltp is None or ltp <= 0 or timestamp is None:
        return None

    is_nifty = segment == "nse_cm" and (
        symbol.upper() in {"NIFTY", "NIFTY 50", "NIFTY50"} or token == "Nifty 50"
    )
    instrument = "NIFTY" if is_nifty else symbol or token

    volume = _number(
        getattr(message, "volume_traded_today", None) if getattr(message, "volume_traded_today", None) is not None else getattr(message, "volume", None)
        or getattr(message, "total_volume", None)
        or getattr(message, "volume_trade_for_the_day", None)
    )
    open_interest = _number(
        getattr(message, "open_interest", None)
        or getattr(message, "open_int", None)
    )

    return LiveTick(
        instrument=instrument,
        exchange_segment=segment,
        instrument_token=token,
        trading_symbol=symbol or token,
        timestamp=timestamp,
        ltp=ltp,
        volume=volume,
        open_interest=open_interest,
    )


@dataclass
class _CandleState:
    bucket: pd.Timestamp
    instrument: str
    exchange_segment: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: float | None
    ticks: int


class FiveMinuteCandleBuilder:
    """Build completed 5-minute candles from live ticks only."""

    def __init__(self, timeframe: str = "5min") -> None:
        if timeframe != "5min":
            raise ValueError("FiveMinuteCandleBuilder currently supports only 5min.")
        self._states: dict[str, _CandleState] = {}
        self._last_cumulative_volume: dict[str, float] = {}

    @staticmethod
    def _key(tick: LiveTick) -> str:
        return f"{tick.exchange_segment}|{tick.instrument_token}"

    def _volume_delta(self, tick: LiveTick) -> float:
        if tick.volume is None:
            return 0.0
        key = self._key(tick)
        current = max(float(tick.volume), 0.0)
        previous = self._last_cumulative_volume.get(key)
        self._last_cumulative_volume[key] = current
        if previous is None:
            return 0.0
        return current if current < previous else current - previous

    def update(self, tick: LiveTick) -> list[dict[str, Any]]:
        bucket = pd.Timestamp(tick.timestamp).floor("5min")
        key = self._key(tick)
        state = self._states.get(key)
        completed: list[dict[str, Any]] = []
        # Kotak SFeed volume_traded_today is cumulative; convert it before
        # assigning/accumulating candle volume so every update has a defined
        # per-tick delta, including the first tick of a new bucket.
        volume_delta: float = self._volume_delta(tick)

        if state is not None and bucket > state.bucket:
            completed.append(self._to_row(state))
            del self._states[key]
            state = None

        if state is None:
            self._states[key] = _CandleState(
                bucket=bucket,
                instrument=tick.instrument,
                exchange_segment=tick.exchange_segment,
                open=tick.ltp,
                high=tick.ltp,
                low=tick.ltp,
                close=tick.ltp,
                volume=volume_delta,
                open_interest=tick.open_interest,
                ticks=1,
            )
        else:
            state.high = max(state.high, tick.ltp)
            state.low = min(state.low, tick.ltp)
            state.close = tick.ltp
            state.volume += volume_delta
            state.open_interest = tick.open_interest
            state.ticks += 1
        return completed

    def flush_completed(self, now: datetime) -> list[dict[str, Any]]:
        current_bucket = pd.Timestamp(now).floor("5min")
        completed: list[dict[str, Any]] = []
        for key, state in list(self._states.items()):
            if state.bucket < current_bucket:
                completed.append(self._to_row(state))
                del self._states[key]
        return completed

    @staticmethod
    def _to_row(state: _CandleState) -> dict[str, Any]:
        return {
            "timestamp": state.bucket.to_pydatetime(),
            "instrument": state.instrument,
            "exchange_segment": state.exchange_segment,
            "open": state.open,
            "high": state.high,
            "low": state.low,
            "close": state.close,
            "volume": state.volume,
            "open_interest": state.open_interest,
            "tick_count": state.ticks,
            "data_source": "KOTAK_CAPTURED",
        }


async def stream_kotak_sfeed(
    client: Any,
    mcx_tokens: Iterable[str],
    seconds: int,
    on_tick: Callable[[LiveTick], None],
) -> dict[str, int]:
    """Capture NIFTY index and MCX futures SFeed ticks for a bounded window."""

    if not 5 <= seconds <= 86_400:
        raise ValueError("seconds must be between 5 and 86400.")

    from neo_api_client.websocket.feed import WsToken

    nifty_token = WsToken("nse_cm", "Nifty 50")
    mcx = [WsToken("mcx_fo", str(token)) for token in mcx_tokens if str(token).strip()]
    if not mcx:
        raise ValueError("At least one MCX token is required.")

    counts = {"nifty": 0, "mcx": 0, "invalid": 0, "raw": 0, "market_status": 0, "decoded": 0, "errors": 0, "subscription_count": 0}
    async with client.create_websocket() as ws:
        # Kotak documents Nifty 50 as a valid scrip/LTP subscription. Using
        # the same Scrip feed as MCX gives the capture layer one normalized
        # message type and avoids depending on the separate index decoder.
        tokens = [nifty_token, *mcx]
        await ws.subscribe_scrips(tokens)
        await ws.subscribe_exchange()
        counts["subscription_count"] = int(getattr(ws, "subscription_count", 0) or 0)

        def _on_raw(_raw: str | bytes) -> None:
            counts["raw"] += 1

        def _on_error(_error: Exception) -> None:
            counts["invalid"] += 1

        def _on_connect() -> None:
            return None

        ws.on_raw = _on_raw
        ws.on_error = _on_error
        try:
            async with asyncio.timeout(seconds):
                async for message in ws:
                    tick = normalize_sfeed_message(message)
                    if tick is None:
                        counts["invalid"] += 1
                        continue
                    on_tick(tick)
                    if tick.exchange_segment == "nse_cm":
                        counts["nifty"] += 1
                    elif tick.exchange_segment == "mcx_fo":
                        counts["mcx"] += 1
        except TimeoutError:
            pass
        finally:
            try:
                await ws.unsubscribe_exchange()
            finally:
                await ws.unsubscribe_scrips(tokens)
    return counts
