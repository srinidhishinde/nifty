from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd

from analysis.data_quality import assess_ohlcv
from features.technical.indicators import add_indicators
from marketdata.providers.kotak_neo import KotakNeoProvider
from strategy.rules import StrategyConfig, StrategySignal, _generate_signal_from_enriched


@dataclass(frozen=True)
class DecisionSnapshot:
    source: str
    mode: str
    status: str
    message: str
    timestamp: pd.Timestamp
    frame: pd.DataFrame | None
    option_chain: list[Any]
    quality_status: str
    quality_reasons: tuple[str, ...]
    pcr_oi: float | None
    pcr_volume: float | None
    option_count: int
    signal: StrategySignal


def _option_features(contracts: list[Any]) -> tuple[float | None, float | None]:
    calls = [x for x in contracts if getattr(x, "option_type", "") == "CE"]
    puts = [x for x in contracts if getattr(x, "option_type", "") == "PE"]
    call_oi = sum(float(getattr(x, "open_interest", 0) or 0) for x in calls)
    put_oi = sum(float(getattr(x, "open_interest", 0) or 0) for x in puts)
    call_volume = sum(float(getattr(x, "volume", 0) or 0) for x in calls)
    put_volume = sum(float(getattr(x, "volume", 0) or 0) for x in puts)
    pcr_oi = put_oi / call_oi if call_oi > 0 else None
    pcr_volume = put_volume / call_volume if call_volume > 0 else None
    return pcr_oi, pcr_volume


def _wait(message: str, timestamp: pd.Timestamp, frame: pd.DataFrame | None = None,
          quality_status: str = "RED", quality_reasons: tuple[str, ...] = ()) -> StrategySignal:
    return StrategySignal("WAIT", (), (message,), 0.0, 0.0, 0.0, 0.0, False)


def load_kotak_decision_snapshot(
    broker: Any,
    *,
    instrument: str = "NIFTY",
    timeframe: str = "5m",
    history_days: int = 10,
    option_count: int = 40,
    option_expiry: str | None = None,
    config: StrategyConfig | None = None,
) -> DecisionSnapshot:
    """Build the production decision input strictly from Kotak Neo.

    This function deliberately has no Yahoo fallback. If Kotak is unavailable,
    the production decision remains WAIT.
    """
    now = pd.Timestamp.now(tz="Asia/Kolkata")

    if broker is None or not broker.connection_status().connected:
        return DecisionSnapshot(
            "KOTAK_NEO", "LIVE", "RED", "Kotak Neo is not connected; live decision is blocked.",
            now, None, [], "RED", ("Kotak Neo connection required",), None, None, 0,
            _wait("Kotak Neo connection required", now),
        )

    try:
        provider = KotakNeoProvider(broker.client)

        if instrument != "NIFTY":
            return DecisionSnapshot(
                "KOTAK_NEO", "LIVE", "RED",
                f"Production decision adapter currently requires NIFTY; {instrument} is not wired for live candles.",
                now, None, [], "RED", ("Live non-NIFTY candle mapping is not configured",), None, None, 0,
                _wait("Live instrument mapping unavailable", now),
            )

        end = now.date()
        start = end - timedelta(days=max(3, int(history_days)))
        candles = provider.get_historical_candles(
            symbol="Nifty 50",
            exchange="NSE",
            timeframe=timeframe,
            start=start,
            end=end,
        )
        rows = [{
            "timestamp": x.timestamp,
            "open": x.open,
            "high": x.high,
            "low": x.low,
            "close": x.close,
            "volume": x.volume,
        } for x in candles]
        frame = pd.DataFrame(rows)

        if frame.empty:
            raise RuntimeError("Kotak Neo returned no completed NIFTY candles.")

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

        quality = assess_ohlcv(frame, expected_minutes=5 if timeframe == "5m" else 15)
        if quality.status == "RED" or len(frame) < 60:
            reason = " | ".join(quality.reasons) or "insufficient completed candles"
            return DecisionSnapshot(
                "KOTAK_NEO", "LIVE", "RED",
                f"Kotak Neo data-quality gate blocked the decision: {reason}",
                now, frame, [], quality.status, tuple(quality.reasons), None, None, 0,
                _wait(f"Data-quality gate: {reason}", now, frame, quality.status, tuple(quality.reasons)),
            )

        enriched = add_indicators(frame.copy())

        # Option-chain data is mandatory for an actual CE/PE decision.
        contracts = provider.get_option_chain(
            underlying="NIFTY",
            exchange="NSE_FO",
            expiry=option_expiry,
            count=option_count,
            enrich_quotes=True,
        )
        pcr_oi, pcr_volume = _option_features(contracts)

        if not contracts or pcr_oi is None:
            signal = _wait("Live option-chain/OI data is unavailable; CE/PE trade is blocked.", now, enriched)
        else:
            enriched["PCR_OI"] = pcr_oi
            enriched["PCR"] = pcr_oi
            if pcr_volume is not None:
                enriched["PCR_VOLUME"] = pcr_volume
            signal = _generate_signal_from_enriched(
                enriched,
                config or StrategyConfig(require_option_confirmation=True),
            )

        return DecisionSnapshot(
            "KOTAK_NEO", "LIVE", quality.status, "Kotak Neo is the primary decision source.",
            now, enriched, contracts, quality.status, tuple(quality.reasons),
            pcr_oi, pcr_volume, len(contracts), signal,
        )
    except Exception as exc:
        return DecisionSnapshot(
            "KOTAK_NEO", "LIVE", "RED",
            f"Kotak Neo decision-data error: {exc}",
            now, None, [], "RED", (str(exc),), None, None, 0,
            _wait(f"Kotak Neo decision-data error: {exc}", now),
        )
