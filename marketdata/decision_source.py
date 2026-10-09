from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Any

import pandas as pd

from analysis.data_quality import assess_ohlcv
from features.technical.indicators import add_indicators
from marketdata.providers.kotak_neo import KotakNeoProvider
from strategy.rules import StrategyConfig, StrategySignal, _generate_signal_from_enriched
from features.option_chain import OptionContract
from strategy.production_hardening import validate_candle_frame, validate_option_contract, validate_contract_identity, runtime_clock_ok


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


_MCX_SYMBOLS = {
    "CRUDEOIL": "CRUDEOIL",
    "NATURALGAS": "NATURALGAS",
    "COPPER": "COPPER",
    "SILVER": "SILVER",
    "GOLD": "GOLD",
}

def _option_market_open(instrument: str, now: pd.Timestamp) -> bool:
    if now.weekday() >= 5:
        return False
    if instrument.upper() == "NIFTY":
        return pd.Timestamp("09:15").time() <= now.time() <= pd.Timestamp("15:30").time()
    return pd.Timestamp("09:00").time() <= now.time() <= pd.Timestamp("23:30").time()


def _cached_option_contracts(instrument: str, now: pd.Timestamp) -> tuple[list[OptionContract], pd.Timestamp | None]:
    from marketdata.daily_store import DailyMarketStore
    frame, captured_at = DailyMarketStore().load_latest_option_chain_snapshot(instrument, before=now)
    contracts: list[OptionContract] = []
    for row in frame.to_dict("records"):
        try:
            contracts.append(OptionContract(
                symbol=str(row.get("symbol") or ""),
                expiry=str(row.get("expiry") or ""),
                strike=float(row["strike"]),
                option_type=str(row.get("option_type") or "").upper(),
                ltp=float(row.get("ltp") or 0),
                bid=float(row["bid"]) if pd.notna(row.get("bid")) else None,
                ask=float(row["ask"]) if pd.notna(row.get("ask")) else None,
                volume=float(row.get("volume") or 0),
                open_interest=float(row.get("open_interest") or 0),
                oi_change=float(row.get("oi_change") or 0),
                implied_volatility=float(row.get("implied_volatility") or 0),
                built_up=str(row.get("built_up") or ""),
                delta=float(row["delta"]) if pd.notna(row.get("delta")) else None,
                theta=float(row["theta"]) if pd.notna(row.get("theta")) else None,
                vega=float(row["vega"]) if pd.notna(row.get("vega")) else None,
                gamma=float(row["gamma"]) if pd.notna(row.get("gamma")) else None,
                ltp_change_pct=float(row.get("ltp_change_pct") or 0),
                quote_timestamp=pd.Timestamp(row["quote_timestamp"]).to_pydatetime() if row.get("quote_timestamp") not in (None, "") and pd.notna(row.get("quote_timestamp")) else None,
                quote_source=str(row.get("quote_source") or ""),
            ))
        except (TypeError, ValueError):
            continue
    return contracts, captured_at


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


def _wait(
    message: str,
    timestamp: pd.Timestamp,
    frame: pd.DataFrame | None = None,
    quality_status: str = "RED",
    quality_reasons: tuple[str, ...] = (),
) -> StrategySignal:
    return StrategySignal("WAIT", (), (message,), 0.0, 0.0, 0.0, 0.0, False)


def _history_request(provider: KotakNeoProvider, instrument: str, timeframe: str, start: date, end: date) -> tuple[str, str, str | None]:
    if instrument == "NIFTY":
        return "Nifty 50", "NSE", None

    symbol = _MCX_SYMBOLS.get(instrument)
    if symbol is None:
        raise RuntimeError(f"Live instrument mapping is not configured for {instrument}.")

    contract = provider.resolve_mcx_futures(symbol)
    return contract["trading_symbol"] or symbol, "MCX_FO", contract["neosymbol"]


def load_kotak_decision_snapshot(
    broker: Any,
    *,
    instrument: str = "NIFTY",
    timeframe: str = "5m",
    history_days: int = 10,
    option_count: int = 40,
    option_expiry: str | None = None,
    config: StrategyConfig | None = None,
    now: pd.Timestamp | None = None,
) -> DecisionSnapshot:
    """Build production decision input from broker option-chain + captured Kotak SFeed candles.

    Both NIFTY and MCX use the same fail-closed architecture:
      Kotak option chain + persisted Kotak SFeed 5m candles -> decision engine.
    The Kotak historical endpoint is never used as a live decision fallback,
    and neither Yahoo nor synthetic candles/contracts are substituted.
    """
    now = pd.Timestamp(now or pd.Timestamp.now(tz="Asia/Kolkata"))
    if now.tzinfo is None:
        now = now.tz_localize("Asia/Kolkata")
    else:
        now = now.tz_convert("Asia/Kolkata")

    if broker is None or not broker.connection_status().connected:
        return DecisionSnapshot(
            "KOTAK_NEO", "LIVE", "RED",
            "Kotak Neo is not connected; live decision is blocked.",
            now, None, [], "RED", ("Kotak Neo connection required",), None, None, 0,
            _wait("Kotak Neo connection required", now),
        )

    # Retain read-only broker evidence if a later processing step fails.
    # Initialize before entering the pipeline so exception handling never hides
    # a chain that was already received from Kotak.
    display_contracts: list[Any] = []
    pcr_oi: float | None = None
    pcr_volume: float | None = None
    instrument_upper = instrument.upper()

    try:
        if instrument_upper not in {"NIFTY", *_MCX_SYMBOLS}:
            raise RuntimeError(f"Live instrument mapping is not configured for {instrument_upper}.")

        provider = KotakNeoProvider(broker.client)

        if instrument_upper == "NIFTY":
            chain_exchange, chain_underlying = "NSE_FO", "NIFTY"
        else:
            chain_exchange, chain_underlying = "MCX_FO", instrument_upper

        # OPEN: fetch the current broker chain. CLOSED: do not poll a live
        # endpoint; load the latest completed real broker snapshot instead.
        market_open = _option_market_open(instrument_upper, now)
        cached_captured_at = None
        if market_open:
            contracts = provider.get_option_chain(
                underlying=chain_underlying,
                exchange=chain_exchange,
                expiry=option_expiry,
                count=option_count,
                enrich_quotes=True,
            )
        else:
            contracts, cached_captured_at = _cached_option_contracts(instrument_upper, now)
        # Preserve the broker-returned chain for read-only display and audit.
        # A separate eligible set controls every executable decision; displaying
        # an unverified contract must never make it tradeable.
        display_contracts = list(contracts)
        pcr_oi, pcr_volume = _option_features(display_contracts)
        executable_contracts = display_contracts

        if market_open:
            executable_contracts = []
            for contract in display_contracts:
                ok, _ = validate_option_contract(
                    contract, now=now, max_quote_age_seconds=60.0,
                    require_broker_quote=True, max_spread_pct=0.05,
                )
                identity_ok, _ = validate_contract_identity(
                    contract, expected_instrument=instrument_upper,
                    expected_exchange=chain_exchange.lower(),
                )
                if ok and identity_ok:
                    executable_contracts.append(contract)
            pcr_oi, pcr_volume = _option_features(executable_contracts)

        # Chain visibility and execution readiness are different states. Keep
        # the real raw chain visible even if the identity/quote gates reject it.
        if not display_contracts:
            return DecisionSnapshot(
                "KOTAK_NEO", "LIVE", "RED",
                f"Real {instrument_upper} option-chain data is unavailable.",
                now, None, [], "RED",
                (f"{instrument_upper} option-chain data unavailable",),
                None, None, 0,
                _wait(
                    f"Real {instrument_upper} option-chain data is unavailable; CE/PE trade is blocked.",
                    now,
                ),
            )

        missing_executable_sides = False
        if market_open:
            sides = {str(getattr(c, "option_type", "")).upper() for c in executable_contracts}
            missing_executable_sides = not {"CE", "PE"}.issubset(sides)
            contracts = executable_contracts
            # Do not return yet when CE/PE evidence is incomplete: first report
            # the independent candle-capture state. Missing quotes still block
            # every decision below; this only improves diagnostics and visibility.
            if not missing_executable_sides:
                if instrument_upper == "NIFTY":
                    underlying_quote = provider.get_index_quote("Nifty 50")
                    underlying_ts = pd.Timestamp(underlying_quote.timestamp)
                    if getattr(underlying_quote, "timestamp_source", "BROKER") != "BROKER":
                        raise RuntimeError("NIFTY underlying quote lacks authoritative broker timestamp.")
                else:
                    mcx_contract = provider.resolve_mcx_futures(_MCX_SYMBOLS[instrument_upper])
                    underlying_quote = provider.get_mcx_quote(mcx_contract)
                    raw_ts = underlying_quote.get("timestamp")
                    underlying_ts = pd.Timestamp(raw_ts) if raw_ts not in (None, "") else None
                if underlying_ts is None or not runtime_clock_ok(reference_timestamp=underlying_ts):
                    return DecisionSnapshot(
                        "KOTAK_NEO", "LIVE", "RED",
                        "Broker underlying quote timestamp is missing, stale or not clock-aligned.",
                        now, None, contracts, "RED",
                        ("Underlying broker timestamp validation failed.",),
                        pcr_oi, pcr_volume, len(contracts),
                        _wait("Underlying broker quote is not decision-ready.", now),
                    )

        if not market_open:
            return DecisionSnapshot(
                "KOTAK_NEO", "LAST_SESSION", "GREEN" if contracts else "RED",
                (
                    f"Market closed. Showing the latest completed {instrument_upper} Kotak option-chain snapshot"
                    + (f" from {cached_captured_at.strftime('%Y-%m-%d %H:%M:%S %Z')}." if cached_captured_at is not None else ".")
                    + " No current trade decision is generated while the market is closed."
                ),
                now, None, contracts, "GREEN" if contracts else "RED",
                ("Market closed; last completed broker snapshot only",),
                pcr_oi, pcr_volume, len(contracts),
                _wait("Market closed; current CE/PE trade decision is blocked.", now),
            )

        # Live decision candles come only from the persisted Kotak SFeed
        # recorder. This is the common path for NIFTY and MCX; no historical
        # API, Yahoo data, or synthetic candles are allowed here.
        from marketdata.daily_store import load_captured_candles

        end = now.date()
        start = end - timedelta(days=max(3, int(history_days)))
        frame, source = load_captured_candles(
            instrument_upper,
            start=start,
            end=end,
        )

        if frame.empty:
            return DecisionSnapshot(
                "KOTAK_NEO", "LIVE", "RED",
                (
                    f"Real {instrument_upper} {timeframe} candle history is unavailable. "
                    "Start the Kotak SFeed recorder; no synthetic/Yahoo fallback is permitted."
                ),
                now, None, display_contracts, "RED",
                (
                    f"{instrument_upper} captured SFeed candle history unavailable",
                    "Run the real Kotak SFeed capture before enabling this decision path.",
                    *(
                        ("Both CE and PE must pass current quote and scrip-master identity validation.",)
                        if market_open and missing_executable_sides
                        else ()
                    ),
                ),
                pcr_oi, pcr_volume, len(display_contracts),
                _wait(
                    f"Real {instrument_upper} {timeframe} candle history is unavailable.",
                    now,
                ),
            )

        if market_open and missing_executable_sides:
            return DecisionSnapshot(
                "KOTAK_NEO", "LIVE", "RED",
                f"Broker option-chain is visible, but verified executable CE and PE quotes are unavailable; captured candle source is {source}.",
                now, frame, display_contracts, "RED",
                ("Both CE and PE must pass current quote and scrip-master identity validation.",),
                pcr_oi, pcr_volume, len(display_contracts),
                _wait("Verified CE and PE broker quotes are required; trade is blocked.", now),
            )

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dt.tz_convert("Asia/Kolkata")
        frame = (
            frame.dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
            .reset_index(drop=True)
        )

        expected_minutes = 5 if timeframe == "5m" else 15
        candle_valid, candle_reasons = validate_candle_frame(
            frame, timeframe_minutes=expected_minutes, now=now,
            max_age_minutes=20.0 if timeframe == "5m" else 40.0, minimum_rows=60,
        )
        quality = assess_ohlcv(frame, expected_minutes=expected_minutes)
        if not candle_valid:
            return DecisionSnapshot(
                "KOTAK_NEO", "LIVE", "RED",
                "Broker-captured candle evidence is not decision-ready.",
                now, frame, contracts, "RED",
                tuple(candle_reasons) or tuple(quality.reasons) or ("Captured candle validation failed.",),
                pcr_oi, pcr_volume, len(contracts),
                _wait("Captured candle validation failed; trade is blocked.", now),
            )
        if quality.status == "RED" or len(frame) < 60:
            reason = " | ".join(quality.reasons) or "insufficient completed captured candles"
            return DecisionSnapshot(
                "KOTAK_NEO", "LIVE", "RED",
                f"Real Kotak captured data-quality gate blocked the decision: {reason}",
                now, frame, contracts, quality.status, tuple(quality.reasons),
                pcr_oi, pcr_volume, len(contracts),
                _wait(
                    f"Data-quality gate: {reason}",
                    now, frame, quality.status, tuple(quality.reasons),
                ),
            )

        enriched = add_indicators(frame.copy())
        if pcr_oi is None:
            signal = _wait(
                "Real Kotak option-chain/OI confirmation is unavailable; CE/PE trade is blocked.",
                now, enriched,
            )
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
            "KOTAK_NEO", "LIVE", quality.status,
            f"Kotak Neo option chain + {source} 5m candles are the primary decision source.",
            now, enriched, contracts, quality.status, tuple(quality.reasons),
            pcr_oi, pcr_volume, len(contracts), signal,
        )
    except Exception as exc:
        # Preserve raw chain visibility and diagnostics, but never preserve an
        # executable signal after an unexpected error.
        return DecisionSnapshot(
            "KOTAK_NEO", "LIVE", "RED",
            f"Kotak Neo decision-data error: {exc}",
            now, None, display_contracts, "RED", (str(exc),), pcr_oi, pcr_volume,
            len(display_contracts),
            _wait(f"Kotak Neo decision-data error: {exc}; trade is blocked.", now),
        )
