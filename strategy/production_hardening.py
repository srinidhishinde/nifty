from __future__ import annotations

"""Fail-closed production controls shared by research, paper and live paths.

This module intentionally contains no broker-order side effects.  It is the
single place for evidence validation so the UI cannot accidentally turn stale,
mixed, synthetic or non-executable data into a trade.
"""

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import math
import os
from typing import Any, Mapping, Sequence

import pandas as pd


IST = "Asia/Kolkata"
BROKER_SOURCE = "KOTAK_NEO"


def as_ist_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize(IST)
    return ts.tz_convert(IST)


def timestamp_age_seconds(value: Any, *, now: Any | None = None) -> float | None:
    ts = as_ist_timestamp(value)
    if ts is None:
        return None
    current = as_ist_timestamp(now or pd.Timestamp.now(tz=IST))
    if current is None:
        return None
    return (current - ts).total_seconds()


def is_fresh(value: Any, *, max_age_seconds: float, now: Any | None = None) -> bool:
    age = timestamp_age_seconds(value, now=now)
    return age is not None and 0.0 <= age <= float(max_age_seconds)


def completed_candle_timestamp(timestamp: Any, *, timeframe_minutes: int, now: Any | None = None) -> bool:
    ts = as_ist_timestamp(timestamp)
    current = as_ist_timestamp(now or pd.Timestamp.now(tz=IST))
    if ts is None or current is None:
        return False
    # Candle timestamps represent the bar open.  The bar is complete only
    # after one full timeframe has elapsed.
    return ts + pd.Timedelta(minutes=int(timeframe_minutes)) <= current


def validate_candle_frame(
    frame: pd.DataFrame | None,
    *,
    timeframe_minutes: int,
    now: Any | None = None,
    max_age_minutes: float | None = None,
    minimum_rows: int = 60,
) -> tuple[bool, tuple[str, ...]]:
    if frame is None or frame.empty:
        return False, ("No captured OHLCV frame.",)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        return False, (f"Missing OHLCV columns: {sorted(missing)}",)
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    reasons: list[str] = []
    if data[list(required)].isna().any().any():
        reasons.append("OHLCV contains null/invalid values.")
    if (data["high"] < data[["open", "close"]].max(axis=1)).any() or (data["low"] > data[["open", "close"]].min(axis=1)).any():
        reasons.append("OHLCV high/low violates candle bounds.")
    if (data["volume"] < 0).any():
        reasons.append("Negative volume is invalid.")
    ts = data["timestamp"].dropna().sort_values()
    if ts.duplicated().any():
        reasons.append("Duplicate candle timestamps.")
    if len(ts) < int(minimum_rows):
        reasons.append(f"Only {len(ts)} candles available; {minimum_rows} required.")
    if not ts.empty:
        latest = ts.iloc[-1]
        if not completed_candle_timestamp(latest, timeframe_minutes=timeframe_minutes, now=now):
            reasons.append("Latest candle is incomplete.")
        if max_age_minutes is not None:
            age = timestamp_age_seconds(latest, now=now)
            if age is None or age < 0 or age > float(max_age_minutes) * 60:
                reasons.append("Latest completed candle is stale.")
    return not reasons, tuple(reasons)


def validate_option_contract(
    contract: Any,
    *,
    now: Any | None = None,
    max_quote_age_seconds: float = 60.0,
    require_broker_quote: bool = True,
    max_spread_pct: float = 0.05,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    source = str(getattr(contract, "quote_source", "") or "").upper()
    quote_ts = getattr(contract, "quote_timestamp", None)
    if require_broker_quote and (source != BROKER_SOURCE or quote_ts is None):
        reasons.append("Quote provenance/timestamp is not broker-verified.")
    if quote_ts is not None and not is_fresh(quote_ts, max_age_seconds=max_quote_age_seconds, now=now):
        reasons.append("Option quote is stale or from the future.")
    def _finite_number(name: str) -> float | None:
        raw = getattr(contract, name, 0)
        try:
            value = float(raw if raw not in (None, "") else 0)
        except (TypeError, ValueError):
            reasons.append(f"Invalid numeric option field: {name}.")
            return None
        if not math.isfinite(value):
            reasons.append(f"Non-finite option field: {name}.")
            return None
        return value

    bid = _finite_number("bid")
    ask = _finite_number("ask")
    ltp = _finite_number("ltp")
    volume = _finite_number("volume")
    oi = _finite_number("open_interest")
    strike = _finite_number("strike")
    side = str(getattr(contract, "option_type", "") or "").upper()
    expiry = str(getattr(contract, "expiry", "") or "").strip()
    if side not in {"CE", "PE"}:
        reasons.append("Invalid option side.")
    if strike is None or strike <= 0:
        reasons.append("Invalid strike.")
    if not expiry:
        reasons.append("Missing expiry.")
    elif not option_expiry_valid(expiry, now=now):
        reasons.append("Option expiry is missing, malformed, or already expired.")
    exchange = str(getattr(contract, "exchange", "") or "").strip().lower()
    underlying = str(getattr(contract, "underlying", "") or "").strip().upper()
    token = str(getattr(contract, "instrument_token", "") or "").strip()
    symbol = str(getattr(contract, "symbol", "") or "").strip()
    if exchange not in {"nse_fo", "mcx_fo", "bse_fo"}:
        reasons.append("Invalid option exchange.")
    if not underlying:
        reasons.append("Missing option underlying.")
    if not token or "|" not in token:
        reasons.append("Missing canonical broker instrument token.")
    if not symbol:
        reasons.append("Missing canonical broker trading symbol.")
    if ltp is None or ltp <= 0:
        reasons.append("Non-positive LTP.")
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        reasons.append("Invalid executable bid/ask.")
    elif ask > 0 and (ask - bid) / ask > float(max_spread_pct):
        reasons.append("Bid/ask spread exceeds configured limit.")
    if volume is None or oi is None or volume <= 0 or oi <= 0:
        reasons.append("Option has no executable volume/OI.")
    return not reasons, tuple(reasons)


def synchronized_timestamps(*values: Any, tolerance_seconds: float = 90.0) -> bool:
    timestamps = [as_ist_timestamp(v) for v in values]
    if not timestamps or any(v is None for v in timestamps):
        return False
    ages = [(v - timestamps[0]).total_seconds() for v in timestamps[1:]]
    return all(abs(x) <= float(tolerance_seconds) for x in ages)



def validate_contract_identity(
    contract: Any,
    *,
    expected_instrument: str | None = None,
    expected_exchange: str | None = None,
    expected_expiry: Any | None = None,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    exchange = str(getattr(contract, "exchange", "") or "").strip().lower()
    underlying = str(getattr(contract, "underlying", "") or "").strip().upper()
    symbol = str(getattr(contract, "symbol", "") or "").strip()
    token = str(getattr(contract, "instrument_token", "") or "").strip()
    if expected_instrument and underlying != str(expected_instrument).strip().upper():
        reasons.append("Option underlying does not match decision instrument.")
    if expected_exchange and exchange != str(expected_exchange).strip().lower():
        reasons.append("Option exchange does not match decision exchange.")
    if expected_expiry and str(getattr(contract, "expiry", "")).strip() != str(expected_expiry).strip():
        reasons.append("Option expiry does not match selected decision expiry.")
    if "|" not in token or not token.split("|", 1)[1]:
        reasons.append("Broker instrument token is not canonical.")
    if not symbol:
        reasons.append("Broker trading symbol is missing.")
    if "|" in token and token.split("|", 1)[0].strip().lower() != exchange:
        reasons.append("Broker token exchange does not match contract exchange.")
    return not reasons, tuple(reasons)

def choose_executable_option(
    contracts: Sequence[Any],
    *,
    spot: float,
    direction: str,
    now: Any | None = None,
    max_quote_age_seconds: float = 60.0,
    max_spread_pct: float = 0.05,
) -> Any | None:
    side = "CE" if str(direction).upper() in {"CE", "BUY CE", "BUY"} else "PE"
    candidates: list[tuple[float, Any]] = []
    for contract in contracts:
        if str(getattr(contract, "option_type", "")).upper() != side:
            continue
        ok, _ = validate_option_contract(
            contract,
            now=now,
            max_quote_age_seconds=max_quote_age_seconds,
            require_broker_quote=True,
            max_spread_pct=max_spread_pct,
        )
        identity_ok, _ = validate_contract_identity(contract)
        if not ok or not identity_ok:
            continue
        bid = float(getattr(contract, "bid", 0) or 0)
        ask = float(getattr(contract, "ask", 0) or 0)
        spread = (ask - bid) / ask if ask > 0 else 1.0
        strike = float(getattr(contract, "strike", 0) or 0)
        distance = abs(strike - float(spot)) / max(abs(float(spot)), 1e-9)
        oi = float(getattr(contract, "open_interest", 0) or 0)
        volume = float(getattr(contract, "volume", 0) or 0)
        # Prefer near-ATM, executable liquidity, then OI/volume.  This avoids
        # the previous "maximum OI wins" bias toward remote strikes.
        score = 100.0 * distance + 25.0 * spread - math.log1p(max(oi, 0.0)) * 0.01 - math.log1p(max(volume, 0.0)) * 0.005
        candidates.append((score, contract))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def decision_id(
    *,
    instrument: str,
    timeframe: str,
    candle_timestamp: Any,
    option_snapshot_timestamp: Any,
    direction: str,
    contract_symbol: str = "",
) -> str:
    raw = "|".join([
        str(instrument).upper(),
        str(timeframe),
        str(as_ist_timestamp(candle_timestamp)),
        str(as_ist_timestamp(option_snapshot_timestamp)),
        str(direction).upper(),
        str(contract_symbol),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class AccountEvidence:
    equity: float | None
    available_margin: float | None = None
    source: str = ""
    timestamp: pd.Timestamp | None = None

    @property
    def valid(self) -> bool:
        return self.equity is not None and math.isfinite(float(self.equity)) and float(self.equity) > 0


def resolve_runtime_account(broker: Any) -> AccountEvidence:
    """Read only the broker adapter's canonical account-state contract.

    We intentionally do not guess among arbitrary method names. A broker
    adapter must expose get_account_state() and return a mapping containing
    equity/available_margin plus an authoritative timestamp.
    """
    if broker is None:
        return AccountEvidence(None, source="NO_BROKER")
    fn = getattr(broker, "get_account_state", None)
    if not callable(fn):
        return AccountEvidence(None, source="BROKER_ACCOUNT_ADAPTER_UNAVAILABLE")
    try:
        raw = fn()
    except Exception:
        return AccountEvidence(None, source="BROKER_ACCOUNT_UNAVAILABLE")
    if not isinstance(raw, Mapping):
        return AccountEvidence(None, source="BROKER_ACCOUNT_SCHEMA_INVALID")
    try:
        equity = float(raw["equity"])
        available_margin = float(raw["available_margin"])
    except (KeyError, TypeError, ValueError):
        return AccountEvidence(None, source="BROKER_ACCOUNT_SCHEMA_INVALID")
    if not math.isfinite(equity) or equity <= 0:
        return AccountEvidence(None, source="BROKER_EQUITY_INVALID")
    if not math.isfinite(available_margin) or available_margin < 0:
        return AccountEvidence(None, source="BROKER_MARGIN_INVALID")
    timestamp = as_ist_timestamp(raw.get("timestamp"))
    if timestamp is None:
        return AccountEvidence(None, source="BROKER_ACCOUNT_TIMESTAMP_MISSING")
    return AccountEvidence(
        equity=equity,
        available_margin=available_margin,
        source="KOTAK_NEO_ACCOUNT",
        timestamp=timestamp,
    )


@dataclass(frozen=True)
class DecisionEvidence:
    instrument: str
    timeframe: str
    source: str
    candle_timestamp: pd.Timestamp | None
    option_snapshot_timestamp: pd.Timestamp | None
    underlying_quote_timestamp: pd.Timestamp | None
    candle_valid: bool
    option_chain_valid: bool
    option_quotes_valid: bool
    liquidity_valid: bool
    synchronized: bool
    regime_valid: bool
    ml_validated: bool
    ensemble_valid: bool
    trade_plan_valid: bool
    account_valid: bool
    risk_valid: bool
    journal_valid: bool
    reconciliation_valid: bool
    tests_valid: bool
    backtest_valid: bool
    contract_valid: bool
    expiry_valid: bool
    clock_valid: bool
    schema_valid: bool
    source_policy_valid: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def mandatory_pass(self) -> bool:
        return all((
            self.source == BROKER_SOURCE,
            self.candle_valid,
            self.option_chain_valid,
            self.option_quotes_valid,
            self.liquidity_valid,
            self.synchronized,
            self.regime_valid,
            self.trade_plan_valid,
            self.account_valid,
            self.risk_valid,
            self.journal_valid,
            self.reconciliation_valid,
            self.tests_valid,
            self.backtest_valid,
            self.contract_valid,
            self.expiry_valid,
            self.clock_valid,
            self.schema_valid,
            self.source_policy_valid,
            self.ml_validated,
            self.ensemble_valid,
        ))


def build_decision_evidence(**kwargs: Any) -> DecisionEvidence:
    return DecisionEvidence(**kwargs)


def live_permission(evidence: DecisionEvidence, *, config_enabled: bool = False, broker_orders_allowed: bool = False) -> bool:
    """Authorize only with complete evidence, explicit enable, and broker lock."""
    return bool(evidence.mandatory_pass and config_enabled is True and broker_orders_allowed is True)


def ml_validation_ok(artifact: Any) -> bool:
    if not artifact:
        return False
    if isinstance(artifact, Mapping):
        required = ("validation", "out_of_sample", "walk_forward", "leakage_audit")
        return all(bool(artifact.get(k)) for k in required)
    return all(bool(getattr(artifact, key, False)) for key in ("validation", "out_of_sample", "walk_forward", "leakage_audit"))


def runtime_clock_ok(
    *,
    max_age_seconds: float = 5.0,
    reference_timestamp: Any | None = None,
    now: Any | None = None,
    broker_now: Any | None = None,
    max_drift_seconds: float = 5.0,
) -> bool:
    """Validate broker timestamp freshness against an explicit decision clock.

    This is deliberately named clock_ok only for compatibility. It does not
    claim to measure NTP drift; it certifies that the broker event timestamp
    is not future-dated and is within the decision freshness budget.
    """
    if reference_timestamp is None:
        return False
    age = timestamp_age_seconds(reference_timestamp, now=now)
    if age is None or age < 0.0 or age > float(max_age_seconds):
        return False
    if broker_now is not None:
        local = as_ist_timestamp(now)
        broker = as_ist_timestamp(broker_now)
        if local is None or broker is None or abs((local - broker).total_seconds()) > float(max_drift_seconds):
            return False
    return True


def option_expiry_valid(
    expiry: Any,
    *,
    now: Any | None = None,
    cutoff_hour_ist: int = 15,
    cutoff_minute_ist: int = 30,
) -> bool:
    if expiry in (None, ""):
        return False
    try:
        exp = pd.Timestamp(expiry)
    except (TypeError, ValueError):
        return False
    current = as_ist_timestamp(now or pd.Timestamp.now(tz=IST))
    if current is None:
        return False
    exp = exp.tz_localize(IST) if exp.tzinfo is None else exp.tz_convert(IST)
    if exp.date() < current.date():
        return False
    if exp.date() > current.date():
        return True
    cutoff = current.normalize() + pd.Timedelta(
        hours=int(cutoff_hour_ist),
        minutes=int(cutoff_minute_ist),
    )
    return current <= cutoff

def safe_risk_quantity(
    *,
    equity: float,
    risk_fraction: float,
    entry: float,
    stop: float,
    lot_size: float,
    max_notional_fraction: float = 0.10,
    available_margin: float | None = None,
    margin_per_lot: float | None = None,
    current_exposure: float = 0.0,
    max_exposure_fraction: float = 0.10,
) -> int:
    if (
        equity <= 0 or risk_fraction <= 0 or entry <= 0 or stop <= 0
        or lot_size <= 0
        or not all(math.isfinite(float(x)) for x in (equity, risk_fraction, entry, stop, lot_size))
        or stop >= entry
    ):
        return 0
    risk_per_unit = abs(entry - stop) * lot_size
    if risk_per_unit <= 0:
        return 0
    by_risk = math.floor((equity * risk_fraction) / risk_per_unit)
    by_notional = math.floor((equity * max_notional_fraction) / (entry * lot_size))
    return max(0, min(by_risk, by_notional))


def source_policy_ok(*, environment: str, source: str, synthetic: bool = False, yahoo: bool = False) -> bool:
    """Permit only explicitly labelled sources for each runtime environment."""
    env = str(environment).upper()
    if synthetic or yahoo:
        return env == "BACKTEST"
    if env in {"LIVE", "PAPER", "UAT", "RESEARCH"}:
        return str(source or "").upper() == BROKER_SOURCE
    return env == "BACKTEST" or str(source or "").upper() == BROKER_SOURCE


def idempotency_key(record: Mapping[str, Any]) -> str:
    if record.get("decision_id"):
        return str(record["decision_id"])
    return decision_id(
        instrument=record.get("instrument", ""),
        timeframe=record.get("timeframe", ""),
        candle_timestamp=record.get("candle_timestamp") or record.get("recorded_at"),
        option_snapshot_timestamp=record.get("option_snapshot_timestamp") or record.get("recorded_at"),
        direction=record.get("direction", "WAIT"),
        contract_symbol=record.get("contract_symbol", ""),
    )
