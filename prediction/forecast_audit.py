"""Persistent audit log for walk-forward and subsequently verified forecasts.

This module records forecasts separately from outcomes so an unclosed target candle
cannot be counted as a success. The CSV is local runtime data, not source-controlled.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_PATH = ROOT / "data" / "logs" / "option_forecast_audit.csv"
COLUMNS = [
    "forecast_id", "record_type", "status", "instrument", "contract_symbol",
    "expiry", "option_type", "strike", "timeframe_minutes", "data_source",
    "issued_at", "input_candle_time", "target_candle_time", "predicted_low",
    "predicted_high", "actual_low", "actual_high", "low_abs_error",
    "high_abs_error", "low_bound_ok", "high_bound_ok", "range_covered",
]


def _local_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("Asia/Kolkata")
    return ts.tz_convert("Asia/Kolkata")


def _timestamp_text(value: Any) -> str:
    return _local_timestamp(value).isoformat()


def _timestamp_key(value: Any) -> str:
    return _local_timestamp(value).strftime("%Y-%m-%dT%H:%M:%S%z")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(path)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[COLUMNS]


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.reindex(columns=COLUMNS)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _make_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _result_fields(predicted_low: float, predicted_high: float,
                   actual_low: float, actual_high: float) -> dict[str, Any]:
    low_ok = actual_low >= predicted_low
    high_ok = actual_high <= predicted_high
    return {
        "actual_low": actual_low,
        "actual_high": actual_high,
        "low_abs_error": abs(actual_low - predicted_low),
        "high_abs_error": abs(actual_high - predicted_high),
        "low_bound_ok": low_ok,
        "high_bound_ok": high_ok,
        "range_covered": low_ok and high_ok,
        "status": "VERIFIED",
    }


def record_forecast(
    *,
    instrument: str,
    contract_symbol: str,
    expiry: str,
    option_type: str,
    strike: float,
    timeframe_minutes: int,
    data_source: str,
    issued_at: Any,
    input_candle_time: Any,
    target_candle_time: Any,
    predicted_low: float,
    predicted_high: float,
    path: str | Path | None = None,
) -> bool:
    """Persist one pending forecast; return False for an already-logged forecast."""
    destination = Path(path) if path is not None else DEFAULT_AUDIT_PATH
    input_time = _timestamp_text(input_candle_time)
    target_time = _timestamp_text(target_candle_time)
    forecast_id = _make_id(
        "LIVE", contract_symbol, timeframe_minutes, input_time, target_time
    )
    frame = _read(destination)
    if forecast_id in set(frame["forecast_id"].dropna().astype(str)):
        return False
    row = {
        "forecast_id": forecast_id,
        "record_type": "LIVE",
        "status": "PENDING",
        "instrument": instrument,
        "contract_symbol": contract_symbol,
        "expiry": expiry,
        "option_type": option_type,
        "strike": strike,
        "timeframe_minutes": int(timeframe_minutes),
        "data_source": data_source,
        "issued_at": _timestamp_text(issued_at),
        "input_candle_time": input_time,
        "target_candle_time": target_time,
        "predicted_low": float(predicted_low),
        "predicted_high": float(predicted_high),
    }
    _write(pd.concat([frame, pd.DataFrame([row])], ignore_index=True), destination)
    return True


def settle_pending_forecasts(
    frame: pd.DataFrame,
    *,
    contract_symbol: str,
    timeframe_minutes: int,
    as_of: Any,
    path: str | Path | None = None,
) -> int:
    """Verify pending live forecasts only after the target candle has closed."""
    destination = Path(path) if path is not None else DEFAULT_AUDIT_PATH
    if frame.empty or not {"timestamp", "high", "low"}.issubset(frame.columns):
        return 0
    audit = _read(destination)
    if audit.empty:
        return 0

    candles = frame.copy()
    candles["_time_key"] = candles["timestamp"].map(_timestamp_key)
    candles = candles.drop_duplicates("_time_key", keep="last").set_index("_time_key")
    now = _local_timestamp(as_of)
    settled = 0

    for index, row in audit.iterrows():
        if str(row.get("record_type")) != "LIVE" or str(row.get("status")) != "PENDING":
            continue
        if str(row.get("contract_symbol")) != contract_symbol:
            continue
        try:
            if int(float(row.get("timeframe_minutes"))) != int(timeframe_minutes):
                continue
            target = _local_timestamp(row["target_candle_time"])
            if target + pd.Timedelta(minutes=int(timeframe_minutes)) > now:
                continue
            key = _timestamp_key(target)
            if key not in candles.index:
                continue
            actual = candles.loc[key]
            if isinstance(actual, pd.DataFrame):
                actual = actual.iloc[-1]
            predicted_low = float(row["predicted_low"])
            predicted_high = float(row["predicted_high"])
            actual_low = float(actual["low"])
            actual_high = float(actual["high"])
            if not all(pd.notna(v) for v in (predicted_low, predicted_high, actual_low, actual_high)):
                continue
            # CSV reloads infer all-empty outcome columns as float64. Use object
            # columns before assigning booleans/status strings to remain compatible
            # with newer pandas versions that reject incompatible dtype assignment.
            for column in (
                "actual_low", "actual_high", "low_abs_error", "high_abs_error",
                "low_bound_ok", "high_bound_ok", "range_covered", "status",
            ):
                audit[column] = audit[column].astype(object)
            for column, value in _result_fields(
                predicted_low, predicted_high, actual_low, actual_high
            ).items():
                audit.at[index, column] = value
            settled += 1
        except (TypeError, ValueError, KeyError, OverflowError):
            continue

    if settled:
        _write(audit, destination)
    return settled


def save_walk_forward_evaluation(
    comparison: pd.DataFrame,
    *,
    instrument: str,
    contract_symbol: str,
    expiry: str,
    option_type: str,
    strike: float,
    timeframe_minutes: int,
    data_source: str,
    path: str | Path | None = None,
) -> int:
    """Persist verified historical walk-forward rows idempotently."""
    if comparison.empty:
        return 0
    destination = Path(path) if path is not None else DEFAULT_AUDIT_PATH
    audit = _read(destination)
    known = set(audit["forecast_id"].dropna().astype(str))
    new_rows = []
    for _, result in comparison.iterrows():
        try:
            target_time = _timestamp_text(result["Candle time"])
            input_time = _timestamp_text(result["Input candle time"])
            predicted_low = float(result["Predicted low"])
            predicted_high = float(result["Predicted high"])
            actual_low = float(result["Actual low"])
            actual_high = float(result["Actual high"])
        except (KeyError, TypeError, ValueError):
            continue
        forecast_id = _make_id(
            "WALK_FORWARD", data_source, contract_symbol, timeframe_minutes,
            input_time, target_time,
        )
        if forecast_id in known:
            continue
        row = {
            "forecast_id": forecast_id,
            "record_type": "WALK_FORWARD",
            "instrument": instrument,
            "contract_symbol": contract_symbol,
            "expiry": expiry,
            "option_type": option_type,
            "strike": strike,
            "timeframe_minutes": int(timeframe_minutes),
            "data_source": data_source,
            "issued_at": input_time,
            "input_candle_time": input_time,
            "target_candle_time": target_time,
            "predicted_low": predicted_low,
            "predicted_high": predicted_high,
        }
        row.update(_result_fields(predicted_low, predicted_high, actual_low, actual_high))
        new_rows.append(row)
        known.add(forecast_id)
    if new_rows:
        _write(pd.concat([audit, pd.DataFrame(new_rows)], ignore_index=True), destination)
    return len(new_rows)


def audit_summary(
    path: str | Path | None = None,
    *,
    contract_symbol: str | None = None,
    timeframe_minutes: int | None = None,
) -> dict[str, Any]:
    """Return verified sample counts, boundary errors and range coverage."""
    destination = Path(path) if path is not None else DEFAULT_AUDIT_PATH
    audit = _read(destination)
    if contract_symbol is not None:
        audit = audit[audit["contract_symbol"].astype(str) == contract_symbol]
    if timeframe_minutes is not None:
        numeric_tf = pd.to_numeric(audit["timeframe_minutes"], errors="coerce")
        audit = audit[numeric_tf == int(timeframe_minutes)]
    verified = audit[audit["status"].astype(str) == "VERIFIED"].copy()
    if verified.empty:
        return {
            "verified_count": 0,
            "pending_count": int((audit["status"].astype(str) == "PENDING").sum()),
            "low_mae": None,
            "high_mae": None,
            "range_coverage_pct": None,
            "historical_count": 0,
            "live_count": 0,
        }
    low_errors = pd.to_numeric(verified["low_abs_error"], errors="coerce").dropna()
    high_errors = pd.to_numeric(verified["high_abs_error"], errors="coerce").dropna()
    covered = verified["range_covered"].astype(str).str.lower().isin({"true", "1", "1.0"})
    record_types = verified["record_type"].astype(str)
    return {
        "verified_count": len(verified),
        "pending_count": int((audit["status"].astype(str) == "PENDING").sum()),
        "low_mae": float(low_errors.mean()) if not low_errors.empty else None,
        "high_mae": float(high_errors.mean()) if not high_errors.empty else None,
        "range_coverage_pct": float(covered.mean() * 100),
        "historical_count": int((record_types == "WALK_FORWARD").sum()),
        "live_count": int((record_types == "LIVE").sum()),
    }
