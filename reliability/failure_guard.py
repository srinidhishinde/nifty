"""Fail-closed health checks for market data, forecasts, and broker execution.

This module is deliberately deterministic: it does not pretend to predict a
failure from insufficient evidence. Missing evidence is reported as UNKNOWN and
prevents live order permission. It never places orders or connects to a broker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str
    action: str


@dataclass(frozen=True)
class HealthReport:
    status: str
    score: int
    live_orders_allowed: bool
    checks_passed: int
    checks_total: int
    checked_at_utc: str
    findings: tuple[Finding, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["findings"] = [asdict(f) for f in self.findings]
        return result


def validate_ohlcv(data: Any, *, max_gap_seconds: int = 180) -> list[Finding]:
    """Validate an OHLCV DataFrame without silently repairing bad input."""
    findings: list[Finding] = []
    if not isinstance(data, pd.DataFrame) or data.empty:
        return [Finding("OHLCV_MISSING", "BLOCK", "No candle DataFrame was supplied.", "Wait for a real broker candle; do not substitute synthetic/Yahoo data in the live path.")]
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(data.columns)
    if missing:
        return [Finding("OHLCV_COLUMNS_MISSING", "BLOCK", f"Missing required OHLCV columns: {sorted(missing)}.", "Repair the source schema before calculating indicators or signals.")]
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[["timestamp", "open", "high", "low", "close", "volume"]].isna().any().any():
        findings.append(Finding("OHLCV_NON_NUMERIC", "BLOCK", "Candles contain missing, invalid, or non-numeric values.", "Reject malformed rows at ingestion and record the rejected-row count."))
    if frame["timestamp"].isna().any():
        return findings
    if frame["timestamp"].duplicated().any():
        findings.append(Finding("OHLCV_DUPLICATE_TIME", "BLOCK", "Duplicate candle timestamps detected.", "Deduplicate at ingestion only with a documented source policy; do not score duplicate candles."))
    if not frame["timestamp"].is_monotonic_increasing:
        findings.append(Finding("OHLCV_OUT_OF_ORDER", "BLOCK", "Candle timestamps are not chronological.", "Sort and revalidate the data before feature calculation."))
    valid = (
        (frame["low"] > 0) & (frame["open"] > 0) & (frame["close"] > 0)
        & (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
        & (frame["volume"] >= 0)
    )
    if not bool(valid.fillna(False).all()):
        findings.append(Finding("OHLCV_IMPOSSIBLE_CANDLE", "BLOCK", "Impossible OHLC relationship, non-positive price, or negative volume detected.", "Reject the candle and quarantine the feed until a valid candle arrives."))
    clean_ts = frame["timestamp"].dropna().sort_values()
    if len(clean_ts) >= 3:
        gaps = clean_ts.diff().dt.total_seconds().dropna()
        if bool((gaps > max_gap_seconds).any()):
            findings.append(Finding("OHLCV_CADENCE_GAP", "WARN", f"One or more candle gaps exceed {max_gap_seconds} seconds.", "Check market session boundaries and broker data completeness before backtesting or execution."))
    return findings


def assess_health(
    *,
    now: datetime | None = None,
    broker_connected: bool = False,
    execution_configured: bool = False,
    market_data_timestamp: Any = None,
    ohlcv: Any = None,
    forecast_accuracy_pct: float | None = None,
    forecast_samples: int = 0,
    baseline_accuracy_pct: float | None = None,
    min_accuracy_pct: float = 80.0,
    min_forecast_samples: int = 500,
    max_feed_age_seconds: int = 90,
    risk_flags: Iterable[str] = (),
) -> HealthReport:
    """Return deterministic findings; missing proof fails closed for live orders."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    findings: list[Finding] = []
    passed = 0
    risk_flags = tuple(risk_flags)

    def check(ok: bool, code: str, message: str, action: str, severity: str = "BLOCK") -> None:
        nonlocal passed
        if ok:
            passed += 1
        else:
            findings.append(Finding(code, severity, message, action))

    check(broker_connected, "BROKER_DISCONNECTED", "Kotak Neo is not connected.", "Reconnect and verify broker session status; never infer a successful login.")
    check(execution_configured, "EXECUTION_LOCKED", "Live execution is not explicitly enabled by all required configuration gates.", "Keep PAPER/UAT active until independent broker, risk, and operator checks pass.")

    timestamp = pd.to_datetime(market_data_timestamp, errors="coerce", utc=True)
    if pd.isna(timestamp):
        check(False, "FEED_TIMESTAMP_UNKNOWN", "No valid timestamp proves market-data freshness.", "Obtain a timestamped Kotak Neo quote/candle; do not silently fall back to Yahoo or synthetic data.")
    else:
        age = (pd.Timestamp(now).tz_convert("UTC") - timestamp).total_seconds()
        check(0 <= age <= max_feed_age_seconds, "FEED_STALE", f"Market data age is {age:.1f}s; allowed range is 0–{max_feed_age_seconds}s.", "Pause order execution and refresh the broker feed.")
    candle_findings = validate_ohlcv(ohlcv)
    findings.extend(candle_findings)
    passed += 1 if not candle_findings else 0

    metrics_valid = (
        forecast_accuracy_pct is not None
        and 0 <= float(forecast_accuracy_pct) <= 100
        and forecast_samples >= min_forecast_samples
        and baseline_accuracy_pct is not None
        and 0 <= float(baseline_accuracy_pct) <= 100
    )
    check(metrics_valid, "FORECAST_EVIDENCE_MISSING", f"Forecast evidence needs accuracy, a baseline, and at least {min_forecast_samples} scored unseen candles.", "Run the frozen chronological walk-forward benchmark on versioned real sessions.")
    if metrics_valid:
        check(float(forecast_accuracy_pct) >= min_accuracy_pct, "FORECAST_ACCURACY_BELOW_GATE", f"Forecast accuracy {float(forecast_accuracy_pct):.2f}% is below the {min_accuracy_pct:.2f}% gate.", "Do not tune on the test set; improve only using training data, then evaluate on new unseen sessions.")
        check(float(forecast_accuracy_pct) > float(baseline_accuracy_pct), "FORECAST_NOT_ABOVE_BASELINE", f"Forecast accuracy does not exceed the {float(baseline_accuracy_pct):.2f}% baseline.", "Require reproducible out-of-sample improvement over a predeclared baseline.")

    for flag in risk_flags:
        findings.append(Finding("RISK_FLAG", "BLOCK", str(flag), "Resolve the risk-control alert and rerun health checks before any order."))

    checks_total = 5 + (2 if metrics_valid else 0) + len(risk_flags)
    # Every block finding denies live orders; warnings are visible but not fatal.
    blocking = any(f.severity == "BLOCK" for f in findings)
    status = "BLOCKED" if blocking else ("DEGRADED" if findings else "HEALTHY")
    score = round(100 * passed / max(checks_total, 1))
    return HealthReport(
        status, score, not blocking, passed, checks_total,
        now.astimezone(timezone.utc).isoformat(), tuple(findings)
    )


def append_incident(path: str | Path, report: HealthReport, *, context: dict[str, Any] | None = None) -> None:
    """Append one sanitized JSONL incident record; caller controls path and context."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "report": report.to_dict(),
        "context": context or {},
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, allow_nan=False, default=str) + "\n")
