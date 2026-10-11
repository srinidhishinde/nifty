from datetime import datetime, timezone, timedelta

import pandas as pd

from reliability.failure_guard import assess_health, append_incident, validate_ohlcv


def candles(rows=5):
    ts = pd.date_range("2026-10-11T09:15:00Z", periods=rows, freq="min")
    return pd.DataFrame({
        "timestamp": ts,
        "open": [100.0] * rows,
        "high": [101.0] * rows,
        "low": [99.0] * rows,
        "close": [100.5] * rows,
        "volume": [1000] * rows,
    })


def test_valid_ohlcv_has_no_findings():
    assert validate_ohlcv(candles()) == []


def test_missing_columns_block():
    report = assess_health(
        now=datetime(2026, 10, 11, 9, 20, tzinfo=timezone.utc),
        broker_connected=True, execution_configured=True,
        market_data_timestamp="2026-10-11T09:19:00Z",
        ohlcv=pd.DataFrame({"close": [100]}),
        forecast_accuracy_pct=90, forecast_samples=600, baseline_accuracy_pct=50,
    )
    assert report.status == "BLOCKED"
    assert not report.live_orders_allowed
    assert any(f.code == "OHLCV_COLUMNS_MISSING" for f in report.findings)


def test_stale_feed_and_missing_forecast_evidence_fail_closed():
    report = assess_health(
        now=datetime(2026, 10, 11, 10, 0, tzinfo=timezone.utc),
        broker_connected=True, execution_configured=True,
        market_data_timestamp="2026-10-11T09:00:00Z",
        ohlcv=candles(),
    )
    codes = {f.code for f in report.findings}
    assert "FEED_STALE" in codes
    assert "FORECAST_EVIDENCE_MISSING" in codes
    assert not report.live_orders_allowed


def test_below_gate_or_not_above_baseline_blocks():
    base = dict(
        now=datetime(2026, 10, 11, 9, 20, tzinfo=timezone.utc),
        broker_connected=True, execution_configured=True,
        market_data_timestamp="2026-10-11T09:19:30Z",
        ohlcv=candles(),
        forecast_samples=600, baseline_accuracy_pct=50,
    )
    low = assess_health(**base, forecast_accuracy_pct=79.9)
    assert not low.live_orders_allowed
    assert "FORECAST_ACCURACY_BELOW_GATE" in {f.code for f in low.findings}
    tie = assess_health(**base, forecast_accuracy_pct=50)
    assert not tie.live_orders_allowed
    assert "FORECAST_NOT_ABOVE_BASELINE" in {f.code for f in tie.findings}


def test_healthy_report_requires_all_evidence():
    report = assess_health(
        now=datetime(2026, 10, 11, 9, 20, tzinfo=timezone.utc),
        broker_connected=True, execution_configured=True,
        market_data_timestamp="2026-10-11T09:19:30Z",
        ohlcv=candles(),
        forecast_accuracy_pct=85, forecast_samples=600, baseline_accuracy_pct=50,
    )
    assert report.status == "HEALTHY"
    assert report.live_orders_allowed


def test_incident_log_is_append_only_jsonl(tmp_path):
    report = assess_health()
    path = tmp_path / "incidents" / "health.jsonl"
    append_incident(path, report, context={"environment": "PAPER"})
    append_incident(path, report, context={"environment": "PAPER"})
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
