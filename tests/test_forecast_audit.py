import pandas as pd

from prediction.forecast_audit import (
    audit_summary,
    record_forecast,
    save_walk_forward_evaluation,
    settle_pending_forecasts,
)


def _record_live(path):
    return record_forecast(
        instrument="NIFTY",
        contract_symbol="NIFTY_22500CE",
        expiry="2026-10-15",
        option_type="CE",
        strike=22500,
        timeframe_minutes=1,
        data_source="Kotak Neo",
        issued_at=pd.Timestamp("2026-10-13 09:15:30", tz="Asia/Kolkata"),
        input_candle_time=pd.Timestamp("2026-10-13 09:15", tz="Asia/Kolkata"),
        target_candle_time=pd.Timestamp("2026-10-13 09:16", tz="Asia/Kolkata"),
        predicted_low=90,
        predicted_high=110,
        path=path,
    )


def test_live_forecast_is_persisted_once(tmp_path):
    path = tmp_path / "audit.csv"

    assert _record_live(path) is True
    assert _record_live(path) is False

    audit = pd.read_csv(path)
    assert len(audit) == 1
    assert audit.loc[0, "record_type"] == "LIVE"
    assert audit.loc[0, "status"] == "PENDING"
    assert pd.isna(audit.loc[0, "actual_low"])


def test_live_forecast_only_settles_after_target_candle_closes(tmp_path):
    path = tmp_path / "audit.csv"
    _record_live(path)
    candles = pd.DataFrame([{
        "timestamp": pd.Timestamp("2026-10-13 09:16", tz="Asia/Kolkata"),
        "low": 95.0,
        "high": 105.0,
    }])

    assert settle_pending_forecasts(
        candles,
        contract_symbol="NIFTY_22500CE",
        timeframe_minutes=1,
        as_of=pd.Timestamp("2026-10-13 09:16:30", tz="Asia/Kolkata"),
        path=path,
    ) == 0

    assert settle_pending_forecasts(
        candles,
        contract_symbol="NIFTY_22500CE",
        timeframe_minutes=1,
        as_of=pd.Timestamp("2026-10-13 09:17", tz="Asia/Kolkata"),
        path=path,
    ) == 1

    audit = pd.read_csv(path)
    assert audit.loc[0, "status"] == "VERIFIED"
    assert bool(audit.loc[0, "range_covered"]) is True
    assert float(audit.loc[0, "low_abs_error"]) == 5.0
    assert float(audit.loc[0, "high_abs_error"]) == 5.0


def test_walk_forward_rows_are_idempotent_and_summary_reports_errors(tmp_path):
    path = tmp_path / "audit.csv"
    comparison = pd.DataFrame([{
        "Input candle time": "2026-10-13 09:15",
        "Candle time": "2026-10-13 09:16",
        "Predicted low": 90.0,
        "Actual low": 95.0,
        "Predicted high": 110.0,
        "Actual high": 105.0,
    }])
    kwargs = dict(
        instrument="NIFTY",
        contract_symbol="NIFTY_22500CE",
        expiry="2026-10-15",
        option_type="CE",
        strike=22500,
        timeframe_minutes=1,
        data_source="saved-test.csv",
        path=path,
    )

    assert save_walk_forward_evaluation(comparison, **kwargs) == 1
    assert save_walk_forward_evaluation(comparison, **kwargs) == 0

    summary = audit_summary(
        path, contract_symbol="NIFTY_22500CE", timeframe_minutes=1
    )
    assert summary["verified_count"] == 1
    assert summary["historical_count"] == 1
    assert summary["low_mae"] == 5.0
    assert summary["high_mae"] == 5.0
    assert summary["range_coverage_pct"] == 100.0
