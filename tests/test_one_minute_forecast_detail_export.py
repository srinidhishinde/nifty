import pandas as pd

from app.one_minute_forecast_panel import _comparison_detail
from tests.test_one_minute_forecast_comparison import sample_candles


def test_detail_export_has_run_metadata_and_paired_predictions():
    detail = _comparison_detail(sample_candles(20), quantile=0.9, lookback=10)

    assert len(detail) == 17
    assert detail["input_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert detail["model_version"].eq("rolling-excursion-v1").all()
    assert detail["quantile"].eq(0.9).all()
    assert detail["lookback"].eq(10).all()
    assert detail["forecast_count"].eq(17).all()
    for column in (
        "adaptive_predicted_low",
        "adaptive_predicted_high",
        "baseline_predicted_low",
        "baseline_predicted_high",
        "actual_low",
        "actual_high",
    ):
        assert column in detail.columns
    assert detail["timestamp"].is_monotonic_increasing


def test_detail_export_fingerprint_changes_when_input_changes():
    candles = sample_candles(12)
    first = _comparison_detail(candles, quantile=0.75, lookback=10)
    changed = candles.copy()
    changed.loc[0, "close"] += 0.01
    second = _comparison_detail(changed, quantile=0.75, lookback=10)

    assert first["input_sha256"].iloc[0] != second["input_sha256"].iloc[0]
    assert first["generated_at_utc"].iloc[0]
    assert second["generated_at_utc"].iloc[0]