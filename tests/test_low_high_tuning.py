from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction.low_high_tuning import tune_and_compare


def _candles(n: int = 145) -> pd.DataFrame:
    rng = np.random.default_rng(91)
    close = 100.0 + np.cumsum(rng.normal(0, 0.25, n))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.08, 0.5, n)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread * rng.uniform(0.6, 1.2, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-10-01 09:15", periods=n, freq="min"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(100, 5000, n),
    })


def test_adaptive_historical_quantile_tracks_recent_regime():
    from prediction.low_high_tuning import _adaptive_historical_quantile

    calm = np.full(80, 0.2)
    shifted = np.r_[calm, np.full(40, 1.0)]
    assert _adaptive_historical_quantile(shifted, 0.8) > _adaptive_historical_quantile(calm, 0.8)
    assert _adaptive_historical_quantile(shifted, 0.9) >= _adaptive_historical_quantile(shifted, 0.8)
    assert _adaptive_historical_quantile(calm, 0.8) == pytest.approx(0.2)


def test_tuning_compares_model_and_simple_baselines():
    predictions, summary = tune_and_compare(_candles(), min_train=80, estimator="ridge")
    assert len(predictions) == 145
    assert summary["evaluated_forecasts"] == 144
    assert summary["forecast_timing"] == "before_target_candle_open"
    assert summary["first_candle"] == "opening_forecast_reported_separately_and_excluded_from_preopen_metrics"
    assert predictions.loc[0, "opening_forecast_method"] == "OPENING_CANDLE_OPEN_ANCHORED_FALLBACK"
    assert pd.notna(predictions.loc[0, "opening_predicted_low"])
    assert pd.isna(predictions.loc[0, "model_low"])
    assert set(summary["methods"]) == {
        "model", "online_conformal_model", "online_asymmetric_conformal_model", "rolling_median_range",
        "historical_excursion_quantile", "historical_asymmetric_quantile",
    }
    assert predictions.loc[0, "model_low"] != predictions.loc[0, "model_low"]
    point = summary["point_forecast"]
    assert point["rows"] == 144
    assert point["low_mae"] >= 0
    assert point["high_mae"] >= 0
    assert point["close_mae"] >= 0
    assert 0 <= point["directional_accuracy_pct"] <= 100
    assert point["low_p90_absolute_error"] >= point["low_median_absolute_error"]
    assert point["high_p90_absolute_error"] >= point["high_median_absolute_error"]
    assert "full_range_coverage_pct" not in point
    for method in summary["methods"].values():
        if method["rows"]:
            assert 0 <= method["full_range_coverage_pct"] <= 100
            assert method["mean_predicted_width"] >= 0


def test_online_calibration_does_not_use_future_candles():
    data = _candles(145)
    first, _ = tune_and_compare(data, min_train=80, estimator="ridge")
    changed = data.copy()
    changed.loc[120:, ["open", "high", "low", "close"]] *= 1.07
    second, _ = tune_and_compare(changed, min_train=80, estimator="ridge")
    # The prediction for candle 110 is generated using only data available
    # before candle 110; later candles cannot alter its forecast or calibration.
    for col in ("model_low", "model_high", "conformal_low", "conformal_high",
                "asymmetric_conformal_low", "asymmetric_conformal_high",
                "historical_quantile_low", "historical_quantile_high",
                "historical_asymmetric_quantile_low", "historical_asymmetric_quantile_high",
                "point_predicted_low", "point_predicted_high", "predicted_close", "predicted_direction"):
        assert first.loc[110, col] == pytest.approx(second.loc[110, col], nan_ok=True)


def test_online_calibration_excludes_fallback_forecasts_and_reports_maturity():
    predictions, _ = tune_and_compare(_candles(145), min_train=80, estimator="ridge")
    for i in range(1, len(predictions)):
        prior = predictions.iloc[1:i]
        expected = int(prior["forecast_method"].astype(str).str.contains("MODEL_", regex=False).sum())
        assert int(predictions.loc[i, "calibration_model_rows"]) == expected

    fallback_rows = predictions["forecast_method"].eq("HISTORICAL_RANGE_FALLBACK")
    assert fallback_rows.any()
    # Fallback forecasts are never counted as calibration examples.
    assert predictions.loc[fallback_rows, "calibration_model_rows"].ge(0).all()


def test_calibration_warmup_uses_completed_candle_excursions_until_mature():
    predictions, _ = tune_and_compare(_candles(145), min_train=80, estimator="ridge")
    model_targets = predictions.index[
        predictions["forecast_method"].astype(str).str.contains("MODEL_", regex=False)
    ]
    assert len(model_targets) > 20

    first_model_target = int(model_targets[0])
    first = predictions.loc[first_model_target]
    assert first["calibration_model_rows"] == 0
    assert first["calibration_source"] == "HISTORICAL_EXCURSION_WARMUP"
    assert first["asymmetric_calibration_source"] == "HISTORICAL_EXCURSION_WARMUP"

    # Calibration switches only after 20 earlier, matured model forecasts.
    matured = predictions.loc[
        predictions["calibration_model_rows"] >= 20
    ]
    assert not matured.empty
    assert matured["calibration_source"].eq("ONLINE_CONFORMAL_MODEL_RESIDUALS").all()
    assert matured["asymmetric_calibration_source"].eq("ONLINE_ASYMMETRIC_MODEL_RESIDUALS").all()


def test_calibration_warmup_does_not_use_future_candles():
    data = _candles(145)
    first, _ = tune_and_compare(data, min_train=80, estimator="ridge")
    changed = data.copy()
    changed.loc[120:, ["open", "high", "low", "close"]] *= 1.07
    second, _ = tune_and_compare(changed, min_train=80, estimator="ridge")
    cols = [
        "conformal_low", "conformal_high", "asymmetric_conformal_low",
        "asymmetric_conformal_high", "calibration_source",
        "asymmetric_calibration_source", "calibration_model_rows",
    ]
    for col in cols:
        if col in {"calibration_source", "asymmetric_calibration_source"}:
            assert first.loc[110, col] == second.loc[110, col]
        else:
            assert first.loc[110, col] == pytest.approx(second.loc[110, col], nan_ok=True)


def test_invalid_coverage_is_rejected():
    with pytest.raises(ValueError, match="target_coverage"):
        tune_and_compare(_candles(), target_coverage=1.0, estimator="ridge")
