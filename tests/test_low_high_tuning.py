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


def test_tuning_compares_model_and_simple_baselines():
    predictions, summary = tune_and_compare(_candles(), min_train=80, estimator="ridge")
    assert len(predictions) == 145
    assert summary["evaluated_forecasts"] == 144
    assert summary["forecast_timing"] == "before_target_candle_open"
    assert summary["first_candle"] == "excluded_no_prior_candle_in_input"
    assert set(summary["methods"]) == {
        "model", "online_conformal_model", "online_asymmetric_conformal_model", "rolling_median_range",
        "historical_excursion_quantile", "optimized_asymmetric_quantile",
    }
    assert predictions.loc[0, "model_low"] != predictions.loc[0, "model_low"]
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
                "optimized_quantile_low", "optimized_quantile_high"):
        assert first.loc[110, col] == pytest.approx(second.loc[110, col], nan_ok=True)


def test_invalid_coverage_is_rejected():
    with pytest.raises(ValueError, match="target_coverage"):
        tune_and_compare(_candles(), target_coverage=1.0, estimator="ridge")
