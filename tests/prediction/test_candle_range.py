import pandas as pd
import pytest

from prediction.candle_range import walk_forward_range_forecast


def _candles(n=40):
    opens = [100.0 + i * 0.1 for i in range(n)]
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 09:15", periods=n, freq="min"),
        "open": opens,
        "high": [x + (1.0 if i % 2 else 0.6) for i, x in enumerate(opens)],
        "low": [x - (0.8 if i % 3 else 0.5) for i, x in enumerate(opens)],
    })


def test_width_is_calibrated_to_prior_mean_range():
    data = _candles()
    result = walk_forward_range_forecast(data, window=10, min_periods=5, target_width_ratio=1.0)
    assert not result.forecasts.empty
    row = result.forecasts.iloc[0]
    prior = data.iloc[0:5]
    assert row["forecast_width"] == pytest.approx((prior["high"] - prior["low"]).mean())
    assert result.mean_width_ratio > 0


def test_target_high_low_do_not_change_its_forecast():
    data = _candles()
    baseline = walk_forward_range_forecast(data, window=10, min_periods=5)
    changed = data.copy()
    target_index = 20
    changed.loc[target_index, "high"] += 100
    changed.loc[target_index, "low"] -= 100
    changed_result = walk_forward_range_forecast(changed, window=10, min_periods=5)
    target_time = data.loc[target_index, "timestamp"]
    base_row = baseline.forecasts.loc[baseline.forecasts["timestamp"] == target_time].iloc[0]
    changed_row = changed_result.forecasts.loc[changed_result.forecasts["timestamp"] == target_time].iloc[0]
    assert base_row["predicted_low"] == pytest.approx(changed_row["predicted_low"])
    assert base_row["predicted_high"] == pytest.approx(changed_row["predicted_high"])


def test_invalid_parameters_fail():
    with pytest.raises(ValueError):
        walk_forward_range_forecast(_candles(), window=4, min_periods=5)
    with pytest.raises(ValueError):
        walk_forward_range_forecast(_candles(), target_width_ratio=0)
