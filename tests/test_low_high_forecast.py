from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction.low_high_forecast import walk_forward_low_high


def _candles(n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.22, n))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.05, 0.45, n)
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


def test_walk_forward_forecast_has_next_candle_targets_and_metrics():
    data = _candles()
    result = walk_forward_low_high(data, min_train=80, estimator="ridge")
    assert result.status == "EVALUATED"
    assert len(result.predictions) == len(data)
    assert result.metrics["forecast_rows_complete"] is True
    assert result.predictions["target_timestamp"].reset_index(drop=True).equals(pd.to_datetime(data["timestamp"], utc=True).reset_index(drop=True))
    assert result.predictions["forecast_method"].notna().all()
    assert result.predictions.iloc[0]["forecast_method"] == "OPEN_FALLBACK_NO_PRIOR_HISTORY"
    assert result.metrics["fallback_forecast_rows"] > 0
    assert result.predictions["origin_timestamp"].isna().iloc[0]
    assert (result.predictions["target_timestamp"].iloc[1:].reset_index(drop=True) > result.predictions["origin_timestamp"].iloc[1:].reset_index(drop=True)).all()
    assert (result.predictions["predicted_low"] <= result.predictions["predicted_high"]).all()
    assert 0 <= result.metrics["full_range_coverage_pct"] <= 100
    assert result.metrics["low_mae"] >= 0
    assert result.metrics["high_mae"] >= 0


def test_walk_forward_forecast_rejects_missing_ohlc_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        walk_forward_low_high(pd.DataFrame({"timestamp": ["2026-10-01"], "close": [100]}))


def test_walk_forward_forecast_rejects_invalid_training_size():
    with pytest.raises(ValueError, match="min_train"):
        walk_forward_low_high(_candles(), min_train=10)


def test_walk_forward_forecast_does_not_require_volume():
    data = _candles().drop(columns=["volume"])
    result = walk_forward_low_high(data, min_train=80, estimator="ridge")
    assert result.status == "EVALUATED"
    assert len(result.predictions) == len(data)
    assert not result.predictions.empty
