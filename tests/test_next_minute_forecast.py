from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction.next_minute_forecast import forecast_next_minute


def _candles() -> pd.DataFrame:
    close = np.array([100.0, 100.2, 100.1, 100.4, 100.3, 100.6])
    open_ = np.array([99.9, 100.0, 100.2, 100.1, 100.4, 100.3])
    spread = np.full(len(close), 0.2)
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-10-09 09:15", periods=len(close), freq="min"),
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": np.full(len(close), 1000),
    })


def test_forecast_uses_latest_completed_candle_and_returns_ordered_bounds():
    frame = _candles()
    original = frame.copy(deep=True)
    result = forecast_next_minute(
        frame, generated_at="2026-10-09T09:20:00+05:30", lookback=5, quantile=0.9
    )
    assert result.reference_price == pytest.approx(frame.iloc[-1]["close"])
    assert result.predicted_low <= result.reference_price <= result.predicted_high
    assert result.target_candle_time == (frame.iloc[-1]["timestamp"] + pd.Timedelta(minutes=1)).isoformat()
    assert result.history_bars == 5
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("mutation, message", [
    (lambda d: d.assign(timestamp=["bad"] + list(d["timestamp"].iloc[1:])), "missing or invalid"),
    (lambda d: d.assign(volume=[-1] + list(d["volume"].iloc[1:])), "Volume cannot be negative"),
    (lambda d: d.assign(close=[0] + list(d["close"].iloc[1:])), "prices must be positive"),
    (lambda d: d.assign(high=[1] + list(d["high"].iloc[1:])), "Invalid OHLC row"),
    (lambda d: d.assign(volume=[np.inf] + list(d["volume"].iloc[1:])), "non-finite"),
])
def test_invalid_market_data_is_rejected(mutation, message):
    with pytest.raises(ValueError, match=message):
        forecast_next_minute(mutation(_candles()), lookback=5)


def test_duplicate_timestamps_are_deduplicated_before_forecasting():
    frame = _candles()
    duplicate = pd.concat([frame, frame.iloc[[-1]]], ignore_index=True)
    result = forecast_next_minute(duplicate, lookback=5)
    assert result.history_bars == 5


def test_too_few_unique_candles_are_rejected():
    frame = _candles().iloc[:2]
    with pytest.raises(ValueError, match="At least 3 unique"):
        forecast_next_minute(frame, lookback=5)
