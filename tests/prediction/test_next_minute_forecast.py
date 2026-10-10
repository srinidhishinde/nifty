import pandas as pd
import pytest

from prediction.next_minute_forecast import forecast_next_minute


def sample_candles(n=10):
    ts = pd.date_range("2026-10-09 09:15", periods=n, freq="min")
    close = [100 + i * 0.2 for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts,
        "open": [c - 0.1 for c in close],
        "high": [c + 0.5 for c in close],
        "low": [c - 0.4 for c in close],
        "close": close,
        "volume": [1000] * n,
    })


def test_forecasts_next_candle_not_latest_completed_candle():
    candles = sample_candles()
    forecast = forecast_next_minute(candles, generated_at="2026-10-09T09:24:59")
    assert forecast.target_candle_time.startswith("2026-10-09T09:25:00")
    assert forecast.predicted_low < forecast.reference_price < forecast.predicted_high
    assert forecast.direction == "UP"


def test_rejects_missing_ohlcv_columns():
    with pytest.raises(ValueError, match="Missing OHLCV"):
        forecast_next_minute(pd.DataFrame({"timestamp": ["2026-10-09 09:15"]}))


def test_does_not_require_future_candle():
    candles = sample_candles(3)
    forecast = forecast_next_minute(candles, lookback=3)
    assert forecast.history_bars == 3
    assert forecast.predicted_low < forecast.predicted_high
