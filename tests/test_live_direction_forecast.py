from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction.live_direction_forecast import predict_live_direction


def _candles(rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(20261012)
    returns = rng.normal(0.0001, 0.0012, rows)
    close = 220.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.02, 0.2, rows)
    return pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-10-01 09:15:00", periods=rows, freq="5min", tz="Asia/Kolkata"
        ),
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": rng.integers(100, 5000, rows),
    })


def test_live_direction_uses_prior_closed_outcomes_and_returns_advisory_forecast():
    result = predict_live_direction(_candles(), min_train=30, timeframe_minutes=5)
    assert result["direction"] in {"UP", "DOWN"}
    assert result["method"].startswith(("WALK_FORWARD_TRAINED_", "MOMENTUM_FALLBACK"))
    assert result["history_rows"] == 180
    assert result["training_rows"] < result["history_rows"]
    assert result["target_candle_time"] > result["latest_candle_time"]
    assert result["advisory_only"] is True
    assert result["live_orders_enabled"] is False
    assert result["score_is_calibrated_probability"] is False


def test_live_direction_rejects_too_little_history():
    with pytest.raises(ValueError, match="At least three valid candles"):
        predict_live_direction(_candles(2))


def test_live_direction_rejects_invalid_timeframe():
    with pytest.raises(ValueError, match="timeframe_minutes"):
        predict_live_direction(_candles(), timeframe_minutes=7)
