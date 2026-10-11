from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction.index_aware_option_forecast import forecast_index_aware_option


def _paired():
    ts = pd.date_range("2026-10-09 09:15", periods=40, freq="min", tz="Asia/Kolkata")
    index_close = np.linspace(22400, 22440, len(ts))
    option_close = np.linspace(100, 112, len(ts))
    index_open = index_close - 0.4
    option_open = option_close - 0.15
    index = pd.DataFrame({
        "timestamp": ts, "open": index_open,
        "high": np.maximum(index_open, index_close) + 1.0,
        "low": np.minimum(index_open, index_close) - 1.0,
        "close": index_close, "volume": 1000,
    })
    option = pd.DataFrame({
        "timestamp": ts, "open": option_open,
        "high": np.maximum(option_open, option_close) + 0.3,
        "low": np.minimum(option_open, option_close) - 0.3,
        "close": option_close, "volume": 1000,
    })
    return option, index


def test_returns_index_aware_forecast_and_direction_for_every_candle():
    option, index = _paired()
    result = forecast_index_aware_option(
        option, index, strike=22400, expiry="2026-10-13", lookback=30
    )
    assert result.direction in {"UP", "DOWN"}
    assert result.index_direction in {"UP", "DOWN"}
    assert result.predicted_low <= result.predicted_premium <= result.predicted_high
    assert result.aligned_bars == 30
    assert result.iv_source == "inferred_from_option_close"
    assert result.theta_per_minute <= 0


def test_supplied_iv_accepts_percent_and_reports_theta():
    option, index = _paired()
    option["implied_volatility"] = 18.0
    result = forecast_index_aware_option(
        option, index, strike=22400, expiry="2026-10-13", lookback=30
    )
    assert result.iv_source == "implied_volatility"
    assert result.implied_volatility == pytest.approx(0.18)
    assert result.theta_contribution == pytest.approx(result.theta_per_minute)


def test_no_lookahead_future_rows_do_not_change_current_forecast():
    option, index = _paired()
    first = forecast_index_aware_option(
        option.iloc[:30], index.iloc[:30], strike=22400, expiry="2026-10-13", lookback=30
    )
    changed_option = option.copy()
    changed_index = index.copy()
    changed_option.loc[30:, ["open", "high", "low", "close"]] *= 1.8
    changed_index.loc[30:, ["open", "high", "low", "close"]] *= 1.1
    second = forecast_index_aware_option(
        changed_option.iloc[:30], changed_index.iloc[:30], strike=22400, expiry="2026-10-13", lookback=30
    )
    assert first == second


def test_rejects_misaligned_or_too_short_data():
    option, index = _paired()
    with pytest.raises(ValueError, match="aligned"):
        forecast_index_aware_option(
            option, index.assign(timestamp=index["timestamp"] + pd.Timedelta(seconds=1)),
            strike=22400, expiry="2026-10-13"
        )


def test_invalid_strike_and_range_quantile_rejected():
    option, index = _paired()
    with pytest.raises(ValueError, match="strike"):
        forecast_index_aware_option(option, index, strike=0, expiry="2026-10-13")
    with pytest.raises(ValueError, match="range_quantile"):
        forecast_index_aware_option(option, index, strike=22400, expiry="2026-10-13", range_quantile=1.0)
