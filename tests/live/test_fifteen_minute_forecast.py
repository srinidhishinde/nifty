from datetime import date
from pathlib import Path

import pandas as pd

from marketdata.daily_store import DailyMarketStore, load_captured_candles
from prediction.next_minute_forecast import forecast_next_minute


def _bars() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-10-12 10:00", periods=5, freq="15min", tz="Asia/Kolkata"),
        "open": [100.0, 101.0, 100.5, 102.0, 101.5],
        "high": [102.0, 102.0, 102.0, 103.0, 103.0],
        "low": [99.0, 100.0, 99.5, 101.0, 100.5],
        "close": [101.0, 100.5, 101.5, 101.5, 102.5],
        "volume": [10, 12, 9, 15, 11],
    })


def test_next_candle_forecast_targets_next_15_minute_bucket():
    frame = _bars()
    result = forecast_next_minute(frame, underlying="NIFTY22700CE", timeframe_minutes=15)
    assert pd.Timestamp(result.target_candle_time) == frame["timestamp"].iloc[-1] + pd.Timedelta(minutes=15)
    assert result.reference_price == frame["close"].iloc[-1]


def test_daily_store_keeps_1m_and_15m_candles_separate(tmp_path: Path):
    store = DailyMarketStore(root=tmp_path)
    frame = _bars()
    store.save_candles("NIFTY22700CE", frame.iloc[[0]], date(2026, 10, 12))
    fifteen = frame.iloc[[0]].copy()
    fifteen["timeframe"] = "15min"
    fifteen["data_source"] = "KOTAK_OPTION_SFEED_15M"
    path = store.save_candles("NIFTY22700CE", fifteen, date(2026, 10, 12), timeframe_minutes=15)

    assert path.name == "NIFTY22700CE_15min_ohlcv.csv"
    one_minute, _ = load_captured_candles("NIFTY22700CE", root=tmp_path, timeframe_minutes=1)
    fifteen_minute, _ = load_captured_candles("NIFTY22700CE", root=tmp_path, timeframe_minutes=15)
    assert len(one_minute) == 1
    assert len(fifteen_minute) == 1
    assert str(fifteen_minute.iloc[0]["timeframe"]) == "15min"
