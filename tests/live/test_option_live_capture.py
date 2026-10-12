from datetime import datetime
from zoneinfo import ZoneInfo

from marketdata.kotak_live import LiveTick
from marketdata.option_live_capture import OneMinuteCandleBuilder

IST = ZoneInfo("Asia/Kolkata")


def option_tick(ts, price, volume):
    return LiveTick(
        instrument="NIFTY13OCT2622700CE",
        exchange_segment="nse_fo",
        instrument_token="12345",
        trading_symbol="NIFTY13OCT2622700CE",
        timestamp=datetime.fromisoformat(ts),
        ltp=price,
        volume=volume,
        open_interest=1000,
    )


def test_one_minute_builder_emits_completed_candle_and_volume_delta():
    builder = OneMinuteCandleBuilder()
    assert builder.update(option_tick("2026-10-12T10:01:05+05:30", 100.0, 100)) == []
    assert builder.update(option_tick("2026-10-12T10:01:50+05:30", 103.0, 112)) == []
    rows = builder.update(option_tick("2026-10-12T10:02:01+05:30", 101.0, 120))
    assert len(rows) == 1
    candle = rows[0]
    assert candle["timestamp"].minute == 1
    assert candle["open"] == 100.0
    assert candle["high"] == 103.0
    assert candle["low"] == 100.0
    assert candle["close"] == 103.0
    assert candle["volume"] == 12
    assert candle["tick_count"] == 2
    assert candle["exchange_segment"] == "nse_fo"
    assert candle["data_source"] == "KOTAK_OPTION_SFEED_1M"


def test_one_minute_builder_does_not_flush_partial_current_minute():
    builder = OneMinuteCandleBuilder()
    builder.update(option_tick("2026-10-12T10:01:05+05:30", 100.0, 100))
    assert builder.flush_completed(datetime.fromisoformat("2026-10-12T10:01:59+05:30")) == []
    rows = builder.flush_completed(datetime.fromisoformat("2026-10-12T10:02:00+05:30"))
    assert len(rows) == 1
    assert rows[0]["timestamp"].minute == 1
