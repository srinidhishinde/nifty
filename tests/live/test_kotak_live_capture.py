from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from marketdata.kotak_live import FiveMinuteCandleBuilder, normalize_sfeed_message

IST = ZoneInfo("Asia/Kolkata")


def tick(segment, token, symbol, ts, price, volume=10, oi=None):
    return SimpleNamespace(
        exchange_segment=segment,
        instrument_token=token,
        trading_symbol=symbol,
        timestamp=ts,
        last_traded_price=price,
        volume=volume,
        open_interest=oi,
    )


def test_normalize_sfeed_message_rejects_invalid_market_data():
    assert normalize_sfeed_message(
        tick("nse_cm", "Nifty 50", "Nifty 50", None, 25000)
    ) is None
    assert normalize_sfeed_message(
        tick("nse_cm", "Nifty 50", "Nifty 50", "2026-10-05T10:00:00+05:30", 0)
    ) is None


def test_normalize_sfeed_message_accepts_nifty_index():
    result = normalize_sfeed_message(
        tick(
            "nse_cm", "Nifty 50", "Nifty 50",
            "2026-10-05T10:00:03+05:30", 25001.5,
        )
    )
    assert result is not None
    assert result.instrument == "NIFTY"
    assert result.ltp == 25001.5


def test_five_minute_builder_emits_only_completed_buckets():
    builder = FiveMinuteCandleBuilder()
    builder.update(
        normalize_sfeed_message(
            tick("mcx_fo", "123", "CRUDEOIL26OCT", "2026-10-05T10:01:00+05:30", 7000, 5)
        )
    )
    assert builder.update(
        normalize_sfeed_message(
            tick("mcx_fo", "123", "CRUDEOIL26OCT", "2026-10-05T10:04:59+05:30", 7005, 7)
        )
    ) == []
    completed = builder.update(
        normalize_sfeed_message(
            tick("mcx_fo", "123", "CRUDEOIL26OCT", "2026-10-05T10:05:01+05:30", 6998, 3)
        )
    )
    assert len(completed) == 1
    candle = completed[0]
    assert candle["instrument"] == "CRUDEOIL26OCT"
    assert candle["open"] == 7000
    assert candle["high"] == 7005
    assert candle["low"] == 7000
    assert candle["close"] == 7005
    assert candle["volume"] == 12
    assert candle["data_source"] == "KOTAK_CAPTURED"


def test_flush_completed_never_emits_current_partial_bucket():
    builder = FiveMinuteCandleBuilder()
    builder.update(
        normalize_sfeed_message(
            tick("nse_cm", "Nifty 50", "Nifty 50", "2026-10-05T10:06:00+05:30", 25000)
        )
    )
    assert builder.flush_completed(datetime(2026, 10, 5, 10, 9, tzinfo=IST)) == []
    completed = builder.flush_completed(datetime(2026, 10, 5, 10, 10, tzinfo=IST))
    assert len(completed) == 1
    assert completed[0]["timestamp"].minute == 5
