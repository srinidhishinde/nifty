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
        last_update_time=ts,
        last_trade_time=None,
        last_traded_price=price,
        volume_traded_today=volume,
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
            tick("mcx_fo", "123", "CRUDEOIL26OCT", "2026-10-05T10:04:59+05:30", 7005, 12)
        )
    ) == []
    completed = builder.update(
        normalize_sfeed_message(
            tick("mcx_fo", "123", "CRUDEOIL26OCT", "2026-10-05T10:05:01+05:30", 6998, 15)
        )
    )
    assert len(completed) == 1
    candle = completed[0]
    assert candle["instrument"] == "CRUDEOIL26OCT"
    assert candle["open"] == 7000
    assert candle["high"] == 7005
    assert candle["low"] == 7000
    assert candle["close"] == 7005
    # Kotak volume_traded_today is cumulative: 5 -> 12 = 7 traded in bucket.
    assert candle["volume"] == 7
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


def test_volume_traded_today_is_supported_as_cumulative_volume():
    result = normalize_sfeed_message(
        tick(
            "mcx_fo", "123", "CRUDEOIL26OCT",
            "2026-10-05T10:01:00+05:30", 7000, 1000,
        )
    )
    assert result is not None
    assert result.volume == 1000


def test_candle_builder_converts_cumulative_volume_to_delta():
    builder = FiveMinuteCandleBuilder()
    a = normalize_sfeed_message(tick("mcx_fo", "123", "CRUDEOIL26OCT", "2026-10-05T10:01:00+05:30", 7000, 1000))
    b = normalize_sfeed_message(tick("mcx_fo", "123", "CRUDEOIL26OCT", "2026-10-05T10:02:00+05:30", 7002, 1015))
    c2 = normalize_sfeed_message(tick("mcx_fo", "123", "CRUDEOIL26OCT", "2026-10-05T10:06:00+05:30", 7001, 1030))
    assert builder.update(a) == []
    assert builder.update(b) == []
    completed = builder.update(c2)
    assert completed[0]["volume"] == 15


from marketdata.daily_store import DailyMarketStore
from scripts.capture_daily_market_data import capture_option_chain


def test_capture_option_chain_persists_real_nifty_snapshot(tmp_path):
    class Provider:
        def get_option_chain(self, **kwargs):
            return [
                SimpleNamespace(
                    symbol="NIFTY26OCT25000CE", exchange="NSE_FO", underlying="NIFTY",
                    expiry="2026-10-13", strike=25000.0, option_type="CE",
                    instrument_token="1", ltp=125.5, bid=125.0, ask=126.0,
                    volume=1000, open_interest=2000, oi_change=100,
                    implied_volatility=12.5, built_up="Long Built Up",
                    delta=0.5, theta=-2.0, vega=1.2, gamma=0.01, ltp_change_pct=2.0,
                ),
                SimpleNamespace(
                    symbol="NIFTY26OCT25000PE", exchange="NSE_FO", underlying="NIFTY",
                    expiry="2026-10-13", strike=25000.0, option_type="PE",
                    instrument_token="2", ltp=120.5, bid=120.0, ask=121.0,
                    volume=900, open_interest=1800, oi_change=-50,
                    implied_volatility=13.0, built_up="Short Built Up",
                    delta=-0.5, theta=-2.1, vega=1.3, gamma=0.01, ltp_change_pct=-1.0,
                ),
            ]

    store = DailyMarketStore(tmp_path)
    count, path = capture_option_chain(Provider(), store, "NIFTY", count=40)

    assert count == 2
    assert path.exists()
    frame, captured_at = store.load_latest_option_chain_snapshot("NIFTY")
    assert len(frame) == 2
    assert set(frame["option_type"]) == {"CE", "PE"}
    assert set(frame["strike"]) == {25000.0}
    assert captured_at is not None


def test_option_trade_levels_follow_live_entry_and_never_use_static_target():
    from features.option_signal_engine import generate_option_chain_signal
    from features.option_chain import OptionContract

    base = OptionContract(
        symbol="NIFTY26OCT25000CE", expiry="2026-10-13", strike=25000.0,
        option_type="CE", ltp=11.0, bid=10.8, ask=11.2,
        volume=10000, open_interest=20000, oi_change=500,
        implied_volatility=20.0, ltp_change_pct=1.0,
    )
    signal_a, _ = generate_option_chain_signal(
        [base], spot=25000.0, direction_hint="CE", require_two_sided_quote=True
    )
    changed = OptionContract(**{**base.__dict__, "ltp": 13.0, "bid": 12.8, "ask": 13.2})
    signal_b, _ = generate_option_chain_signal(
        [changed], spot=25000.0, direction_hint="CE", require_two_sided_quote=True
    )

    assert signal_a.direction == "BUY CE"
    assert signal_b.direction == "BUY CE"
    assert signal_a.entry_price == 11.2
    assert signal_b.entry_price == 13.2
    assert signal_a.take_profit != 14.0
    assert signal_b.take_profit != signal_a.take_profit
    assert signal_a.stop_loss < signal_a.entry_price < signal_a.take_profit
    assert signal_b.stop_loss < signal_b.entry_price < signal_b.take_profit


def test_option_trade_is_blocked_without_two_sided_live_quote():
    from features.option_signal_engine import generate_option_chain_signal
    from features.option_chain import OptionContract

    contract = OptionContract(
        symbol="NIFTY26OCT25000CE", expiry="2026-10-13", strike=25000.0,
        option_type="CE", ltp=125.0, bid=None, ask=None,
        volume=10000, open_interest=20000, oi_change=500,
        implied_volatility=20.0,
    )
    signal, rows = generate_option_chain_signal(
        [contract], spot=25000.0, direction_hint="CE", require_two_sided_quote=True
    )
    assert signal.direction == "WAIT"
    assert signal.entry_price is None
    assert rows.empty
