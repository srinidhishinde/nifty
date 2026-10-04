import pandas as pd
import pytest
from marketdata.option_replay import normalize_option_replay, option_execution_price


def test_option_replay_requires_bid_ask_and_ltp():
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_option_replay(pd.DataFrame({"timestamp": [], "symbol": []}))


def test_option_replay_uses_ask_for_buy_and_bid_for_sell():
    data = pd.DataFrame([{
        "timestamp": "2026-10-01 10:00", "symbol": "NIFTY", "expiry": "2026-10-29",
        "strike": 25000, "option_type": "CE", "bid": 100, "ask": 101,
        "ltp": 100.5, "volume": 1000, "oi": 2000, "iv": 15,
    }])
    frame = normalize_option_replay(data)
    assert option_execution_price(frame.iloc[0], "BUY") == 101
    assert option_execution_price(frame.iloc[0], "SELL") == 100
