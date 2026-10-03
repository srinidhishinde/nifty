import pandas as pd

from strategy.live_engine import LiveRuleEngine

def test_live_engine_respects_market_hours():
    candles = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-05 08:00")],
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.0],
        "volume": [1000.0],
    })
    assert LiveRuleEngine("NIFTY").evaluate(candles) is None
