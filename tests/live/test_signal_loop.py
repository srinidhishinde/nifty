import pandas as pd

from features.option_chain import OptionContract
from live.signal_loop import LiveSignalLoop


def test_live_loop_runs_one_cycle(monkeypatch):
    monkeypatch.setattr(
        "live.signal_loop.fetch_global_news",
        lambda: type("N", (), {"sentiment": 0.1})(),
    )
    candles = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-01 09:15", periods=60, freq="5min"),
        "open": [100 + i * 0.1 for i in range(60)],
        "high": [100.2 + i * 0.1 for i in range(60)],
        "low": [99.8 + i * 0.1 for i in range(60)],
        "close": [100 + i * 0.1 for i in range(60)],
        "volume": [1000] * 60,
    })
    options = [OptionContract("CE", "", 25000, "CE", 100, 99, 101, 1000, 2000, 500, 20)]
    loop = LiveSignalLoop(
        lambda: candles,
        lambda: options,
        lambda: 25000,
        interval_seconds=0,
        capital=100000.0,
    )
    result = loop.run_once()

    assert "option_confidence" in result
    assert "ai_confidence" in result
    assert "ai_abstain" in result
    assert "position_quantity" in result
    assert "global_news" in result
