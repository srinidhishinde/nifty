from app.main import build_research_signal, build_research_option_chain


def test_research_signal_fails_closed_without_real_captured_data(monkeypatch):
    empty = __import__("pandas").DataFrame()
    monkeypatch.setattr(
        "marketdata.daily_store.load_captured_candles",
        lambda *args, **kwargs: (empty, None),
    )
    monkeypatch.setattr(
        "marketdata.daily_store.DailyMarketStore.load_latest_option_chain_snapshot",
        lambda *args, **kwargs: (empty, None),
    )
    result = build_research_signal("NIFTY", "5m", seed=42)
    signal = result[3]
    assert signal.decision == "WAIT"
    assert result[1] is None and result[2] is None
    assert result[4] is None


def test_research_option_chain_never_generates_synthetic_contracts(monkeypatch):
    empty = __import__("pandas").DataFrame()
    monkeypatch.setattr(
        "marketdata.daily_store.DailyMarketStore.load_latest_option_chain_snapshot",
        lambda *args, **kwargs: (empty, None),
    )
    assert build_research_option_chain("NIFTY", 25000.0, 42, 50.0) == []
