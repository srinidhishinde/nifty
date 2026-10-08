import pandas as pd

from marketdata.daily_store import DailyMarketStore
from marketdata.option_chain_csv import parse_option_chain_csv
from strategy.market_specs import validate_option_strike


def test_nifty_strike_validation_rejects_corrupted_values():
    assert validate_option_strike("NIFTY", 25000) == 25000
    for strike in (1320250, 1320500, 1324250, 1320, 25125):
        try:
            validate_option_strike("NIFTY", strike)
        except ValueError:
            pass
        else:
            raise AssertionError(f"corrupted NIFTY strike accepted: {strike}")


def test_nifty_csv_import_rejects_corrupted_strikes():
    frame = pd.DataFrame({
        "Strike": [1320250, 1320300, 25000],
        "Calls OI": [1, 1, 1],
        "Puts OI": [1, 1, 1],
    })
    contracts = parse_option_chain_csv(frame)
    assert [c.strike for c in contracts] == [25000.0, 25000.0]


def test_cached_snapshot_uses_symbol_strike_when_payload_is_corrupt(tmp_path):
    store = DailyMarketStore(tmp_path)
    path = store.directory("NIFTY", pd.Timestamp("2026-10-09").date())
    path.mkdir(parents=True)
    pd.DataFrame({
        "captured_at": ["2026-10-09T10:00:00+05:30"],
        "symbol": ["NIFTY26OCT25000CE"],
        "expiry": ["2026-10-27"],
        "strike": [1320250],
        "option_type": ["CE"],
        "ltp": [100],
    }).to_csv(path / "NIFTY_option_chain_latest.csv", index=False)
    frame, _ = store.load_latest_option_chain_snapshot("NIFTY")
    assert frame.empty is False
    assert frame.iloc[0]["strike"] == 25000
