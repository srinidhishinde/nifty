import pandas as pd
import pytest

from marketdata.option_chain_csv import is_option_chain_snapshot, parse_option_chain_csv


OPTION_COLUMNS = [
    "Calls Built Up", "Calls Vega", "Calls Theta ", "Calls Delta",
    "Calls Volume", "Calls CHG IN OI", "Calls OI", "Calls IV",
    "Calls LTP (Chg %)", "Strike", "Puts LTP (Chg %)", "Puts IV",
    "Puts OI", "Puts CHG IN OI", "Puts Volume", "Puts Delta",
    "Puts Theta ", "Puts Vega", "Puts Built Up",
]


def test_option_chain_snapshot_is_detected_and_parsed():
    data = pd.DataFrame([
        ["Long Buildup", 1.0, -2.0, 0.5, 1000, 200, 5000, 15, 2.5, 25000, -1.5, 16, 6000, 300, 1200, -0.4, -1.0, 1.2, "Short Buildup"],
    ], columns=OPTION_COLUMNS)
    assert is_option_chain_snapshot(data.columns)
    contracts = parse_option_chain_csv(data)
    assert len(contracts) == 2
    ce = next(x for x in contracts if x.option_type == "CE")
    assert ce.strike == 25000
    assert ce.open_interest == 5000
    assert ce.oi_change == 200
    assert ce.volume == 1000
    assert ce.ltp_change_pct == 2.5
    assert ce.built_up == "Long Buildup"


def test_option_chain_snapshot_is_not_candle_backtest_input():
    data = pd.DataFrame({"Strike": [25000], "Calls OI": [5000]})
    assert is_option_chain_snapshot(data.columns)
    assert "open" not in data.columns
    assert "high" not in data.columns
    assert "low" not in data.columns
    assert "close" not in data.columns
    assert "volume" not in data.columns
