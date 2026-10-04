import pandas as pd

from features.option_chain import OptionContract
from strategy.closing_session import evaluate_closing_session


def test_closing_session_requires_real_chain_and_window():
    candles = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-08-28 15:15", "2026-08-28 15:20", "2026-08-28 15:25",
            "2026-08-28 15:30", "2026-08-28 15:35"
        ], utc=True),
        "close": [25000, 25010, 25020, 25030, 25035],
    })
    # UTC timestamps are deliberately outside IST closing time; engine must not
    # silently interpret them as local exchange time.
    signal, table = evaluate_closing_session(candles, [])
    assert signal.status == "NO_DATA"


def test_closing_session_produces_risk_levels_from_real_contract():
    candles = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-08-28 15:15+05:30", "2026-08-28 15:20+05:30", "2026-08-28 15:25+05:30",
            "2026-08-28 15:30+05:30", "2026-08-28 15:35+05:30"
        ]),
        "close": [25000, 25010, 25020, 25030, 25035],
    })
    c = OptionContract("NIFTY25000CE", "2026-09-03", 25000, "CE", 100, 99, 101, 10000, 50000, 5000, 18)
    signal, table = evaluate_closing_session(candles, [c], news_score=0)
    assert signal.option_type == "CE"
    assert signal.entry == 101
    assert signal.stop_loss < signal.entry < signal.target
