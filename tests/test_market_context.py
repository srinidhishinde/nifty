import pandas as pd

from analysis.market_context import (
    Evidence,
    analyze_breadth,
    analyze_option_positioning,
    combine_evidence,
)


def test_oi_alone_is_not_used_as_directional_signal():
    chain = pd.DataFrame([
        {"option_type": "CE", "ltp_change": 0, "oi_change": 1000},
        {"option_type": "PE", "ltp_change": 0, "oi_change": 2000},
    ])
    evidence, _ = analyze_option_positioning(chain)
    assert evidence.direction == 0


def test_breadth_counts_advancers_and_decliners():
    breadth = pd.DataFrame([
        {"symbol": "A", "change_pct": 1.0},
        {"symbol": "B", "change_pct": 0.5},
        {"symbol": "C", "change_pct": -0.2},
    ])
    assert analyze_breadth(breadth).direction == 1


def test_conflicting_evidence_returns_wait():
    decision, _ = combine_evidence([
        Evidence("price", 1, "up"),
        Evidence("breadth", -1, "down"),
        Evidence("options", 1, "up"),
    ])
    assert decision == "WAIT"


def test_two_agreeing_inputs_are_required():
    decision, _ = combine_evidence([
        Evidence("price", 1, "up"),
        Evidence("breadth", 1, "up"),
        Evidence("options", 0, "unknown"),
    ])
    assert decision.startswith("BULLISH BIAS")
