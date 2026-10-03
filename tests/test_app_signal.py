from app.main import build_research_signal


def test_research_signal_is_deterministic():

    result1 = build_research_signal(
        instrument="NIFTY",
        timeframe="5m",
        seed=42,
    )

    result2 = build_research_signal(
        instrument="NIFTY",
        timeframe="5m",
        seed=42,
    )

    signal1 = result1[3]
    signal2 = result2[3]

    assert signal1.decision == signal2.decision
    assert signal1.ce_score == signal2.ce_score
    assert signal1.pe_score == signal2.pe_score


def test_research_signal_can_produce_non_bullish_states():

    decisions = set()

    for seed in range(1, 101):

        result = build_research_signal(
            instrument="NIFTY",
            timeframe="5m",
            seed=seed,
        )

        context = result[0]
        signal = result[3]

        decisions.add(signal.decision)

    assert len(decisions) >= 2


def test_signal_uses_configured_thresholds():

    context, ce, pe, signal, probability = (
        build_research_signal(
            instrument="NIFTY",
            timeframe="5m",
            seed=42,
        )
    )

    assert 0 <= signal.ce_score <= 100
    assert 0 <= signal.pe_score <= 100
    assert 0 <= signal.confidence <= 100
    assert signal.edge >= 0
    assert 0 <= probability <= 1
