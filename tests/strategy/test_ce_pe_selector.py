from features.option_chain import (
    OptionAnalysis,
)

from strategy.ce_pe_selector import (
    CEPESelector,
    MarketContext,
)


def make_option(
    option_type: str,
    score: float,
) -> OptionAnalysis:

    return OptionAnalysis(
        option_type=option_type,
        strike=25000,
        ltp=100,
        volume=10000,
        open_interest=20000,
        oi_change=5000,
        implied_volatility=20,
        score=score,
        reasons=("test",),
    )


def test_bullish_context_can_select_ce():

    selector = CEPESelector(
        minimum_confidence=60,
        minimum_edge=10,
    )

    context = MarketContext(
        timeframe="5m",
        trend="BULLISH",
        momentum="POSITIVE",
        price_vs_vwap="ABOVE",
        volatility_regime="NORMAL",
    )

    signal = selector.generate(
        context=context,
        ce=make_option("CE", 65),
        pe=make_option("PE", 40),
    )

    assert signal.decision == "CE"

    assert signal.ce_score > signal.pe_score


def test_bearish_context_can_select_pe():

    selector = CEPESelector(
        minimum_confidence=60,
        minimum_edge=10,
    )

    context = MarketContext(
        timeframe="15m",
        trend="BEARISH",
        momentum="NEGATIVE",
        price_vs_vwap="BELOW",
        volatility_regime="NORMAL",
    )

    signal = selector.generate(
        context=context,
        ce=make_option("CE", 40),
        pe=make_option("PE", 65),
    )

    assert signal.decision == "PE"

    assert signal.pe_score > signal.ce_score


def test_conflicting_scores_return_wait():

    selector = CEPESelector(
        minimum_confidence=60,
        minimum_edge=10,
    )

    context = MarketContext(
        timeframe="5m",
        trend="NEUTRAL",
        momentum="NEUTRAL",
        price_vs_vwap="AT",
        volatility_regime="NORMAL",
    )

    signal = selector.generate(
        context=context,
        ce=make_option("CE", 60),
        pe=make_option("PE", 55),
    )

    assert signal.decision == "WAIT"


def test_high_volatility_reduces_scores():

    selector = CEPESelector(
        minimum_confidence=60,
        minimum_edge=10,
    )

    context = MarketContext(
        timeframe="5m",
        trend="BULLISH",
        momentum="POSITIVE",
        price_vs_vwap="ABOVE",
        volatility_regime="HIGH",
    )

    signal = selector.generate(
        context=context,
        ce=make_option("CE", 60),
        pe=make_option("PE", 40),
    )

    assert signal.ce_score < 80

    assert signal.pe_score < 50