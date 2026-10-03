from dataclasses import dataclass
from typing import Literal


Decision = Literal["CE", "PE", "WAIT"]


# ============================================================
# Legacy / compatibility selector
# ============================================================

def select_direction(
    technical_ce_score: float,
    technical_pe_score: float,
    option_ce_score: float,
    option_pe_score: float,
    news_ce_score: float,
    news_pe_score: float,
    ml_ce_probability: float,
    ml_pe_probability: float,
    minimum_edge: float = 5.0,
) -> tuple[str, float]:
    """
    Combine technical, option-chain, news and ML evidence.

    This function is intentionally deterministic and is retained
    for compatibility with the existing strategy tests.

    Scores:
        technical_* : 0-100
        option_*    : 0-100
        news_*      : 0-100
        ml_*        : 0-1

    Returns:
        ("CE" | "PE" | "WAIT", confidence)
    """

    # Normalize ML probabilities to a 0-100 scale.
    ml_ce_score = max(
        0.0,
        min(100.0, ml_ce_probability * 100.0),
    )

    ml_pe_score = max(
        0.0,
        min(100.0, ml_pe_probability * 100.0),
    )

    # Equal weighting for the initial research model.
    ce_score = (
        technical_ce_score
        + option_ce_score
        + news_ce_score
        + ml_ce_score
    ) / 4.0

    pe_score = (
        technical_pe_score
        + option_pe_score
        + news_pe_score
        + ml_pe_score
    ) / 4.0

    edge = abs(ce_score - pe_score)

    if edge < minimum_edge:

        return (
            "WAIT",
            round(max(ce_score, pe_score), 2),
        )

    if ce_score > pe_score:

        return (
            "CE",
            round(ce_score, 2),
        )

    return (
        "PE",
        round(pe_score, 2),
    )


# ============================================================
# Market context
# ============================================================

@dataclass(frozen=True)
class MarketContext:

    timeframe: str

    trend: Literal[
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    ]

    momentum: Literal[
        "POSITIVE",
        "NEGATIVE",
        "NEUTRAL",
    ]

    price_vs_vwap: Literal[
        "ABOVE",
        "BELOW",
        "AT",
    ]

    volatility_regime: Literal[
        "LOW",
        "NORMAL",
        "HIGH",
    ]


# ============================================================
# Option analysis
# ============================================================

@dataclass(frozen=True)
class OptionAnalysis:

    option_type: Literal["CE", "PE"]

    strike: float

    ltp: float

    volume: float

    open_interest: float

    oi_change: float

    implied_volatility: float

    score: float

    reasons: tuple[str, ...]


# ============================================================
# Final signal
# ============================================================

@dataclass(frozen=True)
class Signal:

    decision: Decision

    confidence: float

    ce_score: float

    pe_score: float

    edge: float

    reasons: tuple[str, ...]


# ============================================================
# CE / PE selector
# ============================================================

class CEPESelector:

    def __init__(
        self,
        minimum_confidence: float = 60.0,
        minimum_edge: float = 10.0,
    ):

        self.minimum_confidence = (
            minimum_confidence
        )

        self.minimum_edge = minimum_edge

    def generate(
        self,
        context: MarketContext,
        ce: OptionAnalysis,
        pe: OptionAnalysis,
    ) -> Signal:

        ce_score = float(ce.score)
        pe_score = float(pe.score)

        reasons: list[str] = []

        # ----------------------------------------------------
        # Trend
        # ----------------------------------------------------

        if context.trend == "BULLISH":

            ce_score += 10

            reasons.append(
                "Market trend is bullish"
            )

        elif context.trend == "BEARISH":

            pe_score += 10

            reasons.append(
                "Market trend is bearish"
            )

        # ----------------------------------------------------
        # Momentum
        # ----------------------------------------------------

        if context.momentum == "POSITIVE":

            ce_score += 7

            reasons.append(
                "Momentum is positive"
            )

        elif context.momentum == "NEGATIVE":

            pe_score += 7

            reasons.append(
                "Momentum is negative"
            )

        # ----------------------------------------------------
        # VWAP
        # ----------------------------------------------------

        if context.price_vs_vwap == "ABOVE":

            ce_score += 5

            reasons.append(
                "Price is above VWAP"
            )

        elif context.price_vs_vwap == "BELOW":

            pe_score += 5

            reasons.append(
                "Price is below VWAP"
            )

        # ----------------------------------------------------
        # Volatility
        # ----------------------------------------------------

        if context.volatility_regime == "HIGH":

            ce_score -= 5
            pe_score -= 5

            reasons.append(
                "High volatility reduces confidence"
            )

        ce_score = max(
            0.0,
            min(100.0, ce_score),
        )

        pe_score = max(
            0.0,
            min(100.0, pe_score),
        )

        edge = abs(
            ce_score - pe_score
        )

        # ----------------------------------------------------
        # Conflict / insufficient edge
        # ----------------------------------------------------

        if edge < self.minimum_edge:

            reasons.append(
                "CE/PE scores are too close"
            )

            return Signal(
                decision="WAIT",
                confidence=round(
                    max(ce_score, pe_score),
                    2,
                ),
                ce_score=round(ce_score, 2),
                pe_score=round(pe_score, 2),
                edge=round(edge, 2),
                reasons=tuple(reasons),
            )

        confidence = max(
            ce_score,
            pe_score,
        )

        # ----------------------------------------------------
        # Confidence filter
        # ----------------------------------------------------

        if confidence < self.minimum_confidence:

            reasons.append(
                "Confidence is below configured threshold"
            )

            return Signal(
                decision="WAIT",
                confidence=round(
                    confidence,
                    2,
                ),
                ce_score=round(ce_score, 2),
                pe_score=round(pe_score, 2),
                edge=round(edge, 2),
                reasons=tuple(reasons),
            )

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        if ce_score > pe_score:

            decision: Decision = "CE"

            reasons.append(
                "CE has stronger combined evidence"
            )

        else:

            decision = "PE"

            reasons.append(
                "PE has stronger combined evidence"
            )

        return Signal(
            decision=decision,
            confidence=round(
                confidence,
                2,
            ),
            ce_score=round(ce_score, 2),
            pe_score=round(pe_score, 2),
            edge=round(edge, 2),
            reasons=tuple(reasons),
        )