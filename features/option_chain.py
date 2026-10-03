from dataclasses import dataclass
from typing import Literal


OptionType = Literal["CE", "PE"]


@dataclass(frozen=True)
class OptionContract:
    """
    Normalized option-chain contract.

    This model is broker/data-source independent so that historical
    data and a future Kotak Neo adapter can use the same structure.
    """

    symbol: str
    expiry: str
    strike: float
    option_type: OptionType

    ltp: float
    bid: float
    ask: float

    volume: float
    open_interest: float
    oi_change: float

    implied_volatility: float
    built_up: str = ""
    delta: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    ltp_change_pct: float = 0.0


@dataclass(frozen=True)
class OptionAnalysis:
    """
    Aggregated analysis for one CE/PE contract.
    """

    option_type: OptionType
    strike: float

    ltp: float
    volume: float
    open_interest: float
    oi_change: float
    implied_volatility: float

    score: float
    reasons: tuple[str, ...]


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:

    if denominator == 0:
        return 0.0

    return numerator / denominator


def analyze_option(
    contract: OptionContract,
) -> OptionAnalysis:

    score = 50.0
    reasons: list[str] = []

    # --------------------------------------------
    # Open-interest behaviour
    # --------------------------------------------

    if contract.oi_change > 0:

        score += 10

        reasons.append(
            "Open interest increased"
        )

    elif contract.oi_change < 0:

        score -= 5

        reasons.append(
            "Open interest decreased"
        )

    # --------------------------------------------
    # Volume confirmation
    # --------------------------------------------

    volume_oi_ratio = _safe_ratio(
        contract.volume,
        contract.open_interest,
    )

    if volume_oi_ratio >= 0.5:

        score += 10

        reasons.append(
            "Strong volume relative to open interest"
        )

    elif volume_oi_ratio < 0.1:

        score -= 5

        reasons.append(
            "Low volume relative to open interest"
        )

    # --------------------------------------------
    # IV sanity / risk signal
    # --------------------------------------------

    if contract.implied_volatility > 40:

        score -= 10

        reasons.append(
            "Elevated implied volatility"
        )

    elif (
        contract.implied_volatility > 0
        and contract.implied_volatility < 15
    ):

        score += 5

        reasons.append(
            "Relatively low implied volatility"
        )

    # --------------------------------------------
    # Bid/ask quality
    # --------------------------------------------

    if contract.ltp > 0:

        spread_pct = (
            (contract.ask - contract.bid)
            / contract.ltp
        ) * 100

        if spread_pct <= 1:

            score += 5

            reasons.append(
                "Tight bid/ask spread"
            )

        elif spread_pct >= 5:

            score -= 10

            reasons.append(
                "Wide bid/ask spread"
            )

    score = max(
        0.0,
        min(100.0, score),
    )

    return OptionAnalysis(
        option_type=contract.option_type,
        strike=contract.strike,
        ltp=contract.ltp,
        volume=contract.volume,
        open_interest=contract.open_interest,
        oi_change=contract.oi_change,
        implied_volatility=contract.implied_volatility,
        score=round(score, 2),
        reasons=tuple(reasons),
    )


def compare_ce_pe(
    ce: OptionAnalysis,
    pe: OptionAnalysis,
    minimum_edge: float = 10.0,
) -> tuple[str, float, str]:

    difference = ce.score - pe.score

    if abs(difference) < minimum_edge:

        return (
            "WAIT",
            round(abs(difference), 2),
            "CE/PE scores are too close",
        )

    if difference > 0:

        return (
            "CE",
            round(difference, 2),
            "CE has the stronger option-chain score",
        )

    return (
        "PE",
        round(abs(difference), 2),
        "PE has the stronger option-chain score",
    )