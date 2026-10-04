from dataclasses import dataclass
from typing import Literal

OptionType = Literal["CE", "PE"]

@dataclass(frozen=True)
class OptionContract:
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
    gamma: float | None = None
    ltp_change_pct: float = 0.0

@dataclass(frozen=True)
class OptionAnalysis:
    option_type: OptionType
    strike: float
    ltp: float
    volume: float
    open_interest: float
    oi_change: float
    implied_volatility: float
    score: float
    reasons: tuple[str, ...]

def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator

def analyze_option(contract: OptionContract) -> OptionAnalysis:
    score = 50.0
    reasons: list[str] = []
    if contract.oi_change > 0:
        score += 10; reasons.append("Open interest increased")
    elif contract.oi_change < 0:
        score -= 5; reasons.append("Open interest decreased")
    volume_oi_ratio = _safe_ratio(contract.volume, contract.open_interest)
    if volume_oi_ratio >= 0.5:
        score += 10; reasons.append("Strong volume relative to open interest")
    elif volume_oi_ratio < 0.1:
        score -= 5; reasons.append("Low volume relative to open interest")
    if contract.implied_volatility > 40:
        score -= 10; reasons.append("Elevated implied volatility")
    elif 0 < contract.implied_volatility < 15:
        score += 5; reasons.append("Relatively low implied volatility")
    if contract.ltp > 0 and contract.ask >= contract.bid >= 0:
        spread_pct = ((contract.ask - contract.bid) / contract.ltp) * 100
        if spread_pct <= 1:
            score += 5; reasons.append("Tight bid/ask spread")
        elif spread_pct >= 5:
            score -= 10; reasons.append("Wide bid/ask spread")
    score = max(0.0, min(100.0, score))
    return OptionAnalysis(contract.option_type, contract.strike, contract.ltp, contract.volume, contract.open_interest, contract.oi_change, contract.implied_volatility, round(score, 2), tuple(reasons))

def compare_ce_pe(ce: OptionAnalysis, pe: OptionAnalysis, minimum_edge: float = 10.0) -> tuple[str, float, str]:
    difference = ce.score - pe.score
    if abs(difference) < minimum_edge:
        return "WAIT", round(abs(difference), 2), "CE/PE scores are too close"
    if difference > 0:
        return "CE", round(difference, 2), "CE has the stronger option-chain score"
    return "PE", round(abs(difference), 2), "PE has the stronger option-chain score"
