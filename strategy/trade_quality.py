from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TradeQualityResult:
    approved: bool
    score: float
    expected_r: float
    probability_win: float
    reasons: tuple[str, ...]


def calculate_expected_r(
    probability_win: float,
    avg_win_r: float,
    avg_loss_r: float,
    cost_r: float = 0.0,
) -> float:
    p = max(0.0, min(1.0, float(probability_win)))
    win = max(0.0, float(avg_win_r))
    loss = max(0.0, float(avg_loss_r))
    return p * win - (1.0 - p) * loss - max(0.0, float(cost_r))


def evaluate_trade_quality(
    *,
    probability_win: float,
    avg_win_r: float,
    avg_loss_r: float,
    cost_r: float,
    trend_alignment: float,
    rule_agreement_pct: float,
    regime_tradable: bool,
    robustness_score: float,
    uncertainty: float,
    liquidity_score: float,
    min_expected_r: float = 1.5,
    min_probability: float = 0.60,
    min_quality: float = 70.0,
    min_robustness: float = 0.60,
) -> TradeQualityResult:
    p = max(0.0, min(1.0, float(probability_win)))
    expected_r = calculate_expected_r(p, avg_win_r, avg_loss_r, cost_r)
    score = (
        p * 30.0
        + max(-100.0, min(100.0, trend_alignment)) * 0.10
        + max(0.0, min(100.0, rule_agreement_pct)) * 0.20
        + max(0.0, min(1.0, robustness_score)) * 15.0
        + max(0.0, min(100.0, liquidity_score)) * 0.10
        + (10.0 if regime_tradable else 0.0)
        + max(0.0, min(3.0, expected_r)) * 5.0
        - max(0.0, min(100.0, uncertainty)) * 0.10
    )
    score = max(0.0, min(100.0, score))
    reasons: list[str] = []
    if not regime_tradable:
        reasons.append("Current regime is not tradable.")
    if p < min_probability:
        reasons.append(f"Calibrated win probability {p:.1%} is below {min_probability:.1%}.")
    if expected_r < min_expected_r:
        reasons.append(f"Net expected R {expected_r:.2f} is below {min_expected_r:.2f}.")
    if robustness_score < min_robustness:
        reasons.append(f"Out-of-sample robustness {robustness_score:.2f} is below {min_robustness:.2f}.")
    if uncertainty >= 60.0:
        reasons.append("Model uncertainty is high.")
    if liquidity_score < 60.0:
        reasons.append("Liquidity quality is weak.")
    if not reasons and score >= min_quality:
        reasons.append("Validated probability, expectancy, regime, liquidity and robustness agree.")
    approved = (
        regime_tradable
        and p >= min_probability
        and expected_r >= min_expected_r
        and robustness_score >= min_robustness
        and uncertainty < 60.0
        and liquidity_score >= 60.0
        and score >= min_quality
    )
    return TradeQualityResult(
        approved=approved,
        score=round(score, 2),
        expected_r=round(expected_r, 3),
        probability_win=round(p, 4),
        reasons=tuple(reasons),
    )


def safe_probability(raw: float) -> float:
    if not math.isfinite(float(raw)):
        return 0.0
    return max(0.0, min(1.0, float(raw)))
