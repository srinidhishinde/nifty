from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    quantity: int
    entry: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    reward_risk: float
    reason: str


def calculate_risk_levels(
    entry: float,
    atr: float,
    direction: str,
    stop_pct: float = 0.02,
    atr_multiple: float = 1.5,
    target_multiple: float = 2.0,
) -> tuple[float, float, float, float]:
    direction = direction.upper()
    if direction not in {"BUY", "SELL"} or entry <= 0:
        raise ValueError("entry must be positive and direction must be BUY or SELL")
    atr = max(float(atr), 0.0)
    stop_distance = max(entry * stop_pct, atr * atr_multiple)
    target_distance = max(entry * stop_pct * target_multiple, stop_distance * target_multiple)
    stop = entry - stop_distance if direction == "BUY" else entry + stop_distance
    target = entry + target_distance if direction == "BUY" else entry - target_distance
    reward_risk = target_distance / stop_distance if stop_distance else 0.0
    return stop_distance, stop, target, reward_risk


def size_position(
    entry: float,
    atr: float,
    direction: str,
    capital: float,
    risk_fraction: float = 0.01,
    max_position_fraction: float = 0.20,
    stop_pct: float = 0.02,
    atr_multiple: float = 1.5,
    target_multiple: float = 2.0,
    min_reward_risk: float = 1.5,
) -> RiskDecision:
    if entry <= 0 or capital <= 0:
        return RiskDecision(False, 0, entry, entry, entry, 0, 0, "Invalid entry or capital")
    if not 0 < risk_fraction <= 0.05:
        return RiskDecision(False, 0, entry, entry, entry, 0, 0, "Risk fraction must be between 0 and 5%")
    try:
        stop_distance, stop, target, rr = calculate_risk_levels(
            entry, atr, direction, stop_pct, atr_multiple, target_multiple
        )
    except ValueError as exc:
        return RiskDecision(False, 0, entry, entry, entry, 0, 0, str(exc))

    risk_amount = capital * risk_fraction
    qty_by_risk = math.floor(risk_amount / stop_distance) if stop_distance else 0
    qty_by_capital = math.floor((capital * max_position_fraction) / entry)
    qty = max(0, min(qty_by_risk, qty_by_capital))
    allowed = qty > 0 and rr >= min_reward_risk
    reason = "Risk and reward constraints satisfied" if allowed else "Position rejected by risk/reward or capital limits"
    return RiskDecision(
        allowed,
        qty,
        round(entry, 4),
        round(stop, 4),
        round(target, 4),
        round(qty * stop_distance, 2),
        round(rr, 3),
        reason,
    )
