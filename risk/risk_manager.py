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
    min_reward_risk: float = 1.8,
    lot_size: int = 1,
    point_value: float = 1.0,
) -> RiskDecision:
    if entry <= 0 or capital <= 0 or lot_size <= 0 or point_value <= 0:
        return RiskDecision(False, 0, entry, entry, entry, 0, 0, "Invalid entry, capital or contract specification")
    if not 0 < risk_fraction <= 0.05:
        return RiskDecision(False, 0, entry, entry, entry, 0, 0, "Risk fraction must be between 0 and 5%")
    try:
        stop_distance, stop, target, rr = calculate_risk_levels(
            entry, atr, direction, stop_pct, atr_multiple, target_multiple
        )
    except ValueError as exc:
        return RiskDecision(False, 0, entry, entry, entry, 0, 0, str(exc))

    risk_amount = capital * risk_fraction
    risk_per_lot = stop_distance * point_value * lot_size
    qty_by_risk_lots = math.floor(risk_amount / risk_per_lot) if risk_per_lot else 0
    capital_per_lot = entry * lot_size
    qty_by_capital_lots = math.floor((capital * max_position_fraction) / capital_per_lot) if capital_per_lot else 0
    lots = max(0, min(qty_by_risk_lots, qty_by_capital_lots))
    qty = lots * lot_size
    allowed = lots > 0 and rr >= min_reward_risk
    reason = "Risk and reward constraints satisfied" if allowed else "Position rejected by risk/reward or capital limits"
    return RiskDecision(
        allowed,
        qty,
        round(entry, 4),
        round(stop, 4),
        round(target, 4),
        round(lots * risk_per_lot, 2),
        round(rr, 3),
        reason,
    )


def daily_loss_allowed(realized_pnl: float, max_daily_loss: float) -> tuple[bool, str]:
    """Hard gate: realized loss cannot exceed the configured daily limit."""
    if max_daily_loss <= 0:
        return False, "Daily loss limit must be positive"
    if realized_pnl <= -abs(max_daily_loss):
        return False, "Daily loss limit reached"
    return True, "Daily loss limit available"
