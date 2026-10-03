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
) -> RiskDecision:
    if entry <= 0 or capital <= 0:
        return RiskDecision(False, 0, entry, entry, entry, 0, 0, "Invalid entry or capital")

    stop_distance = max(entry * stop_pct, atr * atr_multiple if atr > 0 else 0)
    # Target is measured from the actual stop distance so the requested
    # reward/risk floor remains valid even when ATR widens the stop.
    target_distance = max(entry * stop_pct * target_multiple, stop_distance * target_multiple)

    stop = entry - stop_distance if direction == "BUY" else entry + stop_distance
    target = entry + target_distance if direction == "BUY" else entry - target_distance
    risk_amount = capital * risk_fraction
    qty_by_risk = math.floor(risk_amount / stop_distance) if stop_distance else 0
    qty_by_capital = math.floor((capital * max_position_fraction) / entry)
    qty = max(0, min(qty_by_risk, qty_by_capital))
    rr = target_distance / stop_distance if stop_distance else 0
    allowed = qty > 0 and rr >= 1.5
    reason = "Risk and reward constraints satisfied" if allowed else "Position rejected by risk/reward or capital limits"
    return RiskDecision(allowed, qty, round(entry, 4), round(stop, 4), round(target, 4), round(qty * stop_distance, 2), round(rr, 3), reason)
