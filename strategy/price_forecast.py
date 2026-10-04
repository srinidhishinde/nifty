from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PriceForecast:
    direction: str
    entry_low: float
    entry_high: float
    invalidation: float
    target_1: float
    target_2: float
    probability_target_1: float
    probability_target_2: float
    expected_r: float
    horizon_bars: int
    uncertainty: float
    reason: str


def forecast_trade_plan(
    *,
    price: float,
    atr: float,
    direction: str,
    probability: float,
    support: float | None = None,
    resistance: float | None = None,
    horizon_bars: int = 12,
    cost_r: float = 0.05,
) -> PriceForecast:
    """Convert a directional probability into an executable, bounded trade plan.

    This is a deterministic baseline forecast layer. It deliberately produces
    zones and probabilities rather than pretending an exact future price is known.
    A production ML model can replace the probability/level estimates without
    changing the trade-plan contract.
    """
    if price <= 0 or atr < 0:
        raise ValueError("price must be positive and ATR cannot be negative")
    direction = direction.upper()
    if direction not in {"BUY", "SELL"}:
        raise ValueError("direction must be BUY or SELL")
    p = max(0.0, min(1.0, float(probability)))
    unit = max(float(atr), price * 0.0025)
    if direction == "BUY":
        entry_low = price - min(unit * 0.25, price * 0.005)
        entry_high = price + min(unit * 0.10, price * 0.0025)
        if resistance and resistance > price:
            entry_high = min(entry_high, float(resistance))
        invalidation = min(price - unit, support if support and support < price else price - unit)
        target_1 = price + unit * 1.5
        target_2 = price + unit * 2.5
    else:
        entry_low = price - min(unit * 0.10, price * 0.0025)
        entry_high = price + min(unit * 0.25, price * 0.005)
        if support and support < price:
            entry_low = max(entry_low, float(support))
        invalidation = max(price + unit, resistance if resistance and resistance > price else price + unit)
        target_1 = price - unit * 1.5
        target_2 = price - unit * 2.5

    risk = abs(price - invalidation)
    reward_1 = abs(target_1 - price)
    reward_2 = abs(target_2 - price)
    p1 = max(0.0, min(1.0, p))
    p2 = max(0.0, min(p1, p1 * 0.68))
    expected_r = (p1 * reward_1 + p2 * reward_2 - (2.0 - p1 - p2) * risk) / risk - cost_r if risk else -math.inf
    uncertainty = 1.0 - abs(p - 0.5) * 2.0
    return PriceForecast(
        direction=direction,
        entry_low=round(min(entry_low, entry_high), 4),
        entry_high=round(max(entry_low, entry_high), 4),
        invalidation=round(invalidation, 4),
        target_1=round(target_1, 4),
        target_2=round(target_2, 4),
        probability_target_1=round(p1, 4),
        probability_target_2=round(p2, 4),
        expected_r=round(expected_r, 4),
        horizon_bars=max(1, int(horizon_bars)),
        uncertainty=round(uncertainty, 4),
        reason="Probability-driven ATR trade plan; exact future price is not assumed",
    )
