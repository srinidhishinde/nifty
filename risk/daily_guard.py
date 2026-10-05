from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class DailyRiskDecision:
    allowed: bool
    size_multiplier: float
    cooloff_minutes: int
    reason: str


def enforce_daily_loss(
    daily_loss: float,
    daily_loss_cap: float,
    volatility_band: float = 1.0,
    now: datetime | None = None,
    cooloff_until: datetime | None = None,
) -> DailyRiskDecision:
    if daily_loss_cap <= 0:
        raise ValueError("daily_loss_cap must be positive")
    loss = max(0.0, float(daily_loss))
    if cooloff_until is not None and now is not None and now < cooloff_until:
        return DailyRiskDecision(False, 0.0, max(0, int((cooloff_until - now).total_seconds() / 60)), "cooloff_active")
    if loss < daily_loss_cap:
        return DailyRiskDecision(True, 1.0, 0, "within_daily_loss_cap")
    if loss < daily_loss_cap * 1.5:
        minutes = max(15, min(60, round(max(0.0, float(volatility_band)) * 20)))
        return DailyRiskDecision(True, 0.5, minutes, "first_daily_loss_breach_reduced_size")
    return DailyRiskDecision(False, 0.0, 0, "hard_stop_after_second_daily_loss_breach")
