from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PortfolioRiskSnapshot:
    proposed_risk: float
    open_risk: float
    pending_risk: float
    total_risk: float
    max_risk: float
    exposure_pct: float
    allowed: bool
    reason: str


class PortfolioRiskEngine:
    def __init__(self, capital: float, max_portfolio_risk_pct: float = 5.0, max_exposure_pct: float = 50.0):
        if capital <= 0:
            raise ValueError("capital must be positive")
        if max_portfolio_risk_pct <= 0 or max_exposure_pct <= 0:
            raise ValueError("portfolio risk limits must be positive")
        self.capital = float(capital)
        self.max_portfolio_risk = self.capital * max_portfolio_risk_pct / 100.0
        self.max_exposure = self.capital * max_exposure_pct / 100.0

    def validate(
        self,
        *,
        proposed_risk: float,
        open_risk: float = 0.0,
        pending_risk: float = 0.0,
        proposed_exposure: float = 0.0,
        existing_exposure: float = 0.0,
    ) -> PortfolioRiskSnapshot:
        total_risk = max(0.0, float(proposed_risk)) + max(0.0, float(open_risk)) + max(0.0, float(pending_risk))
        exposure = max(0.0, float(proposed_exposure)) + max(0.0, float(existing_exposure))
        allowed = total_risk <= self.max_portfolio_risk and exposure <= self.max_exposure
        if total_risk > self.max_portfolio_risk:
            reason = "Portfolio risk budget exceeded."
        elif exposure > self.max_exposure:
            reason = "Portfolio exposure limit exceeded."
        else:
            reason = "Portfolio risk and exposure are within configured limits."
        return PortfolioRiskSnapshot(
            proposed_risk=round(float(proposed_risk), 2),
            open_risk=round(float(open_risk), 2),
            pending_risk=round(float(pending_risk), 2),
            total_risk=round(total_risk, 2),
            max_risk=round(self.max_portfolio_risk, 2),
            exposure_pct=round(exposure / self.capital * 100.0, 3),
            allowed=allowed,
            reason=reason,
        )


def correlation_group_risk(positions: Iterable[tuple[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for group, risk in positions:
        result[group] = result.get(group, 0.0) + max(0.0, float(risk))
    return result
