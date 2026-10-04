from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DailyObjectiveState:
    start_equity: float
    realized_pnl: float
    unrealized_pnl: float
    target_pct: float
    loss_limit_pct: float
    net_return_pct: float
    remaining_target_pct: float
    target_reached: bool
    loss_limit_reached: bool
    new_entries_allowed: bool
    reason: str


class DailyObjectiveController:
    """Tracks a daily objective without increasing risk to chase the objective."""

    def __init__(self, start_equity: float, target_pct: float = 30.0, loss_limit_pct: float = 2.0) -> None:
        if start_equity <= 0:
            raise ValueError("start_equity must be positive")
        if target_pct < 0:
            raise ValueError("target_pct cannot be negative")
        if loss_limit_pct <= 0:
            raise ValueError("loss_limit_pct must be positive")
        self.start_equity = float(start_equity)
        self.target_pct = float(target_pct)
        self.loss_limit_pct = float(loss_limit_pct)
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

    def update(self, *, realized_pnl: float | None = None, unrealized_pnl: float | None = None) -> DailyObjectiveState:
        if realized_pnl is not None:
            self.realized_pnl = float(realized_pnl)
        if unrealized_pnl is not None:
            self.unrealized_pnl = float(unrealized_pnl)
        return self.state()

    def state(self) -> DailyObjectiveState:
        net_pnl = self.realized_pnl + self.unrealized_pnl
        net_return_pct = net_pnl / self.start_equity * 100.0
        target_reached = net_return_pct >= self.target_pct
        loss_limit_reached = net_return_pct <= -self.loss_limit_pct
        remaining = max(0.0, self.target_pct - net_return_pct)
        if target_reached:
            reason = "Daily objective reached; block new entries."
        elif loss_limit_reached:
            reason = "Daily loss limit reached; halt new entries."
        else:
            reason = "Normal operation; evaluate opportunities using standard risk limits."
        return DailyObjectiveState(
            start_equity=self.start_equity,
            realized_pnl=round(self.realized_pnl, 2),
            unrealized_pnl=round(self.unrealized_pnl, 2),
            target_pct=self.target_pct,
            loss_limit_pct=self.loss_limit_pct,
            net_return_pct=round(net_return_pct, 3),
            remaining_target_pct=round(remaining, 3),
            target_reached=target_reached,
            loss_limit_reached=loss_limit_reached,
            new_entries_allowed=not target_reached and not loss_limit_reached,
            reason=reason,
        )
