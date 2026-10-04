from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class SafetyStatus:
    halted: bool
    reason: str
    orders_last_minute: int
    rejections_today: int
    reconciliation_ok: bool


class SafetyController:
    def __init__(self, max_orders_per_minute: int = 20, max_rejections: int = 5):
        if max_orders_per_minute <= 0 or max_rejections <= 0:
            raise ValueError("safety limits must be positive")
        self.max_orders_per_minute = int(max_orders_per_minute)
        self.max_rejections = int(max_rejections)
        self._submissions: deque[datetime] = deque()
        self.rejections_today = 0
        self.halted = False
        self.reason = ""
        self.reconciliation_ok = True

    def halt(self, reason: str) -> None:
        self.halted = True
        self.reason = reason

    def set_reconciliation(self, ok: bool) -> None:
        self.reconciliation_ok = bool(ok)
        if not ok:
            self.halt("Broker reconciliation mismatch")

    def before_order(self, now: datetime) -> SafetyStatus:
        cutoff = now - timedelta(minutes=1)
        while self._submissions and self._submissions[0] < cutoff:
            self._submissions.popleft()
        if self.halted:
            return self.status()
        if not self.reconciliation_ok:
            self.halt("Broker reconciliation unavailable")
        elif len(self._submissions) >= self.max_orders_per_minute:
            self.halt("Order-rate limit reached")
        return self.status()

    def record_submission(self, now: datetime) -> SafetyStatus:
        self._submissions.append(now)
        return self.before_order(now)

    def record_rejection(self) -> SafetyStatus:
        self.rejections_today += 1
        if self.rejections_today >= self.max_rejections:
            self.halt("Maximum broker rejections reached")
        return self.status()

    def status(self) -> SafetyStatus:
        return SafetyStatus(
            halted=self.halted,
            reason=self.reason,
            orders_last_minute=len(self._submissions),
            rejections_today=self.rejections_today,
            reconciliation_ok=self.reconciliation_ok,
        )
