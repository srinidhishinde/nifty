from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ValidationSnapshot:
    strategy_id: str
    market: str
    timeframe: str
    regime: str
    oos_expectancy_r: float
    profit_factor: float
    max_drawdown_pct: float
    sample_size: int
    calibration_error: float
    validated: bool
    model_version: str
    validated_at: str


class RobustnessRegistry:
    """Read-mostly registry for precomputed, time-ordered validation evidence."""

    def __init__(self, snapshots: Mapping[tuple[str, str, str, str], ValidationSnapshot] | None = None):
        self._snapshots = dict(snapshots or {})

    def put(self, snapshot: ValidationSnapshot) -> None:
        self._snapshots[(snapshot.strategy_id, snapshot.market, snapshot.timeframe, snapshot.regime)] = snapshot

    def lookup(self, strategy_id: str, market: str, timeframe: str, regime: str) -> ValidationSnapshot | None:
        return self._snapshots.get((strategy_id, market, timeframe, regime))

    def approved(self, strategy_id: str, market: str, timeframe: str, regime: str, *, min_expectancy_r: float = 0.0, max_drawdown_pct: float = 20.0) -> bool:
        snapshot = self.lookup(strategy_id, market, timeframe, regime)
        return bool(
            snapshot
            and snapshot.validated
            and snapshot.oos_expectancy_r >= min_expectancy_r
            and snapshot.max_drawdown_pct <= max_drawdown_pct
            and snapshot.sample_size >= 50
        )
