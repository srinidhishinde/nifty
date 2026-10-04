from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadinessGate:
    name: str
    passed: bool
    severity: str
    detail: str


@dataclass(frozen=True)
class SystemReadiness:
    score: float
    status: str
    gates: tuple[ReadinessGate, ...]


def assess_readiness(
    *,
    tests_passed: bool,
    warmup_ready: bool,
    risk_engine_ready: bool,
    ml_available: bool,
    live_order_enabled: bool,
    realistic_backtest_available: bool,
    option_premium_history_available: bool,
) -> SystemReadiness:
    gates = (
        ReadinessGate("Regression/UAT", tests_passed, "P0", "All automated suites must pass before merge."),
        ReadinessGate("Indicator warm-up", warmup_ready, "P0", "Signals must abstain until required features are ready."),
        ReadinessGate("Unified risk engine", risk_engine_ready, "P0", "Live and backtest SL/TP/sizing must use the same risk model."),
        ReadinessGate("ML availability", ml_available, "P0", "ML must either provide a valid prediction or abstain."),
        ReadinessGate("Realistic OHLCV backtest", realistic_backtest_available, "P0", "Cost/slippage-aware walk-forward evidence is required."),
        ReadinessGate("Historical option premium replay", option_premium_history_available, "P1", "Required for honest option-premium P&L validation."),
        ReadinessGate("Live order submission", not live_order_enabled, "P0", "Live order submission remains disabled until production validation is complete."),
    )
    weights = {"P0": 2.0, "P1": 1.0}
    total = sum(weights[g.severity] for g in gates)
    earned = sum(weights[g.severity] for g in gates if g.passed)
    score = round(earned / total * 100.0, 1) if total else 0.0
    p0_failed = any(g.severity == "P0" and not g.passed for g in gates)
    status = "BLOCKED" if p0_failed else "PAPER_READY" if score >= 85 else "RESEARCH_READY"
    return SystemReadiness(score, status, gates)
