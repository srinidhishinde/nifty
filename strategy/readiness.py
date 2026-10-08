from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReadinessGate:
    name: str
    passed: bool
    severity: str
    detail: str
    points: int = 5


@dataclass(frozen=True)
class SystemReadiness:
    score: float
    status: str
    gates: tuple[ReadinessGate, ...]

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(g.name + ": " + g.detail for g in self.gates if not g.passed)

    @property
    def all_passed(self) -> bool:
        return self.score == 100.0 and not self.blockers


def assess_readiness(
    *,
    tests_passed: bool,
    warmup_ready: bool,
    risk_engine_ready: bool,
    ml_available: bool,
    live_order_enabled: bool,
    realistic_backtest_available: bool,
    option_premium_history_available: bool,
    broker_connected: bool = False,
    source_integrity: bool = False,
    no_synthetic_fallback: bool = False,
    underlying_quote_fresh: bool = False,
    candle_fresh: bool = False,
    data_quality_ready: bool = False,
    option_chain_present: bool = False,
    option_quotes_fresh: bool = False,
    option_liquidity_ready: bool = False,
    option_flow_ready: bool = False,
    regime_ready: bool = False,
    ensemble_ready: bool = False,
    trade_plan_ready: bool = False,
    risk_budget_ready: bool = False,
    risk_controls_ready: bool = False,
    duplicate_guard_ready: bool = False,
    journal_ready: bool = False,
    execution_ready: bool = False,
    reconciliation_ready: bool = False,
    live_approval: bool = False,
) -> SystemReadiness:
    """
    Evidence-only 100-point production gate.

    Exactly twenty gates are worth five points each. Missing evidence is a
    failure, never an inferred pass. A score is a diagnostic, not permission
    to trade. LIVE is possible only when every gate passes and the explicit
    live execution approval/configuration is enabled.

    The default values intentionally fail closed so older callers cannot
    accidentally turn an incomplete system into a ready one.
    """
    gates = (
        ReadinessGate("Automated regression/UAT", tests_passed, "P0",
                      "Repository-local regression/UAT evidence is required."),
        ReadinessGate("Broker connectivity", broker_connected, "P0",
                      "Current Kotak Neo connection must be confirmed."),
        ReadinessGate("Source integrity", source_integrity, "P0",
                      "Decision inputs must be traceable to the declared broker/capture source."),
        ReadinessGate("No synthetic production fallback", no_synthetic_fallback, "P0",
                      "Synthetic/Yahoo substitution must be impossible on the decision path."),
        ReadinessGate("Fresh underlying quote", underlying_quote_fresh, "P0",
                      "The current underlying quote must have a verified broker timestamp."),
        ReadinessGate("Fresh completed candles", candle_fresh, "P0",
                      "Decision candles must be real captured Kotak candles within the freshness budget."),
        ReadinessGate("OHLCV quality gate", data_quality_ready, "P0",
                      "No invalid/duplicate/cadence-breaking decision data may pass."),
        ReadinessGate("Real option chain", option_chain_present, "P0",
                      "A real broker CE/PE chain must be present for an option trade."),
        ReadinessGate("Fresh two-sided option quotes", option_quotes_fresh, "P0",
                      "Executable option quotes require verified freshness and bid/ask."),
        ReadinessGate("Option liquidity", option_liquidity_ready, "P0",
                      "Volume, OI and spread must satisfy the configured liquidity rules."),
        ReadinessGate("Option-flow/OI confirmation", option_flow_ready, "P0",
                      "OI/PCR confirmation must be present and finite."),
        ReadinessGate("Indicator warm-up", warmup_ready, "P0",
                      "All required indicators must be warmed up on completed candles."),
        ReadinessGate("Regime classification", regime_ready, "P1",
                      "A valid market regime must be derived from real data."),
        ReadinessGate("ML advisory validation", ml_available, "P1",
                      "A trained, validated model must be available or the ML layer must abstain."),
        ReadinessGate("CE/PE ensemble agreement", ensemble_ready, "P1",
                      "Rule, regime, ML and option evidence must agree above the configured edge."),
        ReadinessGate("Executable option trade plan", trade_plan_ready, "P0",
                      "Entry, stop, target, spread and reward/risk must be executable from live quotes."),
        ReadinessGate("Explicit risk budget", risk_budget_ready, "P0",
                      "Account equity/capital must be supplied explicitly; no hardcoded capital is allowed."),
        ReadinessGate("Risk controls", risk_controls_ready, "P0",
                      "Per-trade, daily-loss, exposure, position-count and kill-switch controls must pass."),
        ReadinessGate("Duplicate/cooldown protection", duplicate_guard_ready, "P0",
                      "Repeated signals/orders must be suppressed deterministically."),
        ReadinessGate("Journal + execution/reconciliation evidence", journal_ready and execution_ready and reconciliation_ready, "P0",
                      "Every decision/order state must be durably journaled and reconciled."),
    )

    score = round(sum(g.points for g in gates if g.passed) / 100.0 * 100.0, 1)
    blockers = [g for g in gates if not g.passed]
    if blockers:
        p0_failed = any(g.severity == "P0" for g in blockers)
        status = "BLOCKED" if p0_failed else "RESEARCH_ONLY"
    elif not live_order_enabled or not live_approval:
        status = "PAPER_READY"
    else:
        status = "LIVE_READY"

    return SystemReadiness(score, status, gates)
