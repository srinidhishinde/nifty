from strategy.readiness import assess_readiness


def _complete_kwargs():
    return dict(
        tests_passed=True,
        warmup_ready=True,
        risk_engine_ready=True,
        ml_available=True,
        live_order_enabled=True,
        realistic_backtest_available=True,
        option_premium_history_available=True,
        broker_connected=True,
        source_integrity=True,
        no_synthetic_fallback=True,
        underlying_quote_fresh=True,
        candle_fresh=True,
        data_quality_ready=True,
        option_chain_present=True,
        option_quotes_fresh=True,
        option_liquidity_ready=True,
        option_flow_ready=True,
        regime_ready=True,
        ensemble_ready=True,
        trade_plan_ready=True,
        risk_budget_ready=True,
        risk_controls_ready=True,
        duplicate_guard_ready=True,
        journal_ready=True,
        execution_ready=True,
        reconciliation_ready=True,
        live_approval=True,
    )


def test_readiness_blocks_without_test_evidence():
    result = assess_readiness(**{**_complete_kwargs(), "tests_passed": False})
    assert result.status == "BLOCKED"
    assert result.score == 95.0


def test_readiness_is_exactly_100_only_with_all_evidence():
    result = assess_readiness(**_complete_kwargs())
    assert result.score == 100.0
    assert result.status == "LIVE_READY"
    assert result.all_passed


def test_readiness_never_infers_live_permission():
    result = assess_readiness(**{**_complete_kwargs(), "live_order_enabled": False, "live_approval": False})
    assert result.score == 100.0
    assert result.status == "PAPER_READY"


def test_missing_live_evidence_fails_closed():
    result = assess_readiness(**{**_complete_kwargs(), "option_quotes_fresh": False})
    assert result.status == "BLOCKED"
    assert "Fresh two-sided option quotes" in result.blockers[0]
