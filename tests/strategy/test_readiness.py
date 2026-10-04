from strategy.readiness import assess_readiness


def test_readiness_blocks_without_test_evidence():
    result = assess_readiness(
        tests_passed=False,
        warmup_ready=True,
        risk_engine_ready=True,
        ml_available=True,
        live_order_enabled=False,
        realistic_backtest_available=True,
        option_premium_history_available=False,
    )
    assert result.status == "BLOCKED"
    assert result.score < 100
