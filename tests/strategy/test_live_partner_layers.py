import pandas as pd
import pytest

from strategy.session_manager import SessionState, resolve_session
from strategy.cross_market_trend import calculate_trend
from strategy.daily_objective import DailyObjectiveController
from strategy.trade_quality import calculate_expected_r, evaluate_trade_quality
from strategy.portfolio_risk import PortfolioRiskEngine
from execution.order_state import OrderLifecycle, OrderState
from strategy.safety_controller import SafetyController


def candles(n=80, trend=1.0):
    close = [100.0 + trend * i for i in range(n)]
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-10-01 09:15", periods=n, freq="5min"),
        "open": close,
        "high": [x + 1 for x in close],
        "low": [x - 1 for x in close],
        "close": close,
        "volume": [1000.0] * n,
    })


def test_session_has_entry_and_exit_only_windows():
    entry = resolve_session("NIFTY", pd.Timestamp("2026-10-01 14:30"))
    exit_only = resolve_session("NIFTY", pd.Timestamp("2026-10-01 15:20"))
    assert entry.state == SessionState.ENTRY
    assert exit_only.state == SessionState.EXIT_ONLY
    assert not exit_only.entry_allowed
    assert exit_only.exits_allowed


def test_trend_model_needs_history_and_detects_uptrend():
    insufficient = calculate_trend("NIFTY", candles(20))
    assert insufficient.direction == "DATA_INSUFFICIENT"
    strong = calculate_trend("NIFTY", candles(100, 0.5))
    assert strong.direction in {"UP", "STRONG_UP"}


def test_daily_objective_never_increases_risk_to_chase_target():
    controller = DailyObjectiveController(300000, target_pct=30, loss_limit_pct=2)
    state = controller.update(realized_pnl=90000)
    assert state.target_reached
    assert not state.new_entries_allowed


def test_expected_r_is_net_of_costs():
    assert calculate_expected_r(0.6, 2.0, 1.0, 0.1) == pytest.approx(0.5)


def test_trade_quality_rejects_weak_expectancy():
    result = evaluate_trade_quality(
        probability_win=0.55,
        avg_win_r=1.5,
        avg_loss_r=1.0,
        cost_r=0.2,
        trend_alignment=10,
        rule_agreement_pct=55,
        regime_tradable=True,
        robustness_score=0.8,
        uncertainty=20,
        liquidity_score=90,
    )
    assert not result.approved


def test_portfolio_risk_includes_existing_and_pending_risk():
    engine = PortfolioRiskEngine(300000, max_portfolio_risk_pct=5)
    result = engine.validate(proposed_risk=5000, open_risk=8000, pending_risk=3000)
    assert not result.allowed
    assert result.total_risk == 16000


def test_order_lifecycle_rejects_invalid_fill_state():
    order = OrderLifecycle("O1", requested_qty=100)
    order.transition(OrderState.SUBMITTED)
    with pytest.raises(ValueError):
        order.transition(OrderState.FILLED, filled_qty=50)


def test_safety_controller_halts_after_rejections():
    safety = SafetyController(max_rejections=2)
    safety.record_rejection()
    status = safety.record_rejection()
    assert status.halted
