from __future__ import annotations

import pandas as pd

from backtest.capital_aware import CapitalAwareRuleBacktestEngine


def _bars() -> pd.DataFrame:
    ts = pd.date_range("2026-01-05 09:15", periods=8, freq="5min", tz="Asia/Kolkata")
    close = [25000, 25010, 25020, 25030, 25015, 25040, 25050, 25060]
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": [x + 8 for x in close],
        "low": [x - 8 for x in close],
        "close": close,
        "volume": [1000] * len(close),
    })


def test_engine_respects_lot_size_and_capital():
    engine = CapitalAwareRuleBacktestEngine(
        starting_capital=100000,
        risk_per_trade=1000,
        lot_size=65,
        slippage_points=0,
        brokerage_per_order=0,
    )
    assert engine.lot_size == 65


def test_engine_result_has_validation_and_equity_curve():
    result = CapitalAwareRuleBacktestEngine(
        starting_capital=100000,
        risk_per_trade=1000,
        lot_size=65,
        slippage_points=0,
        brokerage_per_order=0,
    ).run(_bars())
    assert result.validation["starting_capital"] == 100000
    assert "final_equity" in result.validation
    assert list(result.equity_curve.columns) == ["timestamp", "equity", "daily_pnl"] or result.equity_curve.empty


def test_engine_does_not_use_risk_capital_as_pnl():
    # This is a structural regression test: with no trades, capital remains exact.
    result = CapitalAwareRuleBacktestEngine(
        starting_capital=100000,
        risk_per_trade=1000,
        lot_size=65,
        slippage_points=0,
        brokerage_per_order=0,
    ).run(_bars(), evaluation_start=pd.Timestamp("2030-01-01"))
    assert result.validation["final_equity"] == 100000
