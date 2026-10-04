import pandas as pd

from backtest.capital_aware import CapitalAwareRuleBacktestEngine

TEST_EQUITY = 100_000.0
TEST_RISK_FRACTION = 0.01


def _frame():
    ts = pd.date_range("2026-01-05 09:15", periods=80, freq="5min", tz="Asia/Kolkata")
    close = [25000 + i * 2 for i in range(len(ts))]
    return pd.DataFrame({"timestamp": ts, "open": close, "high": [x + 5 for x in close], "low": [x - 5 for x in close], "close": close, "volume": [1000] * len(ts)})


def test_capital_is_not_used_as_trade_pnl():
    result = CapitalAwareRuleBacktestEngine(starting_capital=TEST_EQUITY, risk_fraction=TEST_RISK_FRACTION, lot_size=65, slippage_points=0, brokerage_per_order=0).run(_frame(), evaluation_start=pd.Timestamp("2035-01-01"))
    assert result.validation["final_equity"] == TEST_EQUITY


def test_engine_records_validation():
    result = CapitalAwareRuleBacktestEngine(starting_capital=TEST_EQUITY, risk_fraction=TEST_RISK_FRACTION, lot_size=65, slippage_points=0, brokerage_per_order=0).run(_frame())
    assert result.validation["status"] == "PASS"
    assert result.validation["starting_capital"] == TEST_EQUITY
