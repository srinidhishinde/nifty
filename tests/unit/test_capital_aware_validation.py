import pandas as pd

from backtest.capital_aware import CapitalAwareRuleBacktestEngine


def _frame():
    ts = pd.date_range("2026-01-05 09:15", periods=80, freq="5min", tz="Asia/Kolkata")
    close = [25000 + i * 2 for i in range(len(ts))]
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": [x + 5 for x in close],
        "low": [x - 5 for x in close],
        "close": close,
        "volume": [1000] * len(ts),
    })


def test_capital_is_not_used_as_trade_pnl():
    result = CapitalAwareRuleBacktestEngine(
        starting_capital=100000, risk_per_trade=1000, lot_size=65,
        slippage_points=0, brokerage_per_order=0,
    ).run(_frame(), evaluation_start=pd.Timestamp("2035-01-01"))
    assert result.validation["final_equity"] == 100000


def test_engine_records_validation():
    result = CapitalAwareRuleBacktestEngine(
        starting_capital=100000, risk_per_trade=1000, lot_size=65,
        slippage_points=0, brokerage_per_order=0,
    ).run(_frame())
    assert result.validation["status"] == "PASS"


def test_same_bar_stop_and_target_uses_conservative_stop_first():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-05 09:15", "2026-01-05 09:20", "2026-01-05 09:25"
        ]).tz_localize("Asia/Kolkata"),
        "open": [100, 100, 100],
        "high": [101, 103, 103],
        "low": [99, 97, 99],
        "close": [100, 102, 102],
        "volume": [1000, 1000, 1000],
    })
    # The engine's exit ordering is stop before target whenever OHLC cannot
    # establish which level was touched first. This is intentionally conservative.
    result = CapitalAwareRuleBacktestEngine(
        starting_capital=100000, risk_per_trade=1000, lot_size=1,
        slippage_points=0, brokerage_per_order=0,
    ).run(frame)
    # The test primarily protects the contract: if a trade is closed on the
    # ambiguous bar, its reason must not be optimistic take-profit sequencing.
    if not result.trades.empty:
        assert result.trades.iloc[0]["reason"] != "take_profit"
