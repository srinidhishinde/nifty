import pandas as pd

from backtest.capital_aware import CapitalAwareRuleBacktestEngine
from strategy.rules import StrategyConfig, evaluate_rules

TEST_EQUITY = 100_000.0
TEST_RISK_FRACTION = 0.01


def test_symmetric_momentum_rules_emit_bearish_signals():
    row = pd.Series({"close": 100.0, "volume": 2000.0, "VOLUME_MA20": 1000.0, "RSI": 65.0, "MACD": -1.0, "MACD_SIGNAL": -0.5, "EMA9": 98.0, "EMA21": 100.0, "ADX": 25.0, "DI_PLUS": 10.0, "DI_MINUS": 20.0, "candlestick_score": -2.0})
    previous = row.copy()
    previous.update({"RSI": 75.0, "MACD": -0.2, "MACD_SIGNAL": -0.5, "EMA9": 101.0, "EMA21": 100.0, "ADX": 24.0, "DI_PLUS": 20.0, "DI_MINUS": 10.0})
    directions = {(s.rule, s.direction) for s in evaluate_rules(row, previous, StrategyConfig())}
    assert ("rsi_overbought_sell", "SELL") in directions
    assert ("macd_bearish", "SELL") in directions
    assert ("adx_trend_confirmation", "SELL") in directions


def _flat_data():
    return pd.DataFrame({"timestamp": pd.date_range("2026-01-05 09:15", periods=80, freq="5min", tz="Asia/Kolkata"), "open": [25000.0] * 80, "high": [25010.0] * 80, "low": [24990.0] * 80, "close": [25000.0] * 80, "volume": [1000.0] * 80})


def test_minimum_reward_risk_is_enforced():
    result = CapitalAwareRuleBacktestEngine(starting_capital=TEST_EQUITY, risk_fraction=TEST_RISK_FRACTION, lot_size=65, slippage_points=0, brokerage_per_order=0).run(_flat_data())
    assert result.validation["starting_capital"] == TEST_EQUITY


def test_capital_aware_backtest_exposes_rejection_funnel():
    result = CapitalAwareRuleBacktestEngine(starting_capital=TEST_EQUITY, risk_fraction=TEST_RISK_FRACTION, lot_size=65, slippage_points=0, brokerage_per_order=0).run(_flat_data())
    assert "qualified_signal_bars" in result.validation
    assert "rejected_risk_budget" in result.validation
    assert "bars_considered" in result.validation


def test_spot_signal_research_produces_point_and_r_outcomes_without_futures_pnl():
    from backtest.signal_research import run_signal_research
    rows = []
    ts = pd.date_range("2026-01-05 09:15", periods=100, freq="5min", tz="Asia/Kolkata")
    for i, stamp in enumerate(ts):
        p = 25000.0 + i * 4
        rows.append({"timestamp": stamp, "open": p, "high": p + 20, "low": p - 20, "close": p + 4, "volume": 5000 + (i % 10) * 100})
    result = run_signal_research(pd.DataFrame(rows))
    assert result.validation["mode"] == "NIFTY_SPOT_SIGNAL_RESEARCH"
    assert "total_R" in result.validation
    assert "pnl" not in result.signals.columns


def test_signal_research_conservative_exit_resolution():
    from backtest.signal_research import _resolve_outcome
    frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-05 09:15", periods=2, freq="5min", tz="Asia/Kolkata"), "open": [100.0, 100.0], "high": [100.0, 103.0], "low": [100.0, 97.0], "close": [100.0, 100.0], "volume": [1000, 1000]})
    price, reason, _ = _resolve_outcome(frame, 1, "BUY", 98.0, 102.0)
    assert price == 98.0 and reason == "stop_loss"


def test_spot_signal_research_rejects_overlapping_signals(monkeypatch):
    from backtest.signal_research import run_signal_research
    from strategy.rules import StrategyConfig, StrategySignal
    monkeypatch.setattr(
        "backtest.signal_research._generate_signal_from_enriched",
        lambda frame, config: StrategySignal("BUY", ("ema_cross", "adx_trend_confirmation"), ("test", "test"), 98.0, 102.0, 1.0, 80.0, True),
    )
    ts = pd.date_range("2026-01-05 09:15", periods=8, freq="5min", tz="Asia/Kolkata")
    data = pd.DataFrame({"timestamp": ts, "open": [100.0] * len(ts), "high": [100.5] * len(ts), "low": [99.5] * len(ts), "close": [100.0] * len(ts), "volume": [1000.0] * len(ts)})
    result = run_signal_research(data, config=StrategyConfig(min_history_bars=1))
    assert result.validation["rejected_overlap"] > 0
