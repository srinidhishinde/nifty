import pandas as pd

from backtest.validation import (
    AcceptanceCriteria,
    acceptance_report,
    monte_carlo_drawdown,
    stress_trade_pnl,
    walk_forward_splits,
)


def _data(days: int = 100) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2026-01-01 09:15", tz="Asia/Kolkata")
    for d in range(days):
        day = start + pd.Timedelta(days=d)
        if day.weekday() >= 5:
            continue
        for bar in range(4):
            ts = day + pd.Timedelta(minutes=5 * bar)
            price = 100 + d + bar
            rows.append({
                "timestamp": ts,
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.5,
                "volume": 1000,
            })
    return pd.DataFrame(rows)


def _trades(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame({
        "pnl": [100.0] * n,
        "costs": [10.0] * n,
        "quantity": [65] * n,
        "entry_price": [100.0] * n,
        "exit_price": [102.0] * n,
    })


def test_walk_forward_is_chronological_and_embargoed():
    data = _data()
    splits = walk_forward_splits(data, train_days=60, validation_days=20, test_days=20)
    assert splits
    first = splits[0]
    assert first.train_end < first.validation_start < first.test_start
    assert first.purge_bars == 1


def test_stress_never_improves_trade_pnl():
    trades = _trades()
    stressed = stress_trade_pnl(trades, cost_multiplier=2, slippage_bps=10)
    assert (stressed["stressed_pnl"] <= trades["pnl"]).all()


def test_monte_carlo_is_deterministic():
    trades = pd.Series([100.0, -50.0, 75.0] * 20)
    a = monte_carlo_drawdown(trades, 100_000, simulations=500, seed=42)
    b = monte_carlo_drawdown(trades, 100_000, simulations=500, seed=42)
    assert a == b


def test_acceptance_fails_when_100_days_are_missing():
    data = _data(80)
    class Metrics:
        total_trades = 40
        profit_factor = 2.0
        expectancy = 100.0
        max_drawdown_pct = 2.0
    report = acceptance_report(data, _trades(), Metrics(), AcceptanceCriteria())
    assert report["status"] == "FAIL"
    assert "minimum_trading_days" in report["failures"]
