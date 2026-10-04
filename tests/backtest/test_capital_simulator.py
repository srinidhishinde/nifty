import pandas as pd
import pytest

from backtest.capital_simulator import simulate_capital


def sample_data(days=110):
    rows = []
    for d in pd.bdate_range("2026-01-02", periods=days):
        for minute in (9, 10, 11, 12, 13, 14, 15):
            ts = d + pd.Timedelta(hours=minute)
            price = 100 + (len(rows) * 0.01)
            rows.append({
                "timestamp": ts,
                "open": price,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price,
                "volume": 1000,
                "sentiment": 0.0,
            })
    return pd.DataFrame(rows)


def test_capital_simulator_requires_100_days():
    with pytest.raises(ValueError):
        simulate_capital(sample_data(99), days=100)


def test_capital_simulator_returns_100_day_curve():
    result = simulate_capital(sample_data(110), starting_capital=100_000, days=100)
    assert result.trading_days == 100
    assert len(result.equity_curve) == 100
    assert result.ending_capital >= 0
    assert result.starting_capital == 100_000
