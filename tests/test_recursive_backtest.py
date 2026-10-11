import pandas as pd

from prediction.recursive_backtest import simulate_recursive_low_high


def _bars(rows):
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])


def _predictions(rows):
    return pd.DataFrame(rows, columns=["target_timestamp", "predicted_low", "predicted_high"])


def test_recursive_backtest_exports_net_pnl_and_compounds_capital():
    candles = _bars([
        ("2026-10-10 09:15:00+05:30", 100, 101, 99, 100),
        ("2026-10-10 09:16:00+05:30", 105, 111, 99, 108),
        ("2026-10-10 09:17:00+05:30", 109, 112, 108, 111),
    ])
    forecasts = _predictions([
        ("2026-10-10 09:16:00+05:30", 100, 110),
        ("2026-10-10 09:17:00+05:30", 109, 111),
    ])
    ledger, summary = simulate_recursive_low_high(
        candles, forecasts, initial_capital=30_000, lot_size=10,
        cost_bps_per_side=0, slippage_bps_per_side=0,
    )

    assert len(ledger) == 2
    assert list(ledger["trade"]) == [1, 2]
    assert ledger.iloc[0]["net_pnl"] > 0
    assert ledger.iloc[1]["capital_before"] == ledger.iloc[0]["capital_after"]
    assert summary["ending_capital"] == ledger.iloc[-1]["capital_after"]
    assert summary["net_profit"] == summary["ending_capital"] - 30_000
    assert summary["status"] == "EVALUATED"


def test_stop_loss_wins_when_stop_and_target_both_hit():
    candles = _bars([
        ("2026-10-10 09:15:00+05:30", 100, 101, 99, 100),
        ("2026-10-10 09:16:00+05:30", 100, 111, 90, 105),
    ])
    forecasts = _predictions([
        ("2026-10-10 09:16:00+05:30", 100, 110),
    ])
    ledger, _ = simulate_recursive_low_high(
        candles, forecasts, initial_capital=30_000, lot_size=10,
        cost_bps_per_side=0, slippage_bps_per_side=0,
    )

    assert len(ledger) == 1
    assert ledger.iloc[0]["exit_reason"] == "STOP_FIRST_BOTH_TOUCHED"
    assert ledger.iloc[0]["exit_price"] == 95


def test_no_fill_returns_zero_profit_without_inventing_trades():
    candles = _bars([
        ("2026-10-10 09:15:00+05:30", 100, 101, 99, 100),
        ("2026-10-10 09:16:00+05:30", 105, 109, 102, 108),
    ])
    forecasts = _predictions([
        ("2026-10-10 09:16:00+05:30", 100, 110),
    ])
    ledger, summary = simulate_recursive_low_high(
        candles, forecasts, initial_capital=30_000, lot_size=10,
    )

    assert ledger.empty
    assert summary["status"] == "NO_TRADES"
    assert summary["ending_capital"] == 30_000
    assert summary["net_profit"] == 0
