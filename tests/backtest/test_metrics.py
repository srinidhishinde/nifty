import pandas as pd

from backtest.metrics import calculate_metrics


def test_metrics():

    trades = pd.DataFrame(
        {
            "pnl": [
                100,
                200,
                -50,
                150,
            ]
        }
    )

    metrics = calculate_metrics(
        trades,
        starting_capital=300000,
    )

    assert metrics.total_trades == 4

    assert metrics.winning_trades == 3

    assert metrics.losing_trades == 1

    assert metrics.win_rate_pct == 75.0

    assert metrics.net_pnl == 400

    assert metrics.return_pct > 0