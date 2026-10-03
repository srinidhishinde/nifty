from backtest.data.sample_data import create_sample_data
from backtest.engine import BacktestEngine


def test_backtest_generates_trades():

    data = create_sample_data()

    engine = BacktestEngine(
        starting_capital=300000,
        max_risk_per_trade=15000,
    )

    trades = engine.run(
        data,
        symbol="TEST",
    )

    assert not trades.empty

    assert "pnl" in trades.columns

    assert "direction" in trades.columns


def test_backtest_only_uses_known_data():

    data = create_sample_data()

    engine = BacktestEngine()

    trades = engine.run(
        data,
        symbol="TEST",
    )

    for _, trade in trades.iterrows():

        assert (
            trade["entry_time"]
            <= trade["exit_time"]
        )
