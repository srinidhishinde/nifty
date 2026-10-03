from datafiles.historical.sample.sample_chain import (
    create_sample_chain,
)

from backtest.contract_engine import (
    ContractReplayEngine,
)


def test_contract_replay_generates_trades():

    snapshots = create_sample_chain(
        rows=60,
    )

    engine = ContractReplayEngine(
        starting_capital=300000,
        max_loss_per_trade=15000,
    )

    result = engine.run(
        snapshots,
        hold_bars=3,
    )

    assert len(result.trades) > 0

    assert (
        result.starting_capital
        == 300000
    )


def test_contract_replay_respects_trade_loss_limit():

    snapshots = create_sample_chain(
        rows=60,
    )

    engine = ContractReplayEngine(
        starting_capital=300000,
        max_loss_per_trade=15000,
    )

    result = engine.run(
        snapshots,
        hold_bars=3,
    )

    for trade in result.trades:

        assert (
            trade.net_pnl
            >= -15000
        )


def test_contract_replay_has_valid_win_rate():

    snapshots = create_sample_chain(
        rows=60,
    )

    engine = ContractReplayEngine(
        starting_capital=300000,
        max_loss_per_trade=15000,
    )

    result = engine.run(
        snapshots,
        hold_bars=3,
    )

    assert 0 <= result.win_rate <= 100


def test_contract_replay_has_non_negative_drawdown():

    snapshots = create_sample_chain(
        rows=60,
    )

    engine = ContractReplayEngine(
        starting_capital=300000,
        max_loss_per_trade=15000,
    )

    result = engine.run(
        snapshots,
        hold_bars=3,
    )

    assert result.max_drawdown >= 0
