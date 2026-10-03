from backtest.costs import (
    TradingCosts,
    apply_buy_cost,
    apply_sell_cost,
    calculate_transaction_cost,
)


def test_buy_slippage_increases_price():

    costs = TradingCosts(
        slippage_pct=0.001
    )

    result = apply_buy_cost(
        100,
        costs,
    )

    assert result == 100.1


def test_sell_slippage_reduces_price():

    costs = TradingCosts(
        slippage_pct=0.001
    )

    result = apply_sell_cost(
        100,
        costs,
    )

    assert result == 99.9


def test_transaction_cost_is_positive():

    costs = TradingCosts()

    result = calculate_transaction_cost(
        entry_price=100,
        exit_price=110,
        quantity=1,
        costs=costs,
    )

    assert result > 0