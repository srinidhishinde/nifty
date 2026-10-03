from execution.paper_broker import PaperBroker
from strategy.position_sizing import (
    calculate_position_size,
)
from strategy.strike_selector import (
    StrikeSelector,
)
from strategy.trade_levels import (
    TradeLevelCalculator,
)


def test_strike_selector_finds_atm():

    levels = StrikeSelector.build_levels(
        [
            24900,
            25000,
            25100,
            25200,
            25300,
        ],
        25040,
    )

    assert levels[0] == (
        25000.0,
        0,
    )


def test_trade_levels_are_calculated():

    levels = TradeLevelCalculator().calculate(
        direction="CE",
        strike=25000,
        entry=100,
    )

    assert levels.entry == 100
    assert levels.stop_loss == 80
    assert levels.target == 140
    assert levels.reward_risk_ratio == 2.0


def test_position_size_respects_risk():

    size = calculate_position_size(
        capital=300000,
        max_loss_per_trade=15000,
        entry=100,
        stop_loss=80,
        lot_size=50,
    )

    assert size.quantity == 700
    assert size.total_risk == 14000


def test_paper_order_does_not_need_broker():

    broker = PaperBroker()

    order = broker.place_order(
        symbol="NIFTY25000CE",
        direction="CE",
        quantity=50,
        entry=100,
        stop_loss=80,
        target=140,
    )

    assert order.status == "OPEN"
    assert order.quantity == 50

    assert (
        broker.get_order(
            order.order_id
        )
        == order
    )


def test_paper_order_can_be_cancelled():

    broker = PaperBroker()

    order = broker.place_order(
        symbol="NIFTY25000CE",
        direction="CE",
        quantity=50,
        entry=100,
        stop_loss=80,
        target=140,
    )

    cancelled = broker.cancel_order(
        order.order_id
    )

    assert cancelled.status == "CANCELLED"
