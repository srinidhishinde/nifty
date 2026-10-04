import pytest

from strategy.price_forecast import forecast_trade_plan


def test_price_forecast_is_bounded_and_directional():
    plan = forecast_trade_plan(
        price=100.0, atr=2.0, direction="BUY", probability=0.70, support=97.0, resistance=103.0
    )
    assert plan.entry_low <= plan.entry_high
    assert plan.invalidation < 100
    assert plan.target_1 > 100
    assert plan.target_2 > plan.target_1
    assert 0 <= plan.probability_target_2 <= plan.probability_target_1 <= 1


def test_price_forecast_rejects_wait_direction():
    with pytest.raises(ValueError):
        forecast_trade_plan(price=100, atr=2, direction="WAIT", probability=.6)
