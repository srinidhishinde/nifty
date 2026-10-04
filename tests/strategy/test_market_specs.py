import pytest

from strategy.market_specs import get_option_strike_step, round_to_strike


@pytest.mark.parametrize(
    ("instrument", "step"),
    [
        ("NIFTY", 50.0),
        ("CRUDEOIL", 50.0),
        ("NATURALGAS", 5.0),
        ("COPPER", 10.0),
        ("GOLD", 500.0),
        ("SILVER", 1000.0),
    ],
)
def test_option_strike_intervals_are_instrument_specific(instrument, step):
    assert get_option_strike_step(instrument) == step


def test_round_to_strike_uses_nearest_valid_increment():
    assert round_to_strike(6512.0, 50.0) == 6500.0
    assert round_to_strike(6526.0, 50.0) == 6550.0
    assert round_to_strike(302.0, 5.0) == 300.0
    assert round_to_strike(304.0, 5.0) == 305.0
