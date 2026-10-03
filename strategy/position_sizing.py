from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PositionSize:
    quantity: int
    risk_per_unit: float
    total_risk: float
    capital_used: float


def calculate_position_size(
    capital: float,
    max_loss_per_trade: float,
    entry: float,
    stop_loss: float,
    lot_size: int,
) -> PositionSize:
    """
    Calculate a risk-controlled position size.

    The calculation deliberately keeps a small safety buffer below
    the configured maximum loss so that the requested risk limit is
    not consumed completely.

    Quantity is always a multiple of lot_size.
    """

    if capital <= 0:
        raise ValueError("capital must be greater than zero")

    if max_loss_per_trade <= 0:
        raise ValueError(
            "max_loss_per_trade must be greater than zero"
        )

    if entry <= 0:
        raise ValueError("entry must be greater than zero")

    if stop_loss <= 0:
        raise ValueError("stop_loss must be greater than zero")

    if lot_size <= 0:
        raise ValueError("lot_size must be greater than zero")

    risk_per_unit = abs(entry - stop_loss)

    if risk_per_unit <= 0:
        raise ValueError(
            "entry and stop_loss must be different"
        )

    # Keep 5% of the configured risk budget unused.
    usable_risk = max_loss_per_trade * 0.95

    # Maximum number of individual units allowed by risk.
    max_units_by_risk = math.floor(
        usable_risk / risk_per_unit
    )

    # Convert to complete lots.
    lots = max_units_by_risk // lot_size

    quantity = lots * lot_size

    total_risk = quantity * risk_per_unit
    capital_used = quantity * entry

    # Never allow the position to exceed available capital.
    max_units_by_capital = math.floor(
        capital / entry
    )

    max_lots_by_capital = (
        max_units_by_capital // lot_size
    )

    capital_limited_quantity = (
        max_lots_by_capital * lot_size
    )

    if quantity > capital_limited_quantity:
        quantity = capital_limited_quantity

        total_risk = quantity * risk_per_unit
        capital_used = quantity * entry

    return PositionSize(
        quantity=quantity,
        risk_per_unit=round(risk_per_unit, 2),
        total_risk=round(total_risk, 2),
        capital_used=round(capital_used, 2),
    )
