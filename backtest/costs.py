from dataclasses import dataclass


@dataclass(frozen=True)
class TradingCosts:
    """
    Simple configurable transaction-cost model.

    All values are expressed as currency per executed unit unless
    otherwise specified.
    """

    brokerage_per_order: float = 20.0
    slippage_pct: float = 0.0005
    exchange_cost_pct: float = 0.0001


def apply_buy_cost(price: float, costs: TradingCosts) -> float:
    """
    Effective buy price after modeled slippage.
    """
    return price * (1.0 + costs.slippage_pct)


def apply_sell_cost(price: float, costs: TradingCosts) -> float:
    """
    Effective sell price after modeled slippage.
    """
    return price * (1.0 - costs.slippage_pct)


def calculate_transaction_cost(
    entry_price: float,
    exit_price: float,
    quantity: int,
    costs: TradingCosts,
) -> float:

    turnover = (
        abs(entry_price * quantity)
        + abs(exit_price * quantity)
    )

    exchange_cost = turnover * costs.exchange_cost_pct

    brokerage = costs.brokerage_per_order * 2

    return exchange_cost + brokerage