from dataclasses import dataclass
from datetime import datetime


@dataclass
class PaperOrder:

    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: datetime


class PaperBroker:

    def __init__(self):

        self.orders = []

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float
    ):

        order = PaperOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=datetime.now()
        )

        self.orders.append(order)

        return order

    def positions(self):

        return self.orders
