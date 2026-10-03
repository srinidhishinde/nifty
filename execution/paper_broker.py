from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class PaperOrder:
    order_id: str
    symbol: str
    direction: str
    quantity: int

    entry: float
    stop_loss: float
    target: float

    status: str = "OPEN"
    created_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class PaperBroker:
    """
    Deterministic in-memory broker for paper trading and tests.

    This class never submits real broker orders.
    """

    def __init__(self):
        self._orders: dict[str, PaperOrder] = {}
        self._next_order_id = 1

    def _generate_order_id(self) -> str:
        order_id = f"PAPER-{self._next_order_id:06d}"
        self._next_order_id += 1
        return order_id

    def place_order(
        self,
        symbol: str,
        direction: str,
        quantity: int,
        entry: float,
        stop_loss: float,
        target: float,
    ) -> PaperOrder:

        if not symbol:
            raise ValueError("symbol is required")

        if direction not in {"CE", "PE"}:
            raise ValueError("direction must be CE or PE")

        if quantity <= 0:
            raise ValueError(
                "quantity must be greater than zero"
            )

        if entry <= 0:
            raise ValueError(
                "entry must be greater than zero"
            )

        if stop_loss <= 0:
            raise ValueError(
                "stop_loss must be greater than zero"
            )

        if target <= 0:
            raise ValueError(
                "target must be greater than zero"
            )

        order = PaperOrder(
            order_id=self._generate_order_id(),
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry=entry,
            stop_loss=stop_loss,
            target=target,
        )

        self._orders[order.order_id] = order

        return order

    def cancel_order(
        self,
        order_id: str,
    ) -> Optional[PaperOrder]:
        """
        Cancel an open order and return the updated order.

        Returns None if the order does not exist.
        Returns the existing order unchanged if it is already closed.
        """

        order = self._orders.get(order_id)

        if order is None:
            return None

        if order.status != "OPEN":
            return order

        order.status = "CANCELLED"
        order.cancelled_at = datetime.now(timezone.utc)

        return order

    def get_order(
        self,
        order_id: str,
    ) -> Optional[PaperOrder]:
        return self._orders.get(order_id)

    def get_open_orders(self) -> list[PaperOrder]:
        return [
            order
            for order in self._orders.values()
            if order.status == "OPEN"
        ]

    def get_all_orders(self) -> list[PaperOrder]:
        return list(self._orders.values())
