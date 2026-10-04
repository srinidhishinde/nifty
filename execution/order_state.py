from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OrderState(str, Enum):
    NEW = "NEW"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


_ALLOWED: dict[OrderState, set[OrderState]] = {
    OrderState.NEW: {OrderState.SUBMITTED, OrderState.CANCELLED},
    OrderState.SUBMITTED: {OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.REJECTED, OrderState.UNKNOWN},
    OrderState.ACKNOWLEDGED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCEL_PENDING, OrderState.EXPIRED, OrderState.UNKNOWN},
    OrderState.PARTIALLY_FILLED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCEL_PENDING, OrderState.UNKNOWN},
    OrderState.FILLED: set(),
    OrderState.CANCEL_PENDING: {OrderState.CANCELLED, OrderState.FILLED, OrderState.PARTIALLY_FILLED, OrderState.UNKNOWN},
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
    OrderState.EXPIRED: set(),
    OrderState.UNKNOWN: {OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.UNKNOWN},
}


@dataclass
class OrderLifecycle:
    order_id: str
    state: OrderState = OrderState.NEW
    requested_qty: int = 0
    filled_qty: int = 0
    average_fill_price: float | None = None

    def transition(self, new_state: OrderState, *, filled_qty: int | None = None, average_fill_price: float | None = None) -> OrderState:
        if new_state not in _ALLOWED[self.state]:
            raise ValueError(f"Invalid order transition {self.state.value} -> {new_state.value}")
        if filled_qty is not None:
            if filled_qty < 0 or filled_qty > self.requested_qty:
                raise ValueError("filled_qty must be between 0 and requested_qty")
            self.filled_qty = int(filled_qty)
        if average_fill_price is not None:
            if average_fill_price <= 0:
                raise ValueError("average_fill_price must be positive")
            self.average_fill_price = float(average_fill_price)
        if new_state == OrderState.FILLED and self.requested_qty > 0 and self.filled_qty != self.requested_qty:
            raise ValueError("FILLED state requires requested_qty == filled_qty")
        if new_state == OrderState.PARTIALLY_FILLED and not (0 < self.filled_qty < self.requested_qty):
            raise ValueError("PARTIALLY_FILLED requires a partial fill")
        self.state = new_state
        return self.state
