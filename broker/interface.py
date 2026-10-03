from dataclasses import dataclass
from typing import Literal, Protocol


OrderSide = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class OrderRequest:

    symbol: str

    side: OrderSide

    quantity: int

    price: float | None = None

    paper: bool = True


@dataclass(frozen=True)
class OrderResult:

    accepted: bool

    order_id: str

    symbol: str

    side: OrderSide

    quantity: int

    price: float | None

    paper: bool

    message: str


class Broker(Protocol):

    def place_order(
        self,
        request: OrderRequest,
    ) -> OrderResult:
        ...
