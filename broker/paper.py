from broker.interface import (
    Broker,
    OrderRequest,
    OrderResult,
)


class PaperBroker:

    """
    Deterministic paper broker.

    This broker NEVER sends orders to an exchange or broker.
    """

    def __init__(self):

        self.orders: list[OrderResult] = []

        self._counter = 0

    def place_order(
        self,
        request: OrderRequest,
    ) -> OrderResult:

        if not request.paper:

            raise RuntimeError(
                "PaperBroker refuses non-paper orders"
            )

        if request.quantity <= 0:

            raise ValueError(
                "Order quantity must be positive"
            )

        self._counter += 1

        order_id = (
            f"PAPER-{self._counter:06d}"
        )

        result = OrderResult(
            accepted=True,
            order_id=order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            price=request.price,
            paper=True,
            message="Paper order accepted",
        )

        self.orders.append(result)

        return result