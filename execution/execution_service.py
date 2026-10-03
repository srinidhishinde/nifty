from dataclasses import dataclass

from broker.interface import OrderRequest
from broker.paper import PaperBroker
from risk.risk_engine import calculate_position_size, validate_risk


@dataclass(frozen=True)
class ExecutionResult:
    accepted: bool
    paper: bool
    order_id: str | None
    quantity: int
    reason: str


class ExecutionService:

    def __init__(self):
        self.broker = PaperBroker()

    def execute(
        self,
        symbol: str,
        side: str,
        entry: float,
        stop_loss: float,
        lot_size: int,
        daily_loss: float,
        trades_today: int,
    ) -> ExecutionResult:

        if lot_size <= 0:
            return ExecutionResult(
                accepted=False,
                paper=True,
                order_id=None,
                quantity=0,
                reason="Invalid lot size",
            )

        if entry <= 0 or stop_loss <= 0:
            return ExecutionResult(
                accepted=False,
                paper=True,
                order_id=None,
                quantity=0,
                reason="Entry and stop loss must be positive",
            )

        if entry == stop_loss:
            return ExecutionResult(
                accepted=False,
                paper=True,
                order_id=None,
                quantity=0,
                reason="Entry and stop loss cannot be equal",
            )

        quantity = calculate_position_size(
            entry=entry,
            stop_loss=stop_loss,
            lot_size=lot_size,
        )

        if quantity <= 0:
            return ExecutionResult(
                accepted=False,
                paper=True,
                order_id=None,
                quantity=0,
                reason="Per-trade risk limit would be exceeded",
            )

        estimated_loss = abs(entry - stop_loss) * quantity * lot_size

        if not validate_risk(
            estimated_loss=estimated_loss,
            daily_loss=daily_loss,
            trades_today=trades_today,
        ):
            if estimated_loss > __import__(
                "config.settings",
                fromlist=["settings"],
            ).settings.max_loss_per_trade:
                reason = "Per-trade loss limit exceeded"
            elif daily_loss >= __import__(
                "config.settings",
                fromlist=["settings"],
            ).settings.max_daily_loss:
                reason = "Daily loss limit exceeded"
            else:
                reason = "Trade limit exceeded"

            return ExecutionResult(
                accepted=False,
                paper=True,
                order_id=None,
                quantity=0,
                reason=reason,
            )

        request = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=entry,
            paper=True,
        )

        result = self.broker.place_order(request)

        return ExecutionResult(
            accepted=result.accepted,
            paper=result.paper,
            order_id=result.order_id if result.accepted else None,
            quantity=quantity if result.accepted else 0,
            reason="Accepted" if result.accepted else "Broker rejected order",
        )
