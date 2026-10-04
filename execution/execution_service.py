from dataclasses import dataclass

from broker.interface import OrderRequest
from broker.paper import PaperBroker
from config.settings import settings
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
        account_equity: float | None = None,
    ) -> ExecutionResult:
        equity = float(account_equity if account_equity is not None else settings.starting_capital)
        if equity <= 0:
            return ExecutionResult(False, True, None, 0, "Account equity is required")

        if lot_size <= 0:
            return ExecutionResult(False, True, None, 0, "Invalid lot size")
        if entry <= 0 or stop_loss <= 0:
            return ExecutionResult(False, True, None, 0, "Entry and stop loss must be positive")
        if entry == stop_loss:
            return ExecutionResult(False, True, None, 0, "Entry and stop loss cannot be equal")

        quantity = calculate_position_size(entry, stop_loss, lot_size, equity)
        if quantity <= 0:
            return ExecutionResult(False, True, None, 0, "Per-trade risk limit would be exceeded")

        estimated_loss = abs(entry - stop_loss) * quantity
        if not validate_risk(estimated_loss, daily_loss, trades_today, equity):
            if estimated_loss > equity * settings.risk_fraction:
                reason = "Per-trade loss limit exceeded"
            elif daily_loss >= equity * settings.max_daily_loss_fraction:
                reason = "Daily loss limit exceeded"
            else:
                reason = "Trade limit exceeded"
            return ExecutionResult(False, True, None, 0, reason)

        result = self.broker.place_order(OrderRequest(
            symbol=symbol, side=side, quantity=quantity, price=entry, paper=True
        ))
        return ExecutionResult(
            result.accepted, result.paper,
            result.order_id if result.accepted else None,
            quantity if result.accepted else 0,
            "Accepted" if result.accepted else "Broker rejected order",
        )
