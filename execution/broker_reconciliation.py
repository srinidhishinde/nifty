from __future__ import annotations

"""Fail-closed broker execution reconciliation.

This module deliberately contains no order-submission side effects. It converts
broker order/trade/position responses into a deterministic reconciliation
decision so an ambiguous submission can never be retried without broker proof.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ACTIVE_STATES = {"CREATED", "SUBMITTED", "ACKNOWLEDGED", "PARTIALLY_FILLED", "CANCEL_PENDING", "AMBIGUOUS"}


@dataclass(frozen=True)
class BrokerExecutionSnapshot:
    found: bool
    order_id: str | None
    state: str
    quantity: int
    filled_quantity: int
    average_price: float | None = None
    remaining_quantity: int | None = None
    broker_timestamp: Any | None = None
    source: str = "KOTAK_NEO"
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    reconciled: bool
    definitive_absence: bool
    retry_allowed: bool
    state: str
    reason: str
    snapshot: BrokerExecutionSnapshot | None = None


def _first(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) not in (None, ""):
            return row[key]
    return None


def _int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if parsed == parsed and abs(parsed) != float("inf") else None
    except (TypeError, ValueError):
        return None


def normalize_order(row: Mapping[str, Any]) -> BrokerExecutionSnapshot:
    order_id = _first(row, "orderId", "order_id", "nOrdNo", "orderNumber")
    status = str(_first(row, "status", "orderStatus", "ordStatus", "stat") or "").strip().upper()
    mapping = {
        "OPEN": "ACKNOWLEDGED",
        "PENDING": "SUBMITTED",
        "TRIGGER_PENDING": "SUBMITTED",
        "OPEN_PENDING": "SUBMITTED",
        "PARTIAL": "PARTIALLY_FILLED",
        "PARTIALLY_FILLED": "PARTIALLY_FILLED",
        "COMPLETE": "FILLED",
        "FILLED": "FILLED",
        "TRADED": "FILLED",
        "CANCELLED": "CANCELLED",
        "REJECTED": "REJECTED",
    }
    state = mapping.get(status, status if status in ACTIVE_STATES | {"CANCELLED", "REJECTED", "FILLED"} else "AMBIGUOUS")
    quantity = _int(_first(row, "quantity", "qty", "orderQty", "order_quantity"))
    filled = _int(_first(row, "filledQuantity", "filledQty", "fillQty", "tradedQuantity"))
    remaining = _int(_first(row, "remainingQuantity", "remainingQty", "unfilledQuantity"), max(quantity - filled, 0))
    return BrokerExecutionSnapshot(
        found=True,
        order_id=str(order_id) if order_id is not None else None,
        state=state,
        quantity=quantity,
        filled_quantity=filled,
        average_price=_float(_first(row, "averagePrice", "avgPrice", "avg_price", "tradePrice")),
        remaining_quantity=remaining,
        broker_timestamp=_first(row, "timestamp", "orderTime", "updateTime", "updatedAt"),
        raw=row,
    )


def reconcile_broker_orders(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    expected_order_id: str | None = None,
    expected_symbol: str | None = None,
    expected_quantity: int | None = None,
    query_authoritative: bool = False,
) -> ReconciliationResult:
    if rows is None or query_authoritative is not True:
        return ReconciliationResult(
            False, False, False, "AMBIGUOUS",
            "Broker order query is missing or has not been proven complete and authoritative.",
        )
    matches = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        snap = normalize_order(row)
        if expected_order_id and snap.order_id != str(expected_order_id):
            continue
        if expected_symbol:
            symbol = str(_first(row, "symbol", "tradingSymbol", "trdSymbol", "pTrdSymbol") or "").strip()
            if symbol and symbol.upper() != str(expected_symbol).strip().upper():
                continue
        if expected_quantity and snap.quantity and snap.quantity != int(expected_quantity):
            continue
        matches.append(snap)
    if not matches:
        return ReconciliationResult(False, True, True, "NOT_FOUND", "Broker query authoritatively found no matching order.")
    if len(matches) > 1:
        return ReconciliationResult(False, False, False, "AMBIGUOUS", "Multiple broker orders matched the decision.")
    snap = matches[0]
    if snap.state in {"FILLED", "PARTIALLY_FILLED", "ACKNOWLEDGED", "SUBMITTED", "CANCELLED", "REJECTED"}:
        return ReconciliationResult(True, False, False, snap.state, "Broker order state reconciled.", snap)
    return ReconciliationResult(False, False, False, "AMBIGUOUS", "Broker returned an unrecognized order state.", snap)


def retry_decision_after_timeout(result: ReconciliationResult) -> bool:
    """Only a definitive broker absence can authorize a retry."""
    return bool(result.definitive_absence and result.retry_allowed and not result.reconciled and result.state == "NOT_FOUND")
