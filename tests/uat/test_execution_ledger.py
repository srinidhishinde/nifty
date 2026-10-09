from execution.execution_ledger import ExecutionLedger

def test_ambiguous_order_cannot_be_retried(tmp_path):
    ledger=ExecutionLedger(tmp_path/"ledger.jsonl")
    ledger.begin("d1","NIFTYCE","BUY",50,"2026-10-09T10:00:00+05:30")
    ledger.mark_ambiguous("d1","timeout","2026-10-09T10:00:01+05:30")
    assert ledger.can_submit("d1") is False


def test_broker_reconciliation_blocks_retry_for_found_order():
    from execution.broker_reconciliation import reconcile_broker_orders, retry_decision_after_timeout
    result = reconcile_broker_orders([{"orderId": "K1", "status": "PARTIALLY_FILLED", "quantity": 50, "filledQuantity": 25, "averagePrice": "101.5"}], expected_order_id="K1", expected_quantity=50, query_authoritative=True)
    assert result.reconciled is True
    assert result.state == "PARTIALLY_FILLED"
    assert result.snapshot.filled_quantity == 25
    assert retry_decision_after_timeout(result) is False

def test_broker_reconciliation_only_retry_when_authoritatively_absent():
    from execution.broker_reconciliation import reconcile_broker_orders, retry_decision_after_timeout
    result = reconcile_broker_orders([], expected_order_id="K1", expected_quantity=50, query_authoritative=True)
    assert result.definitive_absence is True
    assert retry_decision_after_timeout(result) is True

def test_multiple_matching_broker_orders_are_ambiguous():
    from execution.broker_reconciliation import reconcile_broker_orders
    result = reconcile_broker_orders([
        {"orderId": "K1", "status": "COMPLETE", "quantity": 50},
        {"orderId": "K2", "status": "COMPLETE", "quantity": 50},
    ], expected_quantity=50, query_authoritative=True)
    assert result.reconciled is False
    assert result.retry_allowed is False
    assert result.state == "AMBIGUOUS"

def test_malformed_broker_order_state_fails_closed():
    from execution.broker_reconciliation import reconcile_broker_orders
    result = reconcile_broker_orders([{"orderId": "K1", "status": "BROKER_UNKNOWN", "quantity": 50}], expected_order_id="K1", query_authoritative=True)
    assert result.reconciled is False
    assert result.retry_allowed is False
    assert result.state == "AMBIGUOUS"


def test_empty_non_authoritative_query_cannot_authorize_retry():
    from execution.broker_reconciliation import reconcile_broker_orders, retry_decision_after_timeout
    result = reconcile_broker_orders([], expected_order_id="K1", expected_quantity=50)
    assert result.definitive_absence is False
    assert result.retry_allowed is False
    assert retry_decision_after_timeout(result) is False
