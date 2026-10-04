from execution.execution_service import ExecutionService

TEST_EQUITY = 100_000.0


def test_valid_trade_is_sent_to_paper_broker():
    result = ExecutionService().execute("NIFTY", "BUY", 100.0, 80.0, 50, 0.0, 0, TEST_EQUITY)
    assert result.accepted is True
    assert result.paper is True
    assert result.order_id and result.order_id.startswith("PAPER-")
    assert result.quantity > 0


def test_trade_over_loss_limit_is_rejected():
    result = ExecutionService().execute("NIFTY", "BUY", 1000.0, 1.0, 50, 0.0, 0, TEST_EQUITY)
    assert result.accepted is False
    assert result.order_id is None
    assert "per-trade" in result.reason.lower()


def test_daily_loss_limit_is_rejected():
    result = ExecutionService().execute("NIFTY", "BUY", 100.0, 90.0, 50, 2_000.0, 0, TEST_EQUITY)
    assert result.accepted is False
    assert result.order_id is None
    assert "daily loss" in result.reason.lower()


def test_trade_limit_is_rejected():
    result = ExecutionService().execute("NIFTY", "BUY", 100.0, 90.0, 50, 0.0, 5, TEST_EQUITY)
    assert result.accepted is False
    assert result.order_id is None
    assert "trade limit" in result.reason.lower()


def test_missing_equity_is_rejected():
    result = ExecutionService().execute("NIFTY", "BUY", 100.0, 90.0, 50, 0.0, 0)
    assert result.accepted is False
    assert "equity" in result.reason.lower()


def test_invalid_lot_size_is_rejected():
    result = ExecutionService().execute("NIFTY", "BUY", 100.0, 90.0, 0, 0.0, 0, TEST_EQUITY)
    assert result.accepted is False


def test_equal_entry_and_stop_loss_is_rejected():
    result = ExecutionService().execute("NIFTY", "BUY", 100.0, 100.0, 50, 0.0, 0, TEST_EQUITY)
    assert result.accepted is False
