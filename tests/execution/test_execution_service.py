from execution.execution_service import ExecutionService


def test_valid_trade_is_sent_to_paper_broker():

    service = ExecutionService()

    result = service.execute(
        symbol="NIFTY",
        side="BUY",
        entry=100.0,
        stop_loss=80.0,
        lot_size=50,
        daily_loss=0.0,
        trades_today=0,
    )

    assert result.accepted is True
    assert result.paper is True
    assert result.order_id is not None
    assert result.order_id.startswith("PAPER-")
    assert result.quantity > 0


def test_trade_over_loss_limit_is_rejected():

    service = ExecutionService()

    result = service.execute(
        symbol="NIFTY",
        side="BUY",
        entry=1000.0,
        stop_loss=1.0,
        lot_size=50,
        daily_loss=0.0,
        trades_today=0,
    )

    assert result.accepted is False
    assert result.order_id is None
    assert "per-trade" in result.reason.lower()


def test_daily_loss_limit_is_rejected():

    service = ExecutionService()

    result = service.execute(
        symbol="NIFTY",
        side="BUY",
        entry=100.0,
        stop_loss=90.0,
        lot_size=50,
        daily_loss=30000.0,
        trades_today=0,
    )

    assert result.accepted is False
    assert result.order_id is None
    assert "daily loss" in result.reason.lower()


def test_trade_limit_is_rejected():

    service = ExecutionService()

    result = service.execute(
        symbol="NIFTY",
        side="BUY",
        entry=100.0,
        stop_loss=90.0,
        lot_size=50,
        daily_loss=0.0,
        trades_today=15,
    )

    assert result.accepted is False
    assert result.order_id is None
    assert "trade limit" in result.reason.lower()


def test_invalid_lot_size_is_rejected():

    service = ExecutionService()

    result = service.execute(
        symbol="NIFTY",
        side="BUY",
        entry=100.0,
        stop_loss=90.0,
        lot_size=0,
        daily_loss=0.0,
        trades_today=0,
    )

    assert result.accepted is False
    assert result.order_id is None


def test_equal_entry_and_stop_loss_is_rejected():

    service = ExecutionService()

    result = service.execute(
        symbol="NIFTY",
        side="BUY",
        entry=100.0,
        stop_loss=100.0,
        lot_size=50,
        daily_loss=0.0,
        trades_today=0,
    )

    assert result.accepted is False
    assert result.order_id is None
