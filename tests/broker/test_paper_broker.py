import pytest

from broker.interface import OrderRequest
from broker.paper import PaperBroker


def test_paper_order_is_accepted():

    broker = PaperBroker()

    request = OrderRequest(
        symbol="NIFTY",
        side="BUY",
        quantity=50,
        price=100.0,
        paper=True,
    )

    result = broker.place_order(
        request
    )

    assert result.accepted is True

    assert result.paper is True

    assert result.order_id.startswith(
        "PAPER-"
    )


def test_paper_broker_rejects_live_order():

    broker = PaperBroker()

    request = OrderRequest(
        symbol="NIFTY",
        side="BUY",
        quantity=50,
        price=100.0,
        paper=False,
    )

    with pytest.raises(RuntimeError):

        broker.place_order(
            request
        )


def test_paper_broker_rejects_invalid_quantity():

    broker = PaperBroker()

    request = OrderRequest(
        symbol="NIFTY",
        side="BUY",
        quantity=0,
        price=100.0,
        paper=True,
    )

    with pytest.raises(ValueError):

        broker.place_order(
            request
        )