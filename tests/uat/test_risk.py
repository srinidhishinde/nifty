from risk.risk_engine import validate_risk


def test_risk_limit():

    result = validate_risk(
        estimated_loss=15001,
        daily_loss=0,
        trades_today=0
    )

    assert result is False


def test_trade_limit():

    result = validate_risk(
        estimated_loss=1000,
        daily_loss=0,
        trades_today=15
    )

    assert result is False


def test_daily_loss_limit():

    result = validate_risk(
        estimated_loss=1000,
        daily_loss=30000,
        trades_today=0
    )

    assert result is False
