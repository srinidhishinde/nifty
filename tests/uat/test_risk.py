from config.settings import settings
from risk.risk_engine import validate_risk

TEST_EQUITY = 100_000.0


def test_risk_limit():
    assert validate_risk(15_001, 0, 0, TEST_EQUITY) is False


def test_trade_limit():
    assert validate_risk(1_000, 0, settings.max_trades_per_day, TEST_EQUITY) is False


def test_daily_loss_limit():
    assert validate_risk(1_000, 2_000, 0, TEST_EQUITY) is False


def test_missing_equity_is_rejected():
    assert validate_risk(1, 0, 0, 0) is False
