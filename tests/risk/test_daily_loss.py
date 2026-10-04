from risk.risk_manager import daily_loss_allowed


def test_daily_loss_blocks_at_limit():
    assert daily_loss_allowed(-30000, 30000) == (False, "Daily loss limit reached")


def test_daily_loss_allows_before_limit():
    assert daily_loss_allowed(-29999, 30000)[0]
