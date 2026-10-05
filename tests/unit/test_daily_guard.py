from risk.daily_guard import enforce_daily_loss


def test_daily_risk_allows_normal_risk():
    d = enforce_daily_loss(100, 2000, 1.0)
    assert d.allowed and d.size_multiplier == 1.0


def test_daily_risk_reduces_size_after_first_breach():
    d = enforce_daily_loss(2100, 2000, 1.0)
    assert d.allowed and d.size_multiplier == 0.5
    assert 15 <= d.cooloff_minutes <= 60


def test_daily_risk_hard_stops_after_second_breach():
    d = enforce_daily_loss(3001, 2000, 1.0)
    assert not d.allowed and d.size_multiplier == 0.0
