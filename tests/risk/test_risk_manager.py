from risk.risk_manager import size_position, calculate_risk_levels

def test_position_size_respects_capital_and_risk():
    d=size_position(100,2,"BUY",300000)
    assert d.allowed
    assert d.quantity>0
    assert d.stop_loss<d.entry<d.take_profit
    assert d.reward_risk>=1.5

def test_invalid_entry_is_rejected():
    d=size_position(0,1,"BUY",300000)
    assert not d.allowed and d.quantity==0

def test_atr_widened_stop_preserves_reward_risk_floor():
    stop_distance, stop, target, rr = calculate_risk_levels(100, 10, "BUY", stop_pct=0.02, atr_multiple=1.5, target_multiple=2.0)
    assert stop_distance == 15
    assert stop == 85
    assert target == 130
    assert rr >= 2.0

def test_excessive_risk_fraction_is_rejected():
    d=size_position(100,1,"BUY",300000,risk_fraction=0.10)
    assert not d.allowed and d.quantity == 0
