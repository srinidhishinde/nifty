from risk.risk_manager import size_position

def test_position_size_respects_capital_and_risk():
    d=size_position(100,2,"BUY",300000)
    assert d.allowed
    assert d.quantity>0
    assert d.stop_loss<d.entry<d.take_profit
    assert d.reward_risk>=1.5

def test_invalid_entry_is_rejected():
    d=size_position(0,1,"BUY",300000)
    assert not d.allowed and d.quantity==0
