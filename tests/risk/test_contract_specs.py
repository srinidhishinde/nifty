from risk.contract_specs import ContractSpec, lots_for_risk, quantity_for_risk, risk_per_lot


def test_contract_risk_uses_tick_value_and_lot_size():
    spec = ContractSpec("TEST", lot_size=10, tick_size=0.5, tick_value=2.0)
    assert risk_per_lot(1.0, spec) == 40.0
    assert lots_for_risk(100, 1.0, spec) == 2
    assert quantity_for_risk(100, 1.0, spec) == 20
