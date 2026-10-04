from features.option_chain import OptionContract
from strategy.btst_option_selector import rank_btst_options


def test_btst_requires_liquidity_and_returns_risk_levels():
    c = OptionContract("NIFTY25000CE", "2026-09-03", 25000, "CE", 100, 99, 101, 10000, 50000, 5000, 18, ltp_change_pct=2)
    signal, table = rank_btst_options([c], "UP")
    assert signal.decision == "BUY_CE_BTST"
    assert signal.entry == 101
    assert signal.stop_loss < signal.entry < signal.target
