import pandas as pd
from features.option_signal_engine import generate_option_chain_signal
from features.option_chain import OptionContract

def contract(side="CE", ltp=100.0):
    return OptionContract(
        symbol=f"NIFTY25000{side}",
        expiry="2026-10-29",
        option_type=side,
        strike=25000,
        ltp=ltp,
        bid=99.5 if ltp > 0 else 0.0,
        ask=100.5 if ltp > 0 else 0.0,
        ltp_change_pct=2.0,
        implied_volatility=15.0,
        open_interest=100000,
        oi_change=5000,
        volume=20000,
        delta=0.5,
        theta=-5.0,
        vega=8.0,
        built_up="Long Buildup",
    )

def test_option_signal_has_trade_plan_percentages():
    signal, rows = generate_option_chain_signal([contract("CE"), contract("PE")],25000)
    assert {"Entry Price","Stop Loss","Take Profit","Max Gain %","Max Loss %"} <= set(rows.columns)
    assert rows["Max Gain %"].notna().all()
    assert rows["Max Loss %"].notna().all()

def test_option_signal_does_not_fabricate_premium_levels():
    signal, rows = generate_option_chain_signal([contract("CE",0)],25000)
    assert pd.isna(rows.iloc[0]["Entry Price"])
    assert pd.isna(rows.iloc[0]["Take Profit"])\n    assert signal.underlying_take_profit is None
