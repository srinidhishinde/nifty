import pandas as pd
from features.option_chain import OptionContract
from features.option_signal_engine import generate_option_chain_signal

def test_option_signal_has_confidence_and_levels():
    contracts=[
        OptionContract("CE_25000","",25000,"CE",100,99,101,10000,20000,5000,20,ltp_change_pct=2),
        OptionContract("PE_25000","",25000,"PE",80,79,81,5000,20000,-1000,22,ltp_change_pct=-1),
    ]
    signal, rows = generate_option_chain_signal(contracts, 25000, 0.2)
    assert signal.confidence >= 0
    assert {"Confidence","Entry Price","Stop Loss","Take Profit","Global News"} <= set(rows.columns)

