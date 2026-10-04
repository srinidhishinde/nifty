import pandas as pd
from marketdata.option_chain_csv import is_nse_option_chain_export, parse_nse_option_chain_export

def test_nse_export_parser_reads_real_ltp_bid_ask_and_oi():
    data = pd.DataFrame([
        ["CALLS", "NO TEXT", "PUTS"],
        ["NO TEXT", "OI", "CHNG IN OI", "VOLUME", "IV", "LTP", "CHNG", "BID QTY", "BID", "ASK", "ASK QTY", "STRIKE", "BID QTY", "BID", "ASK", "ASK QTY", "CHNG", "LTP", "IV", "VOLUME", "CHNG IN OI", "OI", "NO TEXT"],
        ["", "60", "30", "56", "-", "1,446.30", "-201.75", "260", "1,444.25", "1,461.15", "65", "21,000.00", "7,020", "1.75", "1.80", "6,955", "0.15", "1.80", "24.08", "7,68,948", "35,593", "87,882", ""],
    ])
    assert is_nse_option_chain_export(data)
    contracts = parse_nse_option_chain_export(data, expiry="2026-10-06")
    ce = next(x for x in contracts if x.option_type == "CE")
    pe = next(x for x in contracts if x.option_type == "PE")
    assert ce.strike == 21000
    assert ce.ltp == 1446.30
    assert ce.bid == 1444.25
    assert ce.ask == 1461.15
    assert ce.open_interest == 60\n    assert ce.oi_change == 30\n    assert ce.volume == 56
    assert pe.open_interest == 87882\n    assert pe.oi_change == 35593\n    assert pe.volume == 768948\n    assert pe.ltp == 1.80
    assert pe.bid == 1.75
    assert pe.ask == 1.80
