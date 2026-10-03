import pandas as pd
from strategy.buy_today_sell_tomorrow import run_buy_today_sell_tomorrow

def test_btst_returns_next_day_trades():
    data=pd.DataFrame({
        "timestamp":pd.date_range("2026-01-01",periods=3,freq="D"),
        "open":[100,101,102],"high":[102,103,104],"low":[99,100,101],"close":[101,102,103]
    })
    result=run_buy_today_sell_tomorrow(data)
    assert len(result.trades)==2
    assert result.accuracy_pct==100.0

