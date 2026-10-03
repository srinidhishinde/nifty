import pandas as pd
from features.technical.candlestick import add_candlestick_patterns

def test_bullish_engulfing_is_detected():
    data=pd.DataFrame({"open":[105,99],"high":[106,108],"low":[98,97],"close":[100,107]})
    out=add_candlestick_patterns(data)
    assert int(out.iloc[-1]["bullish_engulfing"])==1

def test_bearish_engulfing_is_detected():
    data=pd.DataFrame({"open":[95,101],"high":[103,104],"low":[94,90],"close":[102,94]})
    out=add_candlestick_patterns(data)
    assert int(out.iloc[-1]["bearish_engulfing"])==1
