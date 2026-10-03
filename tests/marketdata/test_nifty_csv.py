import pandas as pd
from marketdata.nifty_csv import normalize_nifty_csv

def test_normalize_exported_nifty_csv():
    raw=pd.DataFrame({"Date ":["01-OCT-2026"],"Open ":[100],"High ":[105],"Low ":[99],"Close ":[103],"Shares Traded ":[1000],"Turnover (₹ Cr)":[1]})
    out=normalize_nifty_csv(raw)
    assert list(out.columns)==["timestamp","open","high","low","close","volume"]
    assert out.iloc[0]["close"]==103
