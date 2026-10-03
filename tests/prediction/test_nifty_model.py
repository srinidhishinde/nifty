import pandas as pd
from prediction.nifty_model import walk_forward_predict

def test_walk_forward_model_runs():
    n=70
    close=[100 + i * 0.2 + (i % 3) for i in range(n)]
    data=pd.DataFrame({
        "timestamp":pd.date_range("2026-01-01",periods=n,freq="D"),
        "open":close,"high":[x+1 for x in close],"low":[x-1 for x in close],
        "close":close,"volume":[1000+i*10 for i in range(n)],
    })
    result=walk_forward_predict(data)
    assert 0 <= result.accuracy_pct <= 100
    assert {"Prediction","Probability Up","Actual","Correct"} <= set(result.predictions.columns)
