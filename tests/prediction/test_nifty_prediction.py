import pandas as pd
from prediction.nifty_315_340 import predict_315_340, evaluate_next_day_accuracy

def test_intraday_prediction_requires_window():
    data=pd.DataFrame({"timestamp":pd.date_range("2026-01-01 10:00",periods=3,freq="5min"),"high":[1,2,3],"low":[0,1,2],"close":[1,2,3],"volume":[1,1,1]})
    assert predict_315_340(data).prediction=="UNAVAILABLE"

def test_daily_accuracy_evaluator_runs():
    data=pd.DataFrame({"timestamp":pd.date_range("2026-01-01",periods=55,freq="D"),"high":[101+i for i in range(55)],"low":[99+i for i in range(55)],"close":[100+i for i in range(55)],"volume":[1000]*55})
    accuracy, rows=evaluate_next_day_accuracy(data)
    assert len(rows)==54
    assert accuracy==100.0

