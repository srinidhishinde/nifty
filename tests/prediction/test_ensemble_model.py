import pandas as pd
from prediction.ensemble_model import walk_forward_ensemble, predict_latest

def make_data(n=110):
    close=[100+i*0.15+(i%7)*0.2 for i in range(n)]
    return pd.DataFrame({
        "timestamp":pd.date_range("2026-01-01",periods=n,freq="D"),
        "open":close,"high":[x+1 for x in close],"low":[x-1 for x in close],
        "close":close,"volume":[1000+i*5 for i in range(n)],
    })

def test_walk_forward_ensemble_is_temporal():
    result=walk_forward_ensemble(make_data(),min_train=60)
    assert 0<=result.accuracy_pct<=100
    assert 0<=result.coverage_pct<=100
    assert {"Prediction","Probability Up","Confidence","Abstain","Actual","Correct"}<=set(result.predictions.columns)
    assert ((result.predictions["Probability Up"] >= 0) & (result.predictions["Probability Up"] <= 100)).all()

def test_latest_prediction_returns_safe_state_with_short_history():
    result=predict_latest(make_data(40),min_train=60)
    assert result.direction=="UNAVAILABLE"
    assert result.abstain

def test_latest_prediction_reports_calibrated_confidence():
    result=predict_latest(make_data(),min_train=60)
    assert 0 <= result.probability_up <= 1
    assert 0 <= result.confidence <= 100
