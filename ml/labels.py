from __future__ import annotations
import numpy as np
import pandas as pd

def make_direction_labels(data: pd.DataFrame, horizon_bars: int, threshold_pct: float) -> pd.Series:
    close=pd.to_numeric(data["close"],errors="coerce")
    future_return=close.shift(-horizon_bars)/close-1.0
    labels=pd.Series(np.nan,index=data.index,dtype="float64")
    labels[future_return>threshold_pct]=1.0
    labels[future_return<-threshold_pct]=-1.0
    labels[future_return.abs()<=threshold_pct]=0.0
    return labels
