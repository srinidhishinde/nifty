from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .config import MLConfig
from .features import build_features,feature_matrix,drop_highly_correlated
from .model import HorizonModel

@dataclass(frozen=True)
class MLPrediction:
    horizon_minutes:int; ce_probability:float; pe_probability:float; confidence_low:float; confidence_high:float; confidence_pct:float; advisory_direction:str; model_version:str

def predict_latest(data:pd.DataFrame,models:dict[int,HorizonModel],config:MLConfig|None=None)->list[MLPrediction]:
    c=config or MLConfig(); frame=build_features(data,c.feature_columns); X=feature_matrix(frame,c.feature_columns); X,_=drop_highly_correlated(X,c.correlation_threshold); latest=X.iloc[[-1]]
    if latest.isna().any(axis=None): raise ValueError("Latest candle does not contain all required ML features")
    out=[]
    for horizon in c.horizons_minutes:
        model=models.get(horizon)
        if model is None: continue
        p=model.estimator.predict_proba(latest)[0]; mapping={float(cls):float(prob) for cls,prob in zip(model.estimator.classes_,p)}
        pe=mapping.get(-1.0,0.0); ce=mapping.get(1.0,0.0); conf=max(ce,pe); half=1.96*((conf*(1-conf))/max(1,len(data)))**0.5
        lo=max(0,conf-half); hi=min(1,conf+half); direction="CE" if ce>=pe else "PE"
        out.append(MLPrediction(horizon,ce,pe,lo,hi,conf,direction,model.model_version))
    return out
