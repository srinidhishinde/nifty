from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from .config import MLConfig
from .features import build_features,feature_matrix,drop_highly_correlated
from .labels import make_direction_labels
from .metrics import classification_metrics
from .model import fit_model,HorizonModel,save_model

@dataclass(frozen=True)
class MLTrainingResult:
    models:dict[int,HorizonModel]; metrics:dict[int,dict]; samples:int; feature_columns:tuple[str,...]; dropped_correlated:tuple[str,...]; model_version:str; training_start:str; training_end:str

def train_multihorizon(data:pd.DataFrame,config:MLConfig|None=None,model_dir:str|Path|None=None)->MLTrainingResult:
    c=config or MLConfig(); frame=build_features(data,c.feature_columns)
    end=frame["timestamp"].max(); start=end-pd.Timedelta(days=c.rolling_days)
    frame=frame[frame["timestamp"]>=start].reset_index(drop=True)
    if frame.empty or (end-start).days<c.rolling_days: raise ValueError(f"ML requires a full {c.rolling_days}-day rolling window; received {frame[\"timestamp\"].min()} to {end}")
    X_all=feature_matrix(frame,c.feature_columns); X_all,dropped=drop_highly_correlated(X_all,c.correlation_threshold)
    models={}; metrics={}
    for horizon in c.horizons_minutes:
        bars=max(1,round(horizon/c.bar_minutes)); y_all=make_direction_labels(frame,bars,c.label_threshold_pct)
        valid=X_all.notna().all(axis=1)&y_all.notna(); X=X_all.loc[valid].reset_index(drop=True); y=y_all.loc[valid].astype(float).reset_index(drop=True)
        if len(X)<c.min_training_samples: raise ValueError(f"Insufficient ML samples for {horizon}m: {len(X)} < {c.min_training_samples}")
        splitter=TimeSeriesSplit(n_splits=c.cv_splits,test_size=min(c.test_size,max(20,len(X)//(c.cv_splits+1))))
        oos_true=[]; oos_prob=[]
        for train_idx,test_idx in splitter.split(X):
            if len(train_idx)<max(60,c.min_training_samples//2): continue
            fold=fit_model(X.iloc[train_idx],y.iloc[train_idx],feature_columns=tuple(X.columns),horizon_minutes=horizon,l1_c=c.l1_c,l2_regularization=c.l2_regularization,random_state=c.random_state,model_version=c.model_version)
            oos_true.extend(y.iloc[test_idx].tolist()); oos_prob.extend(fold.estimator.predict_proba(X.iloc[test_idx]).tolist())
        if len(oos_true)<c.min_oos_samples: raise ValueError(f"Insufficient OOS samples for {horizon}m: {len(oos_true)} < {c.min_oos_samples}")
        m=classification_metrics(oos_true,np.asarray(oos_prob),(-1.0,0.0,1.0)); m["horizon_minutes"]=horizon; metrics[horizon]=m
        final=fit_model(X,y,feature_columns=tuple(X.columns),horizon_minutes=horizon,l1_c=c.l1_c,l2_regularization=c.l2_regularization,random_state=c.random_state,model_version=c.model_version); models[horizon]=final
        if model_dir is not None:
            Path(model_dir).mkdir(parents=True,exist_ok=True); save_model(final,str(Path(model_dir)/f"{c.model_version}_{horizon}m.joblib"))
    return MLTrainingResult(models,metrics,len(X_all),tuple(X_all.columns),dropped,c.model_version,str(frame["timestamp"].min()),str(frame["timestamp"].max()))
