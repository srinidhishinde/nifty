from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from math import sqrt
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

@dataclass(frozen=True)
class MLConfig:
    window_days: int = 90
    refresh_minutes: int = 5
    horizons_minutes: tuple[int, ...] = (5, 10, 15)
    min_rows: int = 300
    test_fraction: float = 0.20
    cv_splits: int = 4
    correlation_threshold: float = 0.95
    probability_threshold: float = 0.60
    label_threshold_pct: float = 0.0005
    random_state: int = 42
    model_version: str = "ml-advisory-v1"

TECH = ("EMA_SLOPE","RSI","ATR_PCT","VWAP_DEV","MACD_HIST","VOLUME_RATIO")
DERIV = ("PCR","PCE","OI_SHIFT","DELTA_OI_SHIFT","ATM_IV")
SENT = ("SENTIMENT","SENTIMENT_CHANGE","NEWS_COUNT")
BASE = TECH + DERIV + SENT

@dataclass
class HorizonArtifact:
    horizon_minutes: int
    model_version: str
    features: list[str]
    model: object
    validation_accuracy: float
    validation_balanced_accuracy: float
    validation_samples: int
    accuracy_ci_low: float
    accuracy_ci_high: float

@dataclass
class TrainingResult:
    model_version: str
    window_start: str
    window_end: str
    rows: int
    features: list[str]
    horizons: dict

@dataclass(frozen=True)
class Prediction:
    horizon_minutes: int
    ce_probability: float
    pe_probability: float
    confidence: float
    confidence_low: float
    confidence_high: float
    advisory_direction: str
    model_version: str

def _norm(s): return pd.to_numeric(s, errors="coerce")
def _first(df, names):
    for n in names:
        if n in df.columns: return _norm(df[n])
    return pd.Series(np.nan, index=df.index)

def build_features(raw):
    req={"timestamp","open","high","low","close","volume"}
    miss=req-set(raw.columns)
    if miss: raise ValueError(f"Missing columns: {sorted(miss)}")
    d=raw.copy()
    d["timestamp"]=pd.to_datetime(d["timestamp"],errors="raise")
    d=d.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    close=_norm(d.close); high=_norm(d.high); low=_norm(d.low); vol=_norm(d.volume)
    ema9=close.ewm(span=9,adjust=False).mean()
    ema21=close.ewm(span=21,adjust=False).mean()
    ema_slope=ema9.pct_change(3)/3
    tr=pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    delta=close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/14,adjust=False,min_periods=14).mean(); al=loss.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rsi=100-(100/(1+(ag/al.replace(0,np.nan))))
    typical=(high+low+close)/3
    session=d.timestamp.dt.normalize()
    vwap=(typical*vol).groupby(session).cumsum()/vol.groupby(session).cumsum().replace(0,np.nan)
    ema12=close.ewm(span=12,adjust=False).mean(); ema26=close.ewm(span=26,adjust=False).mean()
    macd=ema12-ema26; macds=macd.ewm(span=9,adjust=False).mean()
    vma=vol.rolling(20,min_periods=20).mean()
    out=pd.DataFrame(index=d.index); out["timestamp"]=d.timestamp
    out["EMA_SLOPE"]=ema_slope; out["RSI"]=rsi; out["ATR_PCT"]=atr/close
    out["VWAP_DEV"]=(close-vwap)/vwap.replace(0,np.nan); out["MACD_HIST"]=macd-macds
    out["VOLUME_RATIO"]=vol/vma.replace(0,np.nan)
    out["PCR"]=_first(d,["PCR","PCR_OI","pcr_oi"]); out["PCE"]=_first(d,["PCE","pce"])
    out["OI_SHIFT"]=_first(d,["OI_SHIFT","oi_imbalance","OI_IMBALANCE"])
    out["DELTA_OI_SHIFT"]=_first(d,["DELTA_OI_SHIFT","delta_oi_imbalance","DELTA_OI_IMBALANCE"])
    out["ATM_IV"]=_first(d,["ATM_IV","option_iv","IV"])
    out["SENTIMENT"]=_first(d,["SENTIMENT","sentiment","news_sentiment"])
    out["SENTIMENT_CHANGE"]=_first(d,["SENTIMENT_CHANGE","sentiment_change","news_sentiment_change"])
    out["NEWS_COUNT"]=_first(d,["NEWS_COUNT","news_count"])
    return out.replace([np.inf,-np.inf],np.nan)

def rolling_window(df, days=90):
    d=df.copy(); d["timestamp"]=pd.to_datetime(d["timestamp"],errors="raise")
    end=d["timestamp"].max(); start=end-pd.Timedelta(days=days)
    return d[(d.timestamp>=start)&(d.timestamp<=end)].copy()

def drop_correlated(X, threshold):
    c=X.corr(numeric_only=True).abs()
    upper=c.where(np.triu(np.ones(c.shape),k=1).astype(bool))
    return [col for col in upper.columns if not any(upper[col]>threshold)]

def wilson(wins,n,z=1.96):
    if n==0:return (0.,0.)
    p=wins/n; den=1+z*z/n; center=(p+z*z/(2*n))/den
    half=z*sqrt((p*(1-p)+z*z/(4*n))/n)/den
    return max(0,center-half),min(1,center+half)

def _fit(X,y,c):
    pipe=Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler()),("model",HistGradientBoostingClassifier(max_iter=180,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=c.random_state))])
    return pipe.fit(X,y,model__sample_weight=compute_sample_weight("balanced",y))

def train(raw, cfg=MLConfig(), model_dir=None):
    d=rolling_window(raw,cfg.window_days)
    if len(d)<cfg.min_rows: raise ValueError(f"90-day window has only {len(d)} rows; need {cfg.min_rows}")
    f=build_features(d); candidates=[x for x in BASE if x in f.columns and f[x].notna().any()]
    if not candidates: raise ValueError("No usable ML features are available")
    X=f[candidates].copy(); selected=drop_correlated(X,cfg.correlation_threshold)
    results={}; artifacts={}
    for h in cfg.horizons_minutes:
        bars=max(1,round(h/cfg.refresh_minutes)); future=d.close.shift(-bars)/d.close-1
        y=(future>cfg.label_threshold_pct).astype(int); valid=future.notna()
        Xh=X.loc[valid].reset_index(drop=True); yh=y.loc[valid].reset_index(drop=True)
        split=max(1,int(len(Xh)*(1-cfg.test_fraction)))
        Xtr,Xte=Xh.iloc[:split],Xh.iloc[split:]; ytr,yte=yh.iloc[:split],yh.iloc[split:]
        if len(Xtr)<cfg.min_rows//2 or len(Xte)<30: raise ValueError(f"Insufficient train/test rows for {h}m")
        cv=TimeSeriesSplit(n_splits=min(cfg.cv_splits,3)); cv_scores=[]
        for ti,vi in cv.split(Xtr):
            m=_fit(Xtr.iloc[ti],ytr.iloc[ti],cfg); cv_scores.append(accuracy_score(ytr.iloc[vi],m.predict(Xtr.iloc[vi])))
        final=_fit(Xtr,ytr,cfg); pred=final.predict(Xte); acc=float(accuracy_score(yte,pred)); bal=float(balanced_accuracy_score(yte,pred))
        lo,hi=wilson(int((pred==yte).sum()),len(yte))
        artifacts[h]=HorizonArtifact(h,cfg.model_version,selected,final,acc,bal,len(yte),lo,hi)
        results[h]={"training_accuracy":float(accuracy_score(ytr,final.predict(Xtr))),"validation_accuracy":acc,"validation_balanced_accuracy":bal,"cv_accuracy":float(np.mean(cv_scores)),"validation_samples":len(yte),"confidence_interval_pct":[100*lo,100*hi]}
    if model_dir:
        p=Path(model_dir); p.mkdir(parents=True,exist_ok=True)
        for h,a in artifacts.items(): joblib.dump(a,p/f"{cfg.model_version}_{h}m.joblib")
        (p/"manifest.json").write_text(json.dumps({"model_version":cfg.model_version,"features":selected,"horizons":results},indent=2))
    return TrainingResult(cfg.model_version,str(d.timestamp.min()),str(d.timestamp.max()),len(d),selected,results),artifacts

def predict(raw,artifacts,cfg=MLConfig()):
    if not artifacts: return []
    f=build_features(raw); features=next(iter(artifacts.values())).features
    X=f[features].iloc[[-1]]; out=[]
    for h,a in sorted(artifacts.items()):
        p=a.model.predict_proba(X)[0]; pe=float(p[0]); ce=float(p[1]); confidence=max(ce,pe)
        direction="CE" if ce>=pe else "PE"
        if confidence<cfg.probability_threshold: direction="WAIT"
        out.append(Prediction(h,ce,pe,confidence,100*a.accuracy_ci_low,100*a.accuracy_ci_high,direction,a.model_version))
    return out
