import numpy as np
import pandas as pd
from ml.config import MLConfig
from ml.features import build_features,drop_highly_correlated
from ml.labels import make_direction_labels
from ml.metrics import wilson_interval

def sample_ohlcv(n=420):
    rng=np.random.default_rng(7); ts=pd.date_range("2026-01-01",periods=n,freq="5min",tz="Asia/Kolkata"); close=25000+np.cumsum(rng.normal(0,8,n))+np.linspace(0,250,n)
    return pd.DataFrame({"timestamp":ts,"open":close-rng.uniform(0,2,n),"high":close+rng.uniform(1,4,n),"low":close-rng.uniform(1,4,n),"close":close,"volume":rng.integers(1000,5000,n)})

def test_labels_are_forward_only():
    y=make_direction_labels(sample_ohlcv(20),2,0.0001); assert y.iloc[-2:].isna().all(); assert y.iloc[:-2].notna().all()

def test_feature_builder_exposes_pseudocode_features():
    df=build_features(sample_ohlcv(),MLConfig().feature_columns); assert {"ema_slope_9","ema_slope_21","RSI","ATR_PCT","VWAP_DEV","MACD_HIST","option_pcr","sentiment"}.issubset(df.columns)

def test_correlation_filter_is_deterministic():
    x=pd.DataFrame({"a":range(20),"b":range(20),"c":np.arange(20)[::-1]}); out,dropped=drop_highly_correlated(x,.95); assert "b" in dropped and "a" in out.columns

def test_wilson_interval_is_bounded():
    lo,hi=wilson_interval(70,100); assert 0<=lo<=.70<=hi<=1
