import pandas as pd
from strategy.regime import classify_regime


def test_high_volatility_is_not_tradable():
    result = classify_regime(pd.Series({"ATR_PCT": 0.05, "ADX": 25, "EMA_SPREAD": 1.0, "VWAP_DEV": 0.0}))
    assert result.regime == "HIGH_VOLATILITY"
    assert not result.tradable


def test_trend_regime_uses_current_bar_only():
    result = classify_regime(pd.Series({"ATR_PCT": 0.01, "ADX": 25, "EMA_SPREAD": 2.0, "VWAP_DEV": 0.02}))
    assert result.regime == "TREND_UP"
    assert result.tradable
