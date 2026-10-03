import pandas as pd

from features.technical.indicators import add_indicators


def test_indicators():

    data = pd.DataFrame({

        "open": range(100, 200),

        "high": range(102, 202),

        "low": range(98, 198),

        "close": range(101, 201),

        "volume": [1000] * 100

    })

    result = add_indicators(data)

    assert "EMA20" in result.columns
    assert "EMA50" in result.columns
    assert "RSI" in result.columns
    assert "ATR" in result.columns
