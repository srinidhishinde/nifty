import pandas as pd


def create_sample_data() -> pd.DataFrame:

    timestamps = pd.date_range(
        "2026-01-01 09:15",
        periods=20,
        freq="5min",
    )

    prices = [
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        106,
        105,
        104,
        103,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
    ]

    signals = [
        "WAIT",
        "CE",
        "CE",
        "CE",
        "CE",
        "CE",
        "CE",
        "CE",
        "PE",
        "PE",
        "PE",
        "PE",
        "PE",
        "CE",
        "CE",
        "CE",
        "CE",
        "CE",
        "CE",
        "CE",
    ]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000] * len(prices),
            "signal": signals,
        }
    )