import pandas as pd

from app.one_minute_forecast_panel import _comparison_summary


def sample_candles(n=40):
    timestamp = pd.date_range("2026-10-09 09:15", periods=n, freq="min")
    close = [100 + i * 0.08 + (0.4 if i % 7 == 0 else 0) for i in range(n)]
    return pd.DataFrame({
        "timestamp": timestamp,
        "open": [value - 0.05 for value in close],
        "high": [value + (0.25 if i % 5 else 0.7) for i, value in enumerate(close)],
        "low": [value - (0.2 if i % 6 else 0.55) for i, value in enumerate(close)],
        "close": close,
        "volume": [1000 + i for i in range(n)],
    })


def test_comparison_evaluates_both_methods_on_same_target_count():
    summary = _comparison_summary(sample_candles(), quantile=0.9, lookback=20)

    assert list(summary["Model"]) == [
        "Adaptive (previous-close excursions)",
        "Baseline (open-to-extreme)",
    ]
    assert summary["Forecasts"].nunique() == 1
    assert summary["Forecasts"].iloc[0] == 37
    assert summary[["Low coverage %", "High coverage %", "Full-range coverage %"]].apply(
        lambda column: column.between(0, 100).all()
    ).all()
    assert (summary["Avg predicted width (₹)"] > 0).all()
    assert (summary["Avg actual width (₹)"] > 0).all()
    assert (summary["Width ratio (×)"] > 0).all()


def test_comparison_uses_same_requested_quantile_and_lookback():
    candles = sample_candles(12)
    summary = _comparison_summary(candles, quantile=0.75, lookback=10)

    assert len(summary) == 2
    assert summary["Forecasts"].tolist() == [9, 9]
