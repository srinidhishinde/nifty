from pathlib import Path

import pandas as pd
import pytest

from scripts.run_backtest_validation import load_ohlcv


def test_load_ohlcv_requires_real_backtest_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2026-01-01"], "close": [100]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_ohlcv(path)


def test_load_ohlcv_sorts_and_deduplicates(tmp_path: Path):
    path = tmp_path / "good.csv"
    pd.DataFrame([
        ["2026-01-01 09:20", 101, 102, 100, 101, 1000],
        ["2026-01-01 09:15", 100, 101, 99, 100, 900],
        ["2026-01-01 09:15", 100, 101, 99, 100, 900],
    ], columns=["timestamp", "open", "high", "low", "close", "volume"]).to_csv(path, index=False)
    frame = load_ohlcv(path)
    assert len(frame) == 2
    assert frame.iloc[0]["timestamp"] < frame.iloc[1]["timestamp"]
