from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MLConfig:
    rolling_days: int = 90
    refresh_minutes: int = 5
    horizons_minutes: tuple[int, ...] = (5, 10, 15)
    bar_minutes: int = 5
    label_threshold_pct: float = 0.0005
    min_training_samples: int = 250
    min_oos_samples: int = 50
    cv_splits: int = 4
    test_size: int = 60
    correlation_threshold: float = 0.95
    l1_c: float = 0.25
    l2_regularization: float = 0.5
    random_state: int = 42
    model_version: str = "ml-advisory-v1"
    feature_columns: tuple[str, ...] = field(default_factory=lambda: (
        "ema_slope_9", "ema_slope_21", "RSI", "ATR_PCT", "VWAP_DEV",
        "MACD_HIST", "ADX", "VOLUME_RATIO", "ROC", "BB_WIDTH", "BB_POS",
        "TREND_SPREAD", "close_return_1", "close_return_3", "close_return_6",
        "atr_percentile", "volume_zscore", "option_pcr", "option_pcr_change",
        "option_oi_imbalance", "option_delta_oi_imbalance", "option_iv",
        "sentiment", "sentiment_change", "news_count",
    ))
