from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class MicrostructureTradeLogger:
    def __init__(self, path: str | Path = "logs/microstructure_trades.jsonl"):
        self.path = Path(path)

    def log(
        self,
        *,
        signal: str,
        outcome: str | float,
        reliability: float,
        confidence: float,
        regime_state: str,
        volatility_band: float,
        sentiment_score: float,
        imbalance_ratio: float,
        aggressor: str,
        micro_weight: float,
        threshold: float,
    ) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": signal,
            "outcome": outcome,
            "reliability": float(reliability),
            "confidence": float(confidence),
            "regime_state": regime_state,
            "volatility_band": float(volatility_band),
            "sentiment_score": float(sentiment_score),
            "imbalance_ratio": float(imbalance_ratio),
            "aggressor": aggressor,
            "micro_weight": float(micro_weight),
            "threshold": float(threshold),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
