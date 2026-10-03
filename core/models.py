from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TradingSignal:

    timestamp: datetime

    instrument: str

    timeframe: str

    direction: str

    strike: Optional[float]

    expiry: Optional[str]

    entry: float

    stop_loss: float

    target: float

    score: float

    ml_probability: float

    technical_score: float

    option_score: float

    news_score: float

    regime: str

    reason: list[str]

    valid: bool
