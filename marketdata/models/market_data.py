from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    symbol: str
    exchange: str
    timeframe: str

    open: float
    high: float
    low: float
    close: float

    volume: float = 0.0
    open_interest: Optional[float] = None


@dataclass(frozen=True)
class Quote:
    timestamp: datetime
    symbol: str
    exchange: str

    ltp: float
    volume: float = 0.0
    open_interest: Optional[float] = None

    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    exchange: str
    underlying: str

    expiry: str
    strike: float
    option_type: str

    instrument_token: Optional[str] = None

    ltp: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None

    volume: float = 0.0
    open_interest: float = 0.0
    oi_change: float = 0.0

    # Optional derivatives fields populated when the broker supplies them.
    # Keeping these fields optional preserves compatibility with OHLCV-only
    # consumers while allowing the live option chain to feed the strategy
    # analyzer without an unsafe model conversion at the broker boundary.
    implied_volatility: float = 0.0
    built_up: str = ""
    delta: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    gamma: Optional[float] = None
    ltp_change_pct: float = 0.0
