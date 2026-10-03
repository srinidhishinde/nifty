from dataclasses import dataclass
from datetime import datetime
from typing import Literal


Instrument = Literal[
    "NIFTY",
    "CRUDE",
    "NATGAS",
    "COPPER",
    "SILVER",
    "GOLD",
]

OptionType = Literal["CE", "PE"]


@dataclass(frozen=True)
class OptionSnapshot:
    """
    One point-in-time option-chain observation.

    All prices are normalized into a broker-independent structure.
    """

    timestamp: datetime

    instrument: Instrument

    expiry: str

    strike: float

    option_type: OptionType

    underlying_price: float

    option_ltp: float

    bid: float

    ask: float

    volume: int

    open_interest: int

    oi_change: int

    implied_volatility: float


@dataclass(frozen=True)
class UnderlyingSnapshot:
    timestamp: datetime

    instrument: Instrument

    price: float

    open: float

    high: float

    low: float

    close: float

    volume: int

    vwap: float


@dataclass(frozen=True)
class ChainSnapshot:
    """
    Complete point-in-time snapshot.

    No future rows should be visible to the strategy.
    """

    timestamp: datetime

    underlying: UnderlyingSnapshot

    options: tuple[OptionSnapshot, ...]
