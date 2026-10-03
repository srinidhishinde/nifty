from abc import ABC, abstractmethod
from datetime import date
from typing import List

from marketdata.models.market_data import (
    Candle,
    OptionContract,
    Quote,
)


class MarketDataProvider(ABC):

    @abstractmethod
    def get_quote(
        self,
        symbol: str,
        exchange: str,
    ) -> Quote:
        raise NotImplementedError

    @abstractmethod
    def get_option_chain(
        self,
        underlying: str,
        exchange: str,
        expiry: str | None = None,
        count: int = 40,
    ) -> List[OptionContract]:
        raise NotImplementedError

    @abstractmethod
    def get_historical_candles(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start: date,
        end: date,
    ) -> List[Candle]:
        raise NotImplementedError
