from datetime import date, datetime, timedelta
from typing import List

from marketdata.models.market_data import (
    Candle,
    OptionContract,
    Quote,
)
from marketdata.providers.base import MarketDataProvider


class MockMarketDataProvider(MarketDataProvider):

    def get_quote(
        self,
        symbol: str,
        exchange: str,
    ) -> Quote:

        return Quote(
            timestamp=datetime.now(),
            symbol=symbol,
            exchange=exchange,
            ltp=25000.0,
            volume=100000.0,
            open_interest=1000000.0,
            bid=24999.0,
            ask=25001.0,
        )

    def get_option_chain(
        self,
        underlying: str,
        exchange: str,
        expiry: str | None = None,
        count: int = 40,
    ) -> List[OptionContract]:

        contracts = []

        base = 25000

        for offset in range(-5, 6):

            strike = base + offset * 100

            contracts.append(
                OptionContract(
                    symbol=f"{underlying}{strike}CE",
                    exchange=exchange,
                    underlying=underlying,
                    expiry=expiry or "TEST",
                    strike=strike,
                    option_type="CE",
                    ltp=max(50, 250 - abs(offset) * 20),
                    bid=100,
                    ask=105,
                    volume=10000,
                    open_interest=50000,
                    oi_change=1000,
                )
            )

            contracts.append(
                OptionContract(
                    symbol=f"{underlying}{strike}PE",
                    exchange=exchange,
                    underlying=underlying,
                    expiry=expiry or "TEST",
                    strike=strike,
                    option_type="PE",
                    ltp=max(50, 250 - abs(offset) * 20),
                    bid=100,
                    ask=105,
                    volume=10000,
                    open_interest=50000,
                    oi_change=1000,
                )
            )

        return contracts[:count]

    def get_historical_candles(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start: date,
        end: date,
    ) -> List[Candle]:

        candles = []

        current = datetime.combine(
            start,
            datetime.min.time(),
        )

        price = 25000.0

        while current.date() <= end:

            candles.append(
                Candle(
                    timestamp=current,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    open=price,
                    high=price + 20,
                    low=price - 20,
                    close=price + 5,
                    volume=100000,
                )
            )

            price += 5
            current += timedelta(minutes=5)

        return candles
