from datetime import date, datetime
from typing import List

from marketdata.models.market_data import (
    Candle,
    OptionContract,
    Quote,
)
from marketdata.providers.base import MarketDataProvider


class KotakNeoProvider(MarketDataProvider):

    def __init__(self, client):
        self.client = client

    @staticmethod
    def _float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def get_quote(
        self,
        symbol: str,
        exchange: str,
    ) -> Quote:

        # This method intentionally expects the caller to resolve
        # instrument_token before requesting the quote.
        raise NotImplementedError(
            "Use the scrip master/token resolver before requesting quotes."
        )

    def get_option_chain(
        self,
        underlying: str,
        exchange: str,
        expiry: str | None = None,
        count: int = 40,
    ) -> List[OptionContract]:

        response = self.client.option_chain(
            exchange=exchange,
            underlying=underlying,
            expiry=expiry,
            instrument_type="option",
            count=count,
        )

        contracts = []

        calls = response.get("call", [])
        puts = response.get("put", [])

        for item in calls:
            contract = self._parse_option(
                item,
                underlying=underlying,
                exchange=exchange,
                option_type="CE",
            )

            if contract:
                contracts.append(contract)

        for item in puts:
            contract = self._parse_option(
                item,
                underlying=underlying,
                exchange=exchange,
                option_type="PE",
            )

            if contract:
                contracts.append(contract)

        return contracts

    def _parse_option(
        self,
        item,
        underlying: str,
        exchange: str,
        option_type: str,
    ):

        inst = item.get("inst", {})
        quote = item.get("quote", {})
        oi = item.get("oi", {})

        symbol = inst.get("symbol")

        if not symbol:
            return None

        expiry = inst.get("expiryDt")

        strike = self._float(
            inst.get("strikePrice")
            or inst.get("strike")
        )

        ltp = self._float(
            quote.get("ltp")
        )

        return OptionContract(
            symbol=symbol,
            exchange=exchange,
            underlying=underlying,
            expiry=str(expiry),
            strike=strike,
            option_type=option_type,
            instrument_token=inst.get("neoSymbol"),
            ltp=ltp,
            bid=self._float(
                quote.get("bp"),
                default=0.0,
            ),
            ask=self._float(
                quote.get("sp"),
                default=0.0,
            ),
            volume=self._float(
                quote.get("vol"),
            ),
            open_interest=self._float(
                oi.get("cur"),
            ),
            oi_change=self._float(
                oi.get("chg"),
            ),
        )

    def get_historical_candles(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start: date,
        end: date,
    ) -> List[Candle]:

        response = self.client.historical_data(
            neosymbol=symbol,
            interval=timeframe,
            from_date=start.isoformat(),
            to_date=end.isoformat(),
        )

        candles = []

        for row in response.get("data", {}).get("candles", []):

            if len(row) < 6:
                continue

            timestamp = datetime.fromisoformat(
                str(row[0]).replace(
                    "+0530",
                    "+05:30",
                )
            )

            candles.append(
                Candle(
                    timestamp=timestamp,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    open=self._float(row[1]),
                    high=self._float(row[2]),
                    low=self._float(row[3]),
                    close=self._float(row[4]),
                    volume=self._float(row[5]),
                    open_interest=(
                        self._float(row[6])
                        if len(row) > 6 and row[6] is not None
                        else None
                    ),
                )
            )

        return candles
