from datetime import date, datetime
from typing import Any, List

from marketdata.models.market_data import Candle, OptionContract, Quote
from marketdata.providers.base import MarketDataProvider


class KotakNeoProvider(MarketDataProvider):
    """Adapter for the current Kotak Neo Python SDK market-data APIs."""

    EXCHANGE_ALIASES = {
        "NSE": "nse_fo",
        "NSE_FO": "nse_fo",
        "MCX": "mcx_fo",
        "MCX_FO": "mcx_fo",
        "BSE": "bse_fo",
        "BSE_FO": "bse_fo",
    }

    def __init__(self, client):
        self.client = client

    @classmethod
    def normalize_exchange(cls, exchange: str) -> str:
        value = str(exchange or "").strip()
        return cls.EXCHANGE_ALIASES.get(value.upper(), value.lower())

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _response_data(response: Any) -> dict:
        if not isinstance(response, dict):
            return {}
        data = response.get("data")
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _response_error(response: Any) -> str:
        if not isinstance(response, dict):
            return "Kotak Neo returned an invalid response."
        if response.get("stat") == "Ok":
            return ""
        if response.get("errMsg"):
            return str(response["errMsg"])
        errors = response.get("error")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict) and first.get("message"):
                return str(first["message"])
            return str(first)
        return "Kotak Neo returned no market-data records."

    def get_quote(self, symbol: str, exchange: str) -> Quote:
        raise NotImplementedError(
            "Resolve the Neo instrument token before requesting a quote."
        )

    def get_option_chain(
        self,
        underlying: str,
        exchange: str,
        expiry: str | None = None,
        count: int = 40,
        enrich_quotes: bool = True,
    ) -> List[OptionContract]:
        exchange_segment = self.normalize_exchange(exchange)
        if exchange_segment not in {"nse_fo", "bse_fo", "mcx_fo"}:
            raise ValueError(
                f"Unsupported option-chain exchange '{exchange}'. "
                "Use nse_fo, bse_fo or mcx_fo."
            )

        if count < 10 or count % 10 != 0:
            raise ValueError("Kotak Neo option-chain count must be a multiple of 10.")

        response = self.client.option_chain(
            exchange=exchange_segment,
            underlying=underlying.upper(),
            expiry=expiry,
            instrument_type="option",
            count=count,
        )
        error = self._response_error(response)
        if error:
            raise RuntimeError(f"Kotak Neo option-chain error: {error}")

        data = self._response_data(response)
        calls = data.get("call") or []
        puts = data.get("put") or []
        if not calls and not puts:
            raise RuntimeError(
                "Kotak Neo returned an empty option chain. "
                "Check the underlying, exchange segment and expiry."
            )

        contracts: list[OptionContract] = []
        for item in calls:
            contract = self._parse_option(item, underlying, exchange_segment, "CE", expiry)
            if contract:
                contracts.append(contract)
        for item in puts:
            contract = self._parse_option(item, underlying, exchange_segment, "PE", expiry)
            if contract:
                contracts.append(contract)

        if enrich_quotes and contracts:
            self._enrich_quotes(contracts)

        return contracts

    def _parse_option(
        self,
        item: dict,
        underlying: str,
        exchange: str,
        option_type: str,
        requested_expiry: str | None,
    ) -> OptionContract | None:
        instrument = item.get("instrument") or item.get("inst") or {}
        quote = item.get("quote") or {}
        oi = item.get("openInterest") or item.get("oi") or {}

        symbol = instrument.get("symbol")
        neo_symbol = instrument.get("neoSymbol")
        if not symbol or not neo_symbol:
            return None

        expiry = instrument.get("expiryDt") or requested_expiry or ""
        strike = self._float(instrument.get("strikePrice") or instrument.get("strike"))
        ltp = self._float(quote.get("ltp"))
        volume = self._float(quote.get("volume") or quote.get("vol"))
        current_oi = self._float(oi.get("current") or oi.get("cur"))
        oi_change = self._float(oi.get("change") or oi.get("chg"))

        return OptionContract(
            symbol=str(symbol),
            exchange=exchange,
            underlying=underlying.upper(),
            expiry=str(expiry),
            strike=strike,
            option_type=option_type,
            instrument_token=str(neo_symbol),
            ltp=ltp,
            bid=None,
            ask=None,
            volume=volume,
            open_interest=current_oi,
            oi_change=oi_change,
        )

    def _enrich_quotes(self, contracts: list[OptionContract]) -> None:
        """Populate LTP/depth fields in batches of <=50, per Neo API limits."""
        for start in range(0, len(contracts), 50):
            batch = contracts[start:start + 50]
            tokens = []
            for contract in batch:
                token = contract.instrument_token or ""
                if "|" in token:
                    segment, instrument_token = token.split("|", 1)
                else:
                    segment, instrument_token = contract.exchange, token
                if instrument_token:
                    tokens.append({
                        "instrument_token": instrument_token,
                        "exchange_segment": segment,
                    })

            if not tokens:
                continue

            response = self.client.quotes(
                instrument_tokens=tokens,
                quote_type="all",
            )
            if not isinstance(response, list):
                data = self._response_data(response)
                response = data.get("quotes") or data.get("data") or []

            by_token = {}
            for quote in response if isinstance(response, list) else []:
                if not isinstance(quote, dict):
                    continue
                token = str(quote.get("exchange_token") or quote.get("instrument_token") or "")
                if token:
                    by_token[token] = quote

            for contract in batch:
                token = (contract.instrument_token or "").split("|")[-1]
                quote = by_token.get(token)
                if not quote:
                    continue
                ltp = self._float(quote.get("ltp"), contract.ltp or 0.0)
                depth = quote.get("depth") or {}
                buy = depth.get("buy") or depth.get("buyDepth") or []
                sell = depth.get("sell") or depth.get("sellDepth") or []
                bid = self._float(buy[0].get("price")) if buy and isinstance(buy[0], dict) else None
                ask = self._float(sell[0].get("price")) if sell and isinstance(sell[0], dict) else None

                # Dataclass is frozen; replace in place in the list.
                index = contracts.index(contract)
                contracts[index] = OptionContract(
                    symbol=contract.symbol,
                    exchange=contract.exchange,
                    underlying=contract.underlying,
                    expiry=contract.expiry,
                    strike=contract.strike,
                    option_type=contract.option_type,
                    instrument_token=contract.instrument_token,
                    ltp=ltp,
                    bid=bid,
                    ask=ask,
                    volume=contract.volume,
                    open_interest=contract.open_interest,
                    oi_change=contract.oi_change,
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
        error = self._response_error(response)
        if error:
            raise RuntimeError(f"Kotak Neo historical-data error: {error}")

        candles = []
        data = self._response_data(response)
        for row in data.get("candles", []) or []:
            if len(row) < 6:
                continue
            timestamp = datetime.fromisoformat(str(row[0]).replace("+0530", "+05:30"))
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
                    open_interest=self._float(row[6]) if len(row) > 6 else None,
                )
            )
        return candles
