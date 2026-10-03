from __future__ import annotations


class KotakMarketData:

    def __init__(self, client):
        self.client = client

    def option_chain(
        self,
        exchange: str,
        underlying: str,
        expiry: str | None = None,
        count: int = 40,
    ):

        return self.client.option_chain(
            exchange=exchange,
            underlying=underlying,
            expiry=expiry,
            instrument_type="option",
            count=count,
        )

    def quotes(
        self,
        instrument_tokens: list[dict],
    ):

        return self.client.quotes(
            instrument_tokens=instrument_tokens,
            quote_type="all",
        )

    def expiries(
        self,
        exchange: str,
        underlying: str,
    ):

        return self.client.expiries(
            exchange=exchange,
            underlying=underlying,
        )
