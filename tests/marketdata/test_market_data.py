from datetime import date

from marketdata.models.market_data import (
    Candle,
    OptionContract,
    Quote,
)
from marketdata.providers.mock import MockMarketDataProvider


def test_mock_quote():

    provider = MockMarketDataProvider()

    quote = provider.get_quote(
        symbol="NIFTY",
        exchange="NSE",
    )

    assert isinstance(quote, Quote)
    assert quote.ltp > 0


def test_mock_option_chain_contains_ce_and_pe():

    provider = MockMarketDataProvider()

    chain = provider.get_option_chain(
        underlying="NIFTY",
        exchange="nse_fo",
    )

    assert len(chain) > 0

    types = {
        contract.option_type
        for contract in chain
    }

    assert "CE" in types
    assert "PE" in types


def test_mock_historical_data():

    provider = MockMarketDataProvider()

    candles = provider.get_historical_candles(
        symbol="NIFTY",
        exchange="NSE",
        timeframe="5min",
        start=date(2026, 1, 1),
        end=date(2026, 1, 1),
    )

    assert len(candles) > 0

    assert all(
        isinstance(candle, Candle)
        for candle in candles
    )


def test_option_chain_has_positive_prices():

    provider = MockMarketDataProvider()

    chain = provider.get_option_chain(
        underlying="NIFTY",
        exchange="nse_fo",
    )

    assert all(
        contract.ltp is not None
        and contract.ltp > 0
        for contract in chain
    )
