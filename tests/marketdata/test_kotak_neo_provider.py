from marketdata.providers.kotak_neo import KotakNeoProvider


class FakeNeo:
    def option_chain(self, **kwargs):
        return {
            "data": {
                "common_data": {"unlSymbol": "NIFTY"},
                "call": [{
                    "instrument": {
                        "neoSymbol": "nse_fo|123",
                        "symbol": "NIFTY26OCT25000CE",
                        "strikePrice": "25000",
                        "expiryDt": "2026-10-08",
                    },
                    "quote": {"ltp": "120.5", "volume": 1000},
                    "openInterest": {"current": 2000, "change": 100},
                }],
                "put": [{
                    "instrument": {
                        "neoSymbol": "nse_fo|124",
                        "symbol": "NIFTY26OCT25000PE",
                        "strikePrice": "25000",
                        "expiryDt": "2026-10-08",
                    },
                    "quote": {"ltp": "110.5", "volume": 900},
                    "openInterest": {"current": 1800, "change": -50},
                }],
            },
        }

    def quotes(self, **kwargs):
        return [
            {
                "exchange_token": "123",
                "ltp": "121.0",
                "last_volume": "1000",
                "depth": {
                    "buy": [{"price": "120.9"}],
                    "sell": [{"price": "121.1"}],
                },
            },
            {
                "exchange_token": "124",
                "ltp": "110.0",
                "last_volume": "900",
                "depth": {
                    "buy": [{"price": "109.9"}],
                    "sell": [{"price": "110.1"}],
                },
            },
        ]


def test_kotak_option_chain_normalizes_current_response():
    chain = KotakNeoProvider(FakeNeo()).get_option_chain(
        underlying="NIFTY",
        exchange="NSE",
        count=40,
    )

    assert len(chain) == 2
    assert {x.option_type for x in chain} == {"CE", "PE"}
    assert chain[0].instrument_token.startswith("nse_fo|")
    assert all(x.ltp > 0 for x in chain)
    assert all(x.bid is not None and x.ask is not None for x in chain)


def test_option_chain_success_payload_without_stat_is_accepted():
    chain = KotakNeoProvider(FakeNeo()).get_option_chain(
        underlying="NIFTY",
        exchange="nse_fo",
        count=40,
        enrich_quotes=False,
    )
    assert len(chain) == 2
    assert chain[0].ltp == 120.5


def test_exchange_alias_is_normalized():
    assert KotakNeoProvider.normalize_exchange("NSE") == "nse_fo"
    assert KotakNeoProvider.normalize_exchange("MCX") == "mcx_fo"
