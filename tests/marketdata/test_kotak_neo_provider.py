from datetime import date
import pandas as pd

from marketdata.models.market_data import OptionContract
from marketdata.providers.kotak_neo import KotakNeoProvider


class FakeNeo:
    def expiries(self, **kwargs):
        return {"expiries": ["2026-10-08", "2026-10-15"]}

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
        enrich_quotes=True,
    )

    assert len(chain) == 2
    assert {x.option_type for x in chain} == {"CE", "PE"}
    assert chain[0].instrument_token.startswith("nse_fo|")
    assert all(x.ltp > 0 for x in chain)
    assert all(hasattr(x, "implied_volatility") for x in chain)
    assert all(x.bid is not None and x.ask is not None for x in chain)



def test_mcx_option_chain_uses_canonical_scrip_master_underlying():
    class FakeMCXNeo(FakeNeo):
        def __init__(self):
            self.search_calls = []
            self.option_calls = []

        def search_scrip(self, **kwargs):
            self.search_calls.append(kwargs)
            return [{
                "pSymbol": "569999",
                "pExchSeg": "mcx_fo",
                "pSymbolName": "CRUDEOILM",
                "pTrdSymbol": "CRUDEOILM26OCTFUT",
            }]

        def option_chain(self, **kwargs):
            self.option_calls.append(kwargs)
            return {
                "data": {
                    "common_data": {"unlSymbol": "CRUDEOILM"},
                    "call": [{
                        "instrument": {
                            "neoSymbol": "mcx_fo|700001",
                            "symbol": "CRUDEOILM26OCT8800CE",
                            "strikePrice": "8800",
                            "expiryDt": "2026-10-26",
                        },
                        "quote": {"ltp": "100", "volume": 10},
                        "openInterest": {"current": 20, "change": 1},
                    }],
                    "put": [],
                },
            }

    client = FakeMCXNeo()
    chain = KotakNeoProvider(client).get_option_chain(
        underlying="CRUDEOIL",
        exchange="MCX",
        count=40,
        enrich_quotes=False,
    )

    assert len(chain) == 1
    assert chain[0].underlying == "CRUDEOIL"
    assert client.option_calls[0]["exchange"] == "mcx_fo"
    assert client.option_calls[0]["underlying"] == "CRUDEOILM"
    assert client.search_calls[0]["exchange_segment"] == "mcx_fo"



def test_mcx_option_chain_recovers_strike_from_symbol_when_field_missing():
    class SymbolStrikeNeo(FakeNeo):
        def search_scrip(self, **kwargs):
            return [{
                "pSymbol": "569999",
                "pExchSeg": "mcx_fo",
                "pSymbolName": "CRUDEOILM",
                "pTrdSymbol": "CRUDEOILM26OCTFUT",
            }]

        def option_chain(self, **kwargs):
            return {"data": {"call": [{"instrument": {"neoSymbol": "mcx_fo|700001", "symbol": "CRUDEOILM26OCT8800CE", "expiryDt": "2026-10-26"}, "quote": {"ltp": "100", "volume": 10}, "openInterest": {"current": 20, "change": 1}}], "put": []}}

    chain = KotakNeoProvider(SymbolStrikeNeo()).get_option_chain(
        underlying="CRUDEOIL", exchange="MCX", count=40, enrich_quotes=False
    )
    assert len(chain) == 1
    assert chain[0].strike == 8800.0



def test_option_chain_uses_symbol_strike_when_payload_strike_is_wrong():
    class WrongStrikeNeo(FakeNeo):
        def option_chain(self, **kwargs):
            return {
                "data": {
                    "call": [{
                        "instrument": {
                            "neoSymbol": "nse_fo|900001",
                            "symbol": "NIFTY26OCT25000CE",
                            "strikePrice": "1320",
                            "expiryDt": "2026-10-15",
                        },
                        "quote": {"ltp": "120", "volume": 100},
                        "openInterest": {"current": 2000, "change": 100},
                    }],
                    "put": [{
                        "instrument": {
                            "neoSymbol": "nse_fo|900002",
                            "symbol": "NIFTY26OCT25000PE",
                            "strikePrice": "500",
                            "expiryDt": "2026-10-15",
                        },
                        "quote": {"ltp": "110", "volume": 100},
                        "openInterest": {"current": 1800, "change": 50},
                    }],
                }
            }

    chain = KotakNeoProvider(WrongStrikeNeo()).get_option_chain(
        underlying="NIFTY", exchange="NSE", count=40, enrich_quotes=False
    )
    assert {contract.strike for contract in chain} == {25000.0}


def test_option_chain_defaults_to_broker_payload_without_quote_enrichment():
    class NoQuotesNeo(FakeNeo):
        def quotes(self, **kwargs):
            raise AssertionError("quote enrichment must be opt-in")

    chain = KotakNeoProvider(NoQuotesNeo()).get_option_chain(
        underlying="NIFTY",
        exchange="nse_fo",
        count=40,
    )
    assert len(chain) == 2
    assert chain[0].ltp == 120.5


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


class FakeHistoricalNeo:
    def __init__(self):
        self.calls = []

    def historical_data(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "success", "data": {"candles": []}}




class FakeNiftyScripNeo(FakeHistoricalNeo):
    def __init__(self):
        super().__init__()
        self.search_calls = []

    def search_scrip(self, **kwargs):
        self.search_calls.append(kwargs)
        return [{
            "pSymbol": "99999",
            "pExchSeg": "nse_cm",
            "pSymbolName": "NIFTY",
            "pTrdSymbol": "NIFTY",
        }]


def test_nifty_historical_is_explicitly_unsupported_without_fallback():
    client = FakeHistoricalNeo()
    try:
        KotakNeoProvider(client).get_historical_candles(
            symbol="NIFTY",
            exchange="NSE",
            timeframe="5m",
            start=date(2026, 9, 25),
            end=date(2026, 10, 5),
            neosymbol="nse_cm|26000",
        )
    except RuntimeError as exc:
        assert "does not support NIFTY index candles" in str(exc)
    else:
        raise AssertionError("NIFTY historical candles must fail closed")
    assert client.calls == []


def test_kotak_historical_interval_uses_neo_format():
    client = FakeHistoricalNeo()
    KotakNeoProvider(client).get_historical_candles(
        symbol="RELIANCE",
        exchange="NSE",
        timeframe="5m",
        start=date(2026, 9, 25),
        end=date(2026, 10, 5),
        neosymbol="nse_cm|2885",
    )
    assert client.calls[0]["interval"] == "5min"


def test_kotak_historical_interval_preserves_supported_neo_value():
    client = FakeHistoricalNeo()
    KotakNeoProvider(client).get_historical_candles(
        symbol="RELIANCE",
        exchange="NSE",
        timeframe="15min",
        start=date(2026, 9, 25),
        end=date(2026, 10, 5),
        neosymbol="nse_cm|26000",
    )
    assert client.calls[0]["interval"] == "15min"


def test_kotak_historical_daily_interval_uses_neo_format():
    client = FakeHistoricalNeo()
    KotakNeoProvider(client).get_historical_candles(
        symbol="RELIANCE",
        exchange="NSE",
        timeframe="1d",
        start=date(2026, 9, 25),
        end=date(2026, 10, 5),
        neosymbol="nse_cm|26000",
    )
    assert client.calls[0]["interval"] == "D"


def test_kotak_explicit_error_payload_is_not_treated_as_success():
    class ErrorNeo:
        pass

    provider = KotakNeoProvider(ErrorNeo())
    error = provider._response_error({
        "status": "ERROR",
        "data": {"stale": "payload"},
        "fault": {"code": 422, "message": "Invalid interval value"},
    })
    assert "Invalid interval value" in error


def test_option_contract_strike_is_numeric():
    contract = OptionContract(
        symbol="NIFTY25OCT25000CE",
        exchange="nse_fo",
        underlying="NIFTY",
        expiry="2026-10-25",
        strike=25000.0,
        option_type="CE",
        instrument_token="nse_fo|123",
        ltp=100.0,
        bid=99.0,
        ask=101.0,
        volume=10.0,
        open_interest=20.0,
        oi_change=1.0,
    )
    assert isinstance(contract.strike, float)


def test_daily_store_loader_uses_only_captured_partitions(tmp_path):
    from datetime import date
    from marketdata.daily_store import DailyMarketStore, load_captured_candles

    store = DailyMarketStore(tmp_path)
    frame = pd.DataFrame({
        "timestamp": ["2026-10-01 09:15:00", "2026-10-01 09:20:00"],
        "open": [25000, 25001],
        "high": [25002, 25003],
        "low": [24999, 25000],
        "close": [25001, 25002],
        "volume": [100, 120],
    })
    store.save_candles("NIFTY", frame, date(2026, 10, 1))
    loaded, source = load_captured_candles("NIFTY", root=tmp_path)
    assert source == "KOTAK_CAPTURED"
    assert len(loaded) == 2
    assert set(loaded["data_source"]) == {"KOTAK_CAPTURED"}



def test_nifty_index_quote_uses_documented_name_not_numeric_scrip_token():
    class IndexNeo:
        def __init__(self):
            self.search_calls = []
            self.quote_calls = []

        def search_scrip(self, **kwargs):
            self.search_calls.append(kwargs)
            return [{"pSymbol": "26000", "pExchSeg": "nse_cm", "pSymbolName": "NIFTY 50", "pTrdSymbol": "NIFTY"}]

        def quotes(self, **kwargs):
            self.quote_calls.append(kwargs)
            return [{
                "ltp": "25000.5",
                "timestamp": "2026-10-09T10:00:00+05:30",
                "exchange_segment": "nse_cm",
            }]

    client = IndexNeo()
    provider = KotakNeoProvider(client)

    assert provider.resolve_nifty_index_neosymbol("NIFTY") == "nse_cm|Nifty 50"
    quote = provider.get_index_quote("Nifty 50")

    assert quote.ltp == 25000.5
    assert client.search_calls == []
    assert client.quote_calls[0]["instrument_tokens"] == [
        {"instrument_token": "Nifty 50", "exchange_segment": "nse_cm"}
    ]


def test_nifty_index_resolver_rejects_unknown_index_instead_of_guessing():
    provider = KotakNeoProvider(FakeNeo())
    try:
        provider.resolve_nifty_index_neosymbol("MADE UP INDEX")
    except ValueError as exc:
        assert "Unsupported Kotak index identifier" in str(exc)
    else:
        raise AssertionError("unknown index names must fail closed")


def test_kotak_limits_bridge_error_is_not_treated_as_account_data():
    class LimitsErrorNeo:
        def limits(self):
            return {"stat": "Not_Ok", "stCode": 300015, "errMsg": "bridge API error out"}

    try:
        KotakNeoProvider(LimitsErrorNeo()).get_account_state()
    except RuntimeError as exc:
        assert "bridge API error out" in str(exc)
    else:
        raise AssertionError("broker limits bridge errors must fail closed")
