import pandas as pd

from marketdata.daily_store import DailyMarketStore


def test_daily_store_partitions_by_instrument_and_date(tmp_path):
    store = DailyMarketStore(tmp_path)
    frame = pd.DataFrame([{
        "timestamp": "2026-10-06T09:15:00+05:30",
        "open": 25000, "high": 25010, "low": 24990, "close": 25005, "volume": 1000,
    }])
    path = store.save_candles("NIFTY", frame)
    assert path.parent.name == "2026-10-06"
    assert path.parent.parent.name == "NIFTY"
    assert path.exists()


def test_daily_store_rejects_missing_ohlcv(tmp_path):
    store = DailyMarketStore(tmp_path)
    frame = pd.DataFrame([{"timestamp": "2026-10-06", "close": 1}])
    try:
        store.save_candles("MCX", frame)
    except ValueError as exc:
        assert "Missing required OHLCV" in str(exc)
    else:
        raise AssertionError("Expected missing-column validation")


def test_mcx_resolution_uses_scrip_master_and_never_hardcodes_token():
    from marketdata.providers.kotak_neo import KotakNeoProvider

    class FakeClient:
        def search_scrip(self, **kwargs):
            assert kwargs["exchange_segment"] == "mcx_fo"
            assert kwargs["option_type"] == "FUT"
            return [
                {"pSymbol": 20, "pExchSeg": "mcx_fo", "pSymbolName": "CRUDEOIL",
                 "pTrdSymbol": "CRUDEOIL26OCTFUT", "pOptionType": "XX",
                 "pExpiryDate": "19OCT2026", "lLotSize": 100},
                {"pSymbol": 10, "pExchSeg": "mcx_fo", "pSymbolName": "CRUDEOIL",
                 "pTrdSymbol": "CRUDEOIL26NOVFUT", "pOptionType": "XX",
                 "pExpiryDate": "19NOV2026", "lLotSize": 100},
            ]

    contract = KotakNeoProvider(FakeClient()).resolve_mcx_futures("CRUDEOIL")
    assert contract["neosymbol"] == "mcx_fo|20"
    assert contract["instrument_token"] == "20"
    assert contract["trading_symbol"] == "CRUDEOIL26OCTFUT"
