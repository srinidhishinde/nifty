from datetime import date

import pandas as pd

from marketdata.daily_store import DailyMarketStore


def test_kotak_nifty_expiry_digits_are_not_treated_as_strike(tmp_path):
    store = DailyMarketStore(root=tmp_path)
    snapshot_dir = tmp_path / "NIFTY" / "2026-10-12"
    snapshot_dir.mkdir(parents=True)
    snapshot = pd.DataFrame(
        [
            {
                "captured_at": "2026-10-12T10:24:29.500355+05:30",
                "symbol": "NIFTY26O1320500CE",
                "exchange": "NFO",
                "underlying": "NIFTY",
                "expiry": "2026-10-13",
                "strike": 20500,
                "option_type": "CE",
                "instrument_token": "12345",
                "ltp": 100,
            }
        ]
    )
    snapshot.to_csv(snapshot_dir / "NIFTY_option_chain_latest.csv", index=False)

    loaded, captured_at = store.load_latest_option_chain_snapshot(
        "NIFTY",
        before=pd.Timestamp("2026-10-13T16:00:00+05:30"),
        session_date=date(2026, 10, 12),
    )

    assert len(loaded) == 1
    assert loaded.iloc[0]["strike"] == 20500
    assert loaded.iloc[0]["expiry"] == "2026-10-13"
    assert captured_at == pd.Timestamp("2026-10-12T10:24:29.500355+05:30")
