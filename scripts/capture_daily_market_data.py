from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from broker.kotak_neo import KotakNeoBroker
from config.settings import settings
from marketdata.daily_store import DailyMarketStore
from marketdata.providers.kotak_neo import KotakNeoProvider


def capture_instrument(provider: KotakNeoProvider, store: DailyMarketStore, instrument: str, session_date: date) -> None:
    if instrument == "NIFTY":
        candles = provider.get_historical_candles(
            symbol="Nifty 50",
            exchange="NSE",
            timeframe="5m",
            start=session_date,
            end=session_date + timedelta(days=1),
        )
    else:
        raise RuntimeError(
            f"MCX candle mapping for {instrument} requires an explicit Neo symbol. "
            "Do not guess an MCX token; configure the instrument map first."
        )

    rows = [{
        "timestamp": c.timestamp, "open": c.open, "high": c.high,
        "low": c.low, "close": c.close, "volume": c.volume,
        "open_interest": c.open_interest,
    } for c in candles]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"Kotak Neo returned no {instrument} candles for {session_date}.")
    store.save_candles(instrument, frame, session_date)
    store.save_metadata(instrument, session_date, {
        "source": "KOTAK_NEO",
        "instrument": instrument,
        "timeframe": "5m",
        "rows": len(frame),
        "status": "CAPTURED",
    })


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture completed NIFTY/MCX data into daily research partitions.")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD; defaults to current Asia/Kolkata date.")
    args = parser.parse_args()

    session_date = date.fromisoformat(args.date) if args.date else pd.Timestamp.now(tz="Asia/Kolkata").date()
    broker = KotakNeoBroker()
    print("Authenticate Kotak Neo through the dashboard/session before running this collector.")
    if not broker.connection_status().connected:
        raise SystemExit("Kotak Neo is not connected. No data was written.")

    provider = KotakNeoProvider(broker.client)
    store = DailyMarketStore()

    if settings.capture_nifty_enabled:
        capture_instrument(provider, store, "NIFTY", session_date)

    if settings.capture_mcx_enabled:
        print("MCX capture is enabled but requires explicit configured Neo symbols; no synthetic MCX data will be created.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
