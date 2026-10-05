from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from broker.kotak_neo import KotakNeoBroker
from config.settings import settings
from marketdata.daily_store import DailyMarketStore
from marketdata.kotak_live import FiveMinuteCandleBuilder, LiveTick, stream_kotak_sfeed
from marketdata.providers.kotak_neo import KotakNeoProvider


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture completed NIFTY/MCX 5-minute candles from Kotak SFeed."
    )
    parser.add_argument(
        "--totp",
        required=True,
        help="Current 6-digit Kotak TOTP. Never logged or persisted.",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=300,
        help="Capture window in seconds (5-86400).",
    )
    parser.add_argument(
        "--instrument",
        choices=["NIFTY", "CRUDEOIL", "BOTH"],
        default="BOTH",
        help="Instrument(s) to persist.",
    )
    args = parser.parse_args()

    if not 5 <= args.seconds <= 86_400:
        parser.error("--seconds must be between 5 and 86400.")
    if not args.totp.isdigit() or len(args.totp) != 6:
        parser.error("--totp must be exactly six digits.")

    if not settings.neo_consumer_key or not settings.neo_mobile or not settings.neo_ucc or not settings.neo_mpin:
        print("[FAIL] Configuration: Kotak Neo credentials are not fully configured.")
        return 2

    broker = KotakNeoBroker()
    connection = broker.authenticate(args.totp)
    if not connection.connected:
        print(f"[FAIL] Authentication: {connection.message}")
        return 1

    provider = KotakNeoProvider(broker.client)
    store = DailyMarketStore()
    builder = FiveMinuteCandleBuilder()

    try:
        mcx_contract = provider.resolve_mcx_futures("CRUDEOIL")
        print(
            "[PASS] MCX contract: "
            f"{mcx_contract['trading_symbol']} token={mcx_contract['instrument_token']} "
            f"expiry={mcx_contract['expiry']}"
        )

        selected = args.instrument
        want_nifty = selected in {"NIFTY", "BOTH"} and settings.capture_nifty_enabled
        want_mcx = selected in {"CRUDEOIL", "BOTH"} and settings.capture_mcx_enabled
        if not want_nifty and not want_mcx:
            print("[FAIL] No enabled capture instrument selected.")
            return 2

        counts = {"nifty": 0, "mcx": 0, "candles": 0}

        def on_tick(tick: LiveTick) -> None:
            completed = builder.update(tick)
            for candle in completed:
                instrument = str(candle["instrument"] if "instrument" in candle else tick.instrument)
                if tick.exchange_segment == "nse_cm":
                    instrument = "NIFTY"
                elif tick.exchange_segment == "mcx_fo":
                    instrument = "CRUDEOIL"

                if instrument == "NIFTY" and not want_nifty:
                    continue
                if instrument == "CRUDEOIL" and not want_mcx:
                    continue

                frame = pd.DataFrame([candle])
                session_date = candle["timestamp"].date()
                store.save_candles(instrument, frame, session_date)
                store.save_metadata(instrument, session_date, {
                    "source": "KOTAK_NEO_SFEED",
                    "instrument": instrument,
                    "timeframe": "5m",
                    "status": "CAPTURED",
                    "rows": 1,
                })
                counts["candles"] += 1

        tokens = [mcx_contract["instrument_token"]] if want_mcx else []
        stream_counts = __import__("asyncio").run(
            stream_kotak_sfeed(
                broker.client,
                tokens,
                args.seconds,
                on_tick,
            )
        )
        counts.update(stream_counts)

        # Persist only candles whose bucket has fully closed by the end of capture.
        for candle in builder.flush_completed(datetime.now().astimezone()):
            segment = "nse_cm" if candle["instrument"] == "NIFTY" else "mcx_fo"
            instrument = "NIFTY" if segment == "nse_cm" else "CRUDEOIL"
            if instrument == "NIFTY" and not want_nifty:
                continue
            if instrument == "CRUDEOIL" and not want_mcx:
                continue
            store.save_candles(instrument, pd.DataFrame([candle]), candle["timestamp"].date())
            counts["candles"] += 1

        print(
            "[PASS] Capture complete: "
            f"NIFTY ticks={counts['nifty']} MCX ticks={counts['mcx']} "
            f"completed_5m_candles={counts['candles']}"
        )
        print(f"[INFO] Persistent source: {settings.market_data_root}/*/*/*_ohlcv.csv")
        print("[INFO] Source label: KOTAK_CAPTURED")
        print("[INFO] Order submission: NOT ATTEMPTED")
        return 0
    except Exception as exc:
        print(f"[FAIL] Live capture: {exc}")
        return 1
    finally:
        broker.logout()


if __name__ == "__main__":
    raise SystemExit(main())
