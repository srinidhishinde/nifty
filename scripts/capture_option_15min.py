"""Capture real Kotak ticks for one selected NIFTY option and persist completed 15-minute candles.

Example:
  python scripts/capture_option_15min.py --totp <6-digit-code> --expiry 2026-10-13 --option-type CE --strike 22700 --seconds 1800

Order submission is never attempted.
"""
from __future__ import annotations

import argparse
import asyncio
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
from marketdata.option_live_capture import FifteenMinuteCandleBuilder, stream_option_sfeed
from marketdata.providers.kotak_neo import KotakNeoProvider


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture genuine Kotak SFeed 15-minute candles for a selected NIFTY option.")
    parser.add_argument("--totp", required=True, help="Current 6-digit Kotak TOTP; never logged or persisted.")
    parser.add_argument("--expiry", required=True, help="Exact expiry from the live option-chain selector, YYYY-MM-DD.")
    parser.add_argument("--option-type", choices=["CE", "PE"], required=True)
    parser.add_argument("--strike", type=float, required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    args = parser.parse_args()
    if not args.totp.isdigit() or len(args.totp) != 6:
        parser.error("--totp must be exactly six digits.")
    if not 5 <= args.seconds <= 86_400:
        parser.error("--seconds must be between 5 and 86400.")
    if args.strike <= 0:
        parser.error("--strike must be positive.")
    if not all((settings.neo_consumer_key, settings.neo_mobile, settings.neo_ucc, settings.neo_mpin)):
        print("[FAIL] Kotak Neo credentials are not fully configured.")
        return 2

    broker = KotakNeoBroker()
    connection = broker.authenticate(args.totp)
    if not connection.connected:
        print(f"[FAIL] Authentication: {connection.message}")
        return 1
    try:
        provider = KotakNeoProvider(broker.client)
        contracts = provider.get_option_chain(
            underlying="NIFTY", exchange="NSE_FO", expiry=args.expiry,
            count=40, enrich_quotes=False,
        )
        candidates = [
            item for item in contracts
            if str(item.expiry) == args.expiry
            and str(item.option_type).upper() == args.option_type
            and abs(float(item.strike) - args.strike) < 0.001
        ]
        if not candidates:
            print("[FAIL] Selected expiry/side/strike was not returned by Kotak's option chain.")
            print("Check the expiry and whether the strike is in the returned chain; no substitute contract is used.")
            return 1
        contract = candidates[0]
        token = str(contract.instrument_token or "").strip()
        symbol = str(contract.symbol or "").strip()
        if not token or not symbol:
            print("[FAIL] Kotak contract is missing its instrument token or canonical trading symbol.")
            return 1
        print(f"[PASS] Selected contract: {symbol} token={token} expiry={contract.expiry} side={contract.option_type} strike={contract.strike}", flush=True)

        store = DailyMarketStore()
        builder = FifteenMinuteCandleBuilder()
        saved = 0

        def persist(rows: list[dict]) -> None:
            nonlocal saved
            for row in rows:
                if row["instrument_token"] != token or row["exchange_segment"] != "nse_fo":
                    continue
                frame = pd.DataFrame([row])
                store.save_candles(symbol, frame, row["timestamp"].date(), timeframe_minutes=15)
                saved += 1
                print(f"[CANDLE] {symbol} {row['timestamp']} O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f} ticks={row['tick_count']}", flush=True)

        counts = asyncio.run(stream_option_sfeed(
            broker.client, token, args.seconds,
            on_tick=lambda tick: persist(builder.update(tick)),
            progress_callback=lambda c: print(f"[INFO] Option feed heartbeat: ticks={c['ticks']} invalid={c['invalid']}", flush=True),
        ))
        persist(builder.flush_completed(datetime.now().astimezone()))
        print(f"[INFO] Contract={symbol} token={token} valid_ticks={counts['ticks']} invalid={counts['invalid']} saved_15m_candles={saved}", flush=True)
        print(f"[INFO] Saved path: {store.root / symbol.upper()} / YYYY-MM-DD / {symbol.upper()}_15min_ohlcv.csv", flush=True)
        print("[INFO] Order submission: NOT ATTEMPTED", flush=True)
        if counts["ticks"] == 0:
            print("[FAIL] No valid ticks for the selected option. Live 15-minute forecasting is not verified.", flush=True)
            return 1
        if saved == 0:
            print("[WARN] Ticks arrived but no completed 15-minute candle was saved yet. Run a longer capture spanning a 15-minute boundary (ideally at least 30 minutes).", flush=True)
            return 1
        print("[PASS] Selected-option 15-minute candle capture complete.", flush=True)
        return 0
    except Exception as exc:
        print(f"[FAIL] Option 15-minute capture: {exc}", flush=True)
        return 1
    finally:
        broker.logout()


if __name__ == "__main__":
    raise SystemExit(main())
