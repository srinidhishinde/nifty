"""Read-only Kotak Neo live market-data smoke test.

Authenticates with the existing broker adapter, resolves the current MCX
futures contract from the scrip master, validates REST quotes, then consumes
a bounded SFeed WebSocket window for NIFTY 50 and MCX CRUDEOIL.

This script NEVER places an order and NEVER prints credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow direct execution as python scripts/test_live_kotak_data.py by placing
# the repository root on sys.path before importing project packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from broker.kotak_neo import KotakNeoBroker
from config.settings import settings
from marketdata.kotak_live import FiveMinuteCandleBuilder, LiveTick, stream_kotak_sfeed
from marketdata.providers.kotak_neo import KotakNeoProvider


def _field(obj: Any, name: str, default: Any = None) -> Any:
    value = getattr(obj, name, default)
    return default if value is None else value


def _print_result(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


async def stream_live_data(client: Any, mcx_token: str, seconds: int) -> tuple[bool, dict[str, int]]:
    latest: dict[str, Any] = {}
    builder = FiveMinuteCandleBuilder()
    completed: list[dict[str, Any]] = []

    def on_tick(tick: LiveTick) -> None:
        key = "crudeoil" if tick.exchange_segment == "mcx_fo" else "nifty"
        latest[key] = (tick.trading_symbol, tick.ltp, tick.timestamp)
        completed.extend(builder.update(tick))

    counts = await stream_kotak_sfeed(client, [mcx_token], seconds, on_tick)
    completed.extend(builder.flush_completed(datetime.now().astimezone()))

    for key in ("nifty", "crudeoil"):
        value = latest.get(key)
        count = counts["mcx"] if key == "crudeoil" else counts["nifty"]
        if value:
            print(f"  {key.upper():9s} {value[0]} LTP={value[1]} ts={value[2]} messages={count}")
        else:
            print(f"  {key.upper():9s} no live SFeed message received; messages=0")

    print(f"  COMPLETED 5m candles={len(completed)}")
    ok = counts["nifty"] > 0 and counts["mcx"] > 0
    return ok, {"nifty": counts["nifty"], "crudeoil": counts["mcx"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Kotak Neo live data test.")
    parser.add_argument("--totp", required=True, help="Current 6-digit Kotak TOTP. Never logged or persisted.")
    parser.add_argument("--seconds", type=int, default=15, help="Live SFeed observation window (5-86400 seconds).")
    args = parser.parse_args()

    if not 5 <= args.seconds <= 86_400:
        parser.error("--seconds must be between 5 and 86400.")
    if not args.totp.isdigit() or len(args.totp) != 6:
        parser.error("--totp must be exactly six digits.")

    if not settings.neo_consumer_key or not settings.neo_mobile or not settings.neo_ucc or not settings.neo_mpin:
        print("[FAIL] Configuration: NEO_CONSUMER_KEY, NEO_MOBILE, NEO_UCC and NEO_MPIN must be configured.")
        return 2

    broker = KotakNeoBroker()
    connection = broker.authenticate(args.totp)
    if not connection.connected:
        print(f"[FAIL] Authentication: {connection.message}")
        return 1

    print(f"[PASS] Authentication: connected to Kotak Neo ({connection.base_url or 'prod'})")

    provider = KotakNeoProvider(broker.client)
    failures = 0

    # Current NIFTY index quote through Kotak REST.
    try:
        quote = provider.get_index_quote("Nifty 50")
        ok = quote.ltp > 0
        _print_result("NIFTY REST quote", ok, f"LTP={quote.ltp:.2f}")
        failures += not ok
    except Exception as exc:
        _print_result("NIFTY REST quote", False, str(exc))
        failures += 1

    # Resolve the current CRUDEOIL futures contract; never hardcode its token.
    try:
        contract = provider.resolve_mcx_futures("CRUDEOIL")
        print(
            "[PASS] MCX contract resolution: "
            f"{contract['trading_symbol']} token={contract['instrument_token']} "
            f"expiry={contract['expiry']} lot={contract['lot_size']}"
        )
    except Exception as exc:
        _print_result("MCX CRUDEOIL contract resolution", False, str(exc))
        broker.logout()
        return 1

    # Current MCX quote through Kotak REST.
    try:
        quote = provider.get_mcx_quote(contract)
        ok = quote["ltp"] > 0
        _print_result("MCX CRUDEOIL REST quote", ok, f"{contract['trading_symbol']} LTP={quote['ltp']:.2f}")
        failures += not ok
    except Exception as exc:
        _print_result("MCX CRUDEOIL REST quote", False, str(exc))
        failures += 1

    # Historical MCX is intentionally NOT called: Kotak documents mcx_fo
    # historical candles as unavailable. Live SFeed is the production path.
    print("[INFO] MCX 5m historical API check: SKIPPED by design (mcx_fo unsupported).")

    try:
        live_ok, counts = asyncio.run(
            stream_live_data(
                broker.client,
                contract["instrument_token"],
                args.seconds,
            )
        )
        _print_result(
            "SFeed live stream",
            live_ok,
            f"NIFTY messages={counts['nifty']}, CRUDEOIL messages={counts['crudeoil']}",
        )
        failures += not live_ok
    except Exception as exc:
        _print_result("SFeed live stream", False, str(exc))
        failures += 1
    finally:
        broker.logout()

    print()
    print("LIVE DATA TEST:", "PASSED" if failures == 0 else "FAILED")
    print("Order submission: NOT ATTEMPTED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
