"""Read-only Kotak Neo option-chain smoke test.

Validates that the real broker option-chain API returns usable CE/PE
contracts. It never places an order and never prints credentials.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from broker.kotak_neo import KotakNeoBroker
from config.settings import settings
from marketdata.providers.kotak_neo import KotakNeoProvider


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Kotak Neo option-chain test.")
    parser.add_argument("--totp", required=True, help="Current 6-digit Kotak TOTP. Never logged or persisted.")
    parser.add_argument("--underlying", choices=["NIFTY", "CRUDEOIL"], default="NIFTY")
    parser.add_argument("--exchange", choices=["NSE_FO", "MCX_FO"], default=None)
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()

    if not args.totp.isdigit() or len(args.totp) != 6:
        parser.error("--totp must be exactly six digits.")
    if args.count < 10 or args.count % 10:
        parser.error("--count must be a multiple of 10 and >= 10.")

    if not settings.neo_consumer_key or not settings.neo_mobile or not settings.neo_ucc or not settings.neo_mpin:
        print("[FAIL] Configuration: Kotak Neo credentials are not fully configured.")
        return 2

    exchange = args.exchange or ("NSE_FO" if args.underlying == "NIFTY" else "MCX_FO")
    broker = KotakNeoBroker()
    connection = broker.authenticate(args.totp)
    if not connection.connected:
        print(f"[FAIL] Authentication: {connection.message}")
        return 1

    try:
        provider = KotakNeoProvider(broker.client)
        contracts = provider.get_option_chain(
            underlying=args.underlying,
            exchange=exchange,
            count=args.count,
            enrich_quotes=False,
        )

        ce = [x for x in contracts if x.option_type == "CE"]
        pe = [x for x in contracts if x.option_type == "PE"]
        if not contracts or not ce or not pe:
            print(f"[FAIL] Live option chain: CE={len(ce)} PE={len(pe)} total={len(contracts)}")
            return 1

        valid = [
            x for x in contracts
            if float(x.strike or 0) > 0
            and float(x.ltp or 0) > 0
            and float(x.volume or 0) >= 0
            and float(x.open_interest or 0) >= 0
            and bool(x.expiry)
            and bool(x.instrument_token)
        ]
        if len(valid) != len(contracts):
            print(f"[FAIL] Contract validation: valid={len(valid)} total={len(contracts)}")
            return 1

        expiries = sorted({str(x.expiry) for x in contracts})
        strikes = sorted({float(x.strike) for x in contracts})
        print(f"[PASS] Authentication: connected to Kotak Neo ({connection.base_url or 'prod'})")
        print(f"[PASS] Live option chain: underlying={args.underlying} exchange={exchange} contracts={len(contracts)}")
        print(f"[PASS] Sides: CE={len(ce)} PE={len(pe)}")
        print(f"[PASS] Expiry: {expiries[0]}{' ... ' + expiries[-1] if len(expiries) > 1 else ''}")
        print(f"[PASS] Strike range: {strikes[0]:g} .. {strikes[-1]:g}")
        for side, rows in (("CE", ce), ("PE", pe)):
            rows = sorted(rows, key=lambda x: float(x.strike))
            sample = rows[len(rows) // 2]
            spread = None
            if sample.bid and sample.ask and sample.ask >= sample.bid:
                spread = f" spread={(sample.ask-sample.bid):.2f}"
            print(
                f"  {side} sample strike={sample.strike:g} LTP={sample.ltp:.2f}"
                f" OI={sample.open_interest:.0f} OIchg={sample.oi_change:.0f}"
                f" IV={sample.implied_volatility:.2f}{spread or ''}"
            )
        print("[PASS] Provenance: broker option-chain only; no synthetic/Yahoo substitution")
        print("OPTION CHAIN TEST: PASSED")
        print("Order submission: NOT ATTEMPTED")
        return 0
    except Exception as exc:
        print(f"[FAIL] Live option chain: {exc}")
        print("OPTION CHAIN TEST: FAILED")
        print("Order submission: NOT ATTEMPTED")
        return 1
    finally:
        broker.logout()


if __name__ == "__main__":
    raise SystemExit(main())
