"""Diagnose raw Kotak Neo option-chain responses without exposing credentials.

This is read-only. It does not place orders and does not substitute synthetic data.
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


def _print_diag(label: str, response: object) -> None:
    diag = KotakNeoProvider.option_chain_diagnostics(response)
    print(f"[{label}]")
    for key, value in diag.items():
        print(f"  {key}={value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Kotak Neo option-chain diagnostic.")
    parser.add_argument("--totp", required=True, help="Current 6-digit Kotak TOTP. Never logged or persisted.")
    parser.add_argument("--underlying", choices=["NIFTY", "CRUDEOIL"], default="NIFTY")
    parser.add_argument("--exchange", choices=["NSE_FO", "MCX_FO"], default=None)
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()

    if not args.totp.isdigit() or len(args.totp) != 6:
        parser.error("--totp must be exactly six digits.")
    if args.count < 10 or args.count % 10:
        parser.error("--count must be a positive multiple of 10.")

    exchange = args.exchange or ("NSE_FO" if args.underlying == "NIFTY" else "MCX_FO")
    exchange_segment = exchange.lower()
    broker = KotakNeoBroker()
    try:
        connection = broker.authenticate(args.totp)
        if not connection.connected or broker.client is None:
            print(f"[FAIL] Authentication: {connection.message}")
            return 1

        provider = KotakNeoProvider(broker.client)
        canonical = provider.resolve_option_underlying(args.underlying, exchange_segment)
        print(f"[PASS] Authentication: connected to {connection.base_url}")
        print(f"UNDERLYING: requested={args.underlying} canonical={canonical} exchange={exchange_segment}")

        expiry = None
        try:
            expiries_response = broker.client.expiries(
                exchange=exchange_segment,
                underlying=canonical,
                instrument_type="option",
            )
            _print_diag("EXPIRIES RAW", expiries_response)
            expiries_data = expiries_response.get("data") if isinstance(expiries_response, dict) else None
            if isinstance(expiries_data, list) and expiries_data:
                expiry = str(expiries_data[0])
            elif isinstance(expiries_data, dict):
                values = expiries_data.get("expiry") or expiries_data.get("expiries") or []
                if isinstance(values, list) and values:
                    expiry = str(values[0])
        except Exception as exc:
            print(f"[EXPIRIES EXCEPTION] {type(exc).__name__}: {exc}")

        print(f"SELECTED_EXPIRY: {expiry or 'server-default'}")

        try:
            raw_default = broker.client.option_chain(
                exchange=exchange_segment,
                underlying=canonical,
                expiry=None,
                instrument_type="option",
                count=args.count,
            )
            _print_diag("OPTION CHAIN DEFAULT", raw_default)
        except Exception as exc:
            print(f"[OPTION CHAIN DEFAULT EXCEPTION] {type(exc).__name__}: {exc}")

        if expiry:
            try:
                raw_explicit = broker.client.option_chain(
                    exchange=exchange_segment,
                    underlying=canonical,
                    expiry=expiry,
                    instrument_type="option",
                    count=args.count,
                )
                _print_diag("OPTION CHAIN EXPLICIT", raw_explicit)
            except Exception as exc:
                print(f"[OPTION CHAIN EXPLICIT EXCEPTION] {type(exc).__name__}: {exc}")

        return 0
    finally:
        broker.logout()


if __name__ == "__main__":
    raise SystemExit(main())
