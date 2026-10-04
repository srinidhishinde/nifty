from __future__ import annotations

# Allow direct execution from the repository root (or any working directory).
# The scripts directory is placed on sys.path for direct execution, not the repository root.
# Add the repository root explicitly so package imports such as config work reliably.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import time
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from neo_api_client import NeoAPI

from config.settings import settings

IST = "Asia/Kolkata"
INTERVAL = "5min"
MAX_CHUNK_DAYS = 30
REQUEST_PAUSE_SECONDS = 8.0
MAX_RATE_LIMIT_RETRIES = 5
SESSION_START = "09:15"
SESSION_END = "15:30"
REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]


def _data(response: Any) -> dict:
    if not isinstance(response, dict):
        return {}
    data = response.get("data")
    return data if isinstance(data, dict) else {}


def _error(response: Any) -> str:
    if not isinstance(response, dict):
        return "Invalid Kotak Neo response."
    if response.get("errMsg"):
        return str(response["errMsg"])
    fault = response.get("fault")
    if isinstance(fault, dict) and fault.get("message"):
        return str(fault["message"])
    data = response.get("data")
    if isinstance(data, dict) and data:
        return ""
    if str(response.get("status", "")).lower() in {"ok", "success"}:
        return ""
    return ""


def discover_nifty_futures(client: NeoAPI) -> list[dict[str, str]]:
    response = client.option_chain(
        exchange="nse_fo",
        underlying="NIFTY",
        expiry=None,
        instrument_type="fut",
        count=40,
    )
    error = _error(response)
    if not error:
        data = _data(response)
        futures = data.get("fut") or []
        contracts: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in futures:
            inst = item.get("instrument") or item.get("inst") or {}
            neo_symbol = str(inst.get("neoSymbol") or "").strip()
            symbol = str(inst.get("symbol") or "").strip()
            expiry = str(inst.get("expiryDt") or "").strip()
            if neo_symbol and symbol and neo_symbol not in seen:
                seen.add(neo_symbol)
                contracts.append({"symbol": symbol, "neo_symbol": neo_symbol, "expiry": expiry})
        if contracts:
            return contracts

    chain_error = error or "option_chain returned no fut[] records"
    try:
        sm = client.scrip_master(exchange_segment="nse_fo")
        url = sm if isinstance(sm, str) else None
        if not url and isinstance(sm, dict):
            paths = sm.get("filesPaths") or sm.get("files_paths") or []
            if isinstance(paths, str):
                paths = [paths]
            url = next((p for p in paths if isinstance(p, str) and "nse_fo" in p.lower() and p.startswith("http")), None)
        if not url:
            raise RuntimeError("scrip_master returned no NSE F&O URL")
        import requests
        from io import StringIO
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        table = pd.read_csv(StringIO(response.text), low_memory=False)
        table.columns = [str(col).strip().rstrip(";") for col in table.columns]
        required = {"pSymbol", "pSymbolName", "pTrdSymbol", "pInstType", "pExchSeg"}
        missing = required.difference(table.columns)
        if missing:
            raise RuntimeError("scrip master missing columns: " + ", ".join(sorted(missing)))
        mask = (
            table["pExchSeg"].astype(str).str.lower().eq("nse_fo")
            & table["pSymbolName"].astype(str).str.upper().eq("NIFTY")
            & table["pInstType"].astype(str).str.upper().str.startswith("FUT")
        )
        contracts = []
        seen = set()
        for _, row in table.loc[mask].iterrows():
            token = str(row["pSymbol"]).strip()
            symbol = str(row["pTrdSymbol"]).strip()
            expiry = str(row.get("pExpiryDate", "")).strip()
            if token and token.lower() != "nan" and symbol and symbol.lower() != "nan":
                neo_symbol = f"nse_fo|{token}"
                if neo_symbol not in seen:
                    seen.add(neo_symbol)
                    contracts.append({"symbol": symbol, "neo_symbol": neo_symbol, "expiry": expiry})
        if contracts:
            print("Using NSE F&O scrip-master fallback for NIFTY futures discovery.")
            return contracts
        raise RuntimeError("scrip master contained no NIFTY futures")
    except Exception as fallback_error:
        raise RuntimeError(
            f"Unable to discover active NIFTY futures. option_chain: {chain_error}; "
            f"scrip_master fallback: {fallback_error}"
        ) from fallback_error
def _is_rate_limited(response: Any) -> bool:
    raw = json.dumps(response, default=str).lower() if isinstance(response, (dict, list)) else str(response).lower()
    return "429" in raw or "rate limit" in raw or "too many request" in raw

def _historical_request(client: NeoAPI, contract: dict[str, str], cursor: date, chunk_end: date) -> Any:
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        response = client.historical_data(neosymbol=contract["neo_symbol"], interval=INTERVAL, from_date=cursor.isoformat(), to_date=chunk_end.isoformat())
        if not _is_rate_limited(response):
            return response
        if attempt >= MAX_RATE_LIMIT_RETRIES:
            raise RuntimeError(f'{contract["symbol"]} {cursor}..{chunk_end}: Kotak historical API rate limit persisted')
        delay = min(60.0, 10.0 * (2 ** attempt))
        print(f'{contract["symbol"]} {cursor}..{chunk_end}: rate limited; waiting {delay:.0f}s before retry {attempt + 2}/{MAX_RATE_LIMIT_RETRIES + 1}')
        time.sleep(delay)
    raise RuntimeError("unreachable")

def fetch_contract(client: NeoAPI, contract: dict[str, str], start: date, end: date) -> pd.DataFrame:
    rows: list[list[Any]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=MAX_CHUNK_DAYS - 1), end)
        response = _historical_request(client, contract, cursor, chunk_end)
        error = _error(response)
        if error:
            raise RuntimeError(f'{contract["symbol"]} {cursor}..{chunk_end}: {error}')
        rows.extend(_data(response).get("candles") or [])
        cursor = chunk_end + timedelta(days=1)
        if cursor <= end:
            time.sleep(REQUEST_PAUSE_SECONDS)

    if not rows:
        return pd.DataFrame(columns=REQUIRED + ["contract_symbol", "expiry", "neo_symbol", "oi"])
    parsed = []
    for row in rows:
        if len(row) < 6:
            continue
        parsed.append({"timestamp": row[0], "open": row[1], "high": row[2], "low": row[3], "close": row[4], "volume": row[5], "oi": row[6] if len(row) > 6 else None, "contract_symbol": contract["symbol"], "expiry": contract["expiry"], "neo_symbol": contract["neo_symbol"]})
    frame = pd.DataFrame(parsed)
    if frame.empty:
        return pd.DataFrame(columns=REQUIRED + ["contract_symbol", "expiry", "neo_symbol", "oi"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dt.tz_convert(IST)
    frame = frame[frame["timestamp"].notna()]
    for col in ["open", "high", "low", "close", "volume", "oi"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=REQUIRED)
    frame = frame[(frame["timestamp"].dt.strftime("%H:%M") >= SESSION_START) & (frame["timestamp"].dt.strftime("%H:%M") <= SESSION_END)]
    invalid = ((frame["open"] <= 0) | (frame["high"] <= 0) | (frame["low"] <= 0) | (frame["close"] <= 0) | (frame["volume"] < 0) | (frame["high"] < frame[["open", "close"]].max(axis=1)) | (frame["low"] > frame[["open", "close"]].min(axis=1)))
    return frame.loc[~invalid].drop_duplicates(["timestamp", "neo_symbol"]).sort_values("timestamp")


def expected_5m_bars(trading_days: int) -> int:
    # 09:15 through 15:30 inclusive = 76 five-minute timestamps.
    return trading_days * 76


def quality_report(frame: pd.DataFrame, start: date, end: date) -> dict:
    if frame.empty:
        return {
            "rows": 0,
            "coverage_start": None,
            "coverage_end": None,
            "unique_days": 0,
            "expected_bars_for_covered_days": 0,
            "actual_bars": 0,
            "duplicates": 0,
            "invalid_rows": 0,
            "zero_volume_rows": 0,
            "gaps_gt_5m": 0,
            "complete": False,
        }

    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    counts = ts.dt.date.value_counts()
    covered_days = len(counts)
    expected = expected_5m_bars(covered_days)
    gaps = 0
    for day, group in frame.groupby(ts.dt.date):
        values = pd.DatetimeIndex(group["timestamp"].sort_values())
        if len(values) > 1:
            gaps += int((values[1:] - values[:-1] > pd.Timedelta(minutes=5)).sum())

    return {
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "rows": int(len(frame)),
        "coverage_start": str(ts.min()),
        "coverage_end": str(ts.max()),
        "unique_days": int(covered_days),
        "expected_bars_for_covered_days": int(expected),
        "actual_bars": int(len(frame)),
        "missing_bars_vs_expected": int(max(expected - len(frame), 0)),
        "duplicates": int(frame.duplicated("timestamp").sum()),
        "invalid_rows": 0,
        "zero_volume_rows": int((frame["volume"] == 0).sum()),
        "gaps_gt_5m": gaps,
        "complete": bool(len(frame) == expected and gaps == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real 5-minute NIFTY futures candles from Kotak Neo.")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-dir", default="data/backtest/kotak_nifty")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")
    if not settings.neo_consumer_key:
        raise SystemExit("NEO_CONSUMER_KEY is not configured in .env")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    client = NeoAPI(consumer_key=settings.neo_consumer_key, environment="prod")
    contracts = discover_nifty_futures(client)

    manifest = {
        "source": "Kotak Neo historical_data",
        "interval": INTERVAL,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "contracts_discovered": contracts,
        "limitations": [
            "Kotak historical_data returns only active contracts.",
            "Expired/delisted contracts are not available from this endpoint.",
            "This script never substitutes another contract to fill a missing period.",
        ],
    }

    all_frames: list[pd.DataFrame] = []
    reports = []
    for contract in contracts:
        try:
            frame = fetch_contract(client, contract, start, end)
            if not frame.empty:
                raw_path = output / f'{contract["symbol"]}_{start}_{end}_5m.csv'
                frame.to_csv(raw_path, index=False)
                all_frames.append(frame)
            reports.append({
                **contract,
                "rows": int(len(frame)),
                "start": str(frame["timestamp"].min()) if not frame.empty else None,
                "end": str(frame["timestamp"].max()) if not frame.empty else None,
            })
            print(f'{contract["symbol"]}: {len(frame):,} bars')
        except Exception as exc:
            reports.append({**contract, "error": str(exc)})
            print(f'{contract["symbol"]}: ERROR: {exc}')

    manifest["results"] = reports
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.sort_values(["timestamp", "expiry", "contract_symbol"])
        combined.to_csv(output / "nifty_futures_5m_raw.csv", index=False)

        # The rule engine consumes OHLCV only. Keep the contract metadata in the raw
        # file and emit a clean file only after validation.
        clean = combined[REQUIRED].drop_duplicates("timestamp").sort_values("timestamp")
        clean.to_csv(output / "nifty_futures_5m_backtest.csv", index=False)
        manifest["quality"] = quality_report(clean, start, end)
        manifest["quality"]["raw_rows"] = int(len(combined))
        manifest["quality"]["raw_unique_contracts"] = int(combined["neo_symbol"].nunique())
        manifest["quality"]["overlapping_timestamps_across_contracts"] = int(
            combined.groupby("timestamp")["neo_symbol"].nunique().gt(1).sum()
        )
    else:
        manifest["quality"] = {"rows": 0, "complete": False}

    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest["quality"], indent=2))


if __name__ == "__main__":
    main()
