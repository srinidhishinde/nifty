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
import requests
from neo_api_client import NeoAPI

from config.settings import settings

IST = "Asia/Kolkata"
INTERVAL = "5min"
MAX_CHUNK_DAYS = 30
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


def _extract_futures_from_option_chain(response: Any) -> list[dict[str, str]]:
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
    return contracts


def _extract_scrip_master_url(response: Any) -> str | None:
    if isinstance(response, str) and response.startswith("http"):
        return response
    if not isinstance(response, dict):
        return None
    paths = response.get("filesPaths") or response.get("files_paths") or []
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        if isinstance(path, str) and "nse_fo" in path.lower() and path.startswith("http"):
            return path
    return None


def _discover_from_scrip_master(client: NeoAPI) -> list[dict[str, str]]:
    response = client.scrip_master(exchange_segment="nse_fo")
    url = _extract_scrip_master_url(response)
    if not url:
        raise RuntimeError("Kotak scrip_master returned no NSE F&O CSV URL.")
    download = requests.get(url, timeout=30)
    download.raise_for_status()
    from io import StringIO
    table = pd.read_csv(StringIO(download.text), low_memory=False)
    table.columns = [str(col).strip().rstrip(";") for col in table.columns]

    required = {"pSymbol", "pSymbolName", "pTrdSymbol", "pInstType", "pExchSeg"}
    missing = required.difference(table.columns)
    if missing:
        raise RuntimeError(
            "Kotak NSE F&O scrip master is missing columns: "
            + ", ".join(sorted(missing))
        )

    mask = (
        table["pExchSeg"].astype(str).str.lower().eq("nse_fo")
        & table["pSymbolName"].astype(str).str.upper().eq("NIFTY")
        & table["pInstType"].astype(str).str.upper().str.startswith("FUT")
    )
    rows = table.loc[mask]
    contracts: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in rows.iterrows():
        token = str(row["pSymbol"]).strip()
        symbol = str(row["pTrdSymbol"]).strip()
        expiry = str(row.get("pExpiryDate", "")).strip()
        if not token or token.lower() == "nan" or not symbol or symbol.lower() == "nan":
            continue
        neo_symbol = f"nse_fo|{token}"
        if neo_symbol not in seen:
            seen.add(neo_symbol)
            contracts.append({"symbol": symbol, "neo_symbol": neo_symbol, "expiry": expiry})

    if not contracts:
        raise RuntimeError("NSE F&O scrip master contained no active NIFTY futures.")
    return contracts


def discover_nifty_futures(client: NeoAPI) -> list[dict[str, str]]:
    response = client.option_chain(
        exchange="nse_fo",
        underlying="NIFTY",
        expiry=None,
        instrument_type="fut",
        count=40,
    )
    error = _error(response)
    if error:
        chain_error = error
    else:
        contracts = _extract_futures_from_option_chain(response)
        if contracts:
            return contracts
        chain_error = "option_chain returned no fut[] records"

    try:
        contracts = _discover_from_scrip_master(client)
        print("Kotak option_chain returned no futures; using NSE F&O scrip master fallback.")
        return contracts
    except Exception as fallback_error:
        raise RuntimeError(
            "Unable to discover active NIFTY futures. "
            f"option_chain: {chain_error}; "
            f"scrip_master fallback: {fallback_error}"
        ) from fallback_error

