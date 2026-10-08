from __future__ import annotations

import re
import pandas as pd

from features.option_chain import OptionContract
from strategy.market_specs import validate_option_strike

_FIELD_MAP = {
    "calls built up": ("CE", "built_up"), "calls vega": ("CE", "vega"), "calls theta": ("CE", "theta"),
    "calls delta": ("CE", "delta"), "calls volume": ("CE", "volume"), "calls chg in oi": ("CE", "oi_change"),
    "calls oi": ("CE", "open_interest"), "calls iv": ("CE", "implied_volatility"), "calls ltp (chg %)": ("CE", "ltp_change_pct"),
    "puts built up": ("PE", "built_up"), "puts vega": ("PE", "vega"), "puts theta": ("PE", "theta"),
    "puts delta": ("PE", "delta"), "puts volume": ("PE", "volume"), "puts chg in oi": ("PE", "oi_change"),
    "puts oi": ("PE", "open_interest"), "puts iv": ("PE", "implied_volatility"), "puts ltp (chg %)": ("PE", "ltp_change_pct"),
}


def _normalise_header(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def is_option_chain_snapshot(columns: object) -> bool:
    normalized = {_normalise_header(column) for column in columns}
    return "strike" in normalized and ("calls oi" in normalized or "puts oi" in normalized)


def parse_option_chain_csv(dataframe: pd.DataFrame) -> list[OptionContract]:
    if not is_option_chain_snapshot(dataframe.columns):
        raise ValueError("This file is not a supported option-chain snapshot. Expected Strike plus Calls/Puts OI columns.")

    lookup = {_normalise_header(column): column for column in dataframe.columns}
    strike_column = lookup["strike"]
    contracts: list[OptionContract] = []

    for _, row in dataframe.iterrows():
        try:
            strike = validate_option_strike("NIFTY", float(row[strike_column]))
        except (TypeError, ValueError):
            continue

        for option_type in ("CE", "PE"):
            values: dict[str, object] = {}
            for header, (side, field) in _FIELD_MAP.items():
                if side != option_type or header not in lookup:
                    continue
                raw = row[lookup[header]]
                if field == "built_up":
                    values[field] = "" if pd.isna(raw) else str(raw).strip()
                else:
                    value = pd.to_numeric(raw, errors="coerce")
                    values[field] = float(value) if pd.notna(value) else 0.0

            ltp = float(values.get("ltp", 0.0))
            if not any(key in values for key in ("ltp_change_pct", "volume", "open_interest")):
                continue

            contracts.append(
                OptionContract(
                    symbol=f"{option_type}_{int(strike) if strike.is_integer() else strike:g}",
                    expiry="",
                    strike=strike,
                    option_type=option_type,
                    ltp=ltp,
                    bid=ltp,
                    ask=ltp,
                    volume=float(values.get("volume", 0.0)),
                    open_interest=float(values.get("open_interest", 0.0)),
                    oi_change=float(values.get("oi_change", 0.0)),
                    implied_volatility=float(values.get("implied_volatility", 0.0)),
                    built_up=str(values.get("built_up", "")),
                    delta=float(values.get("delta", 0.0)),
                    theta=float(values.get("theta", 0.0)),
                    vega=float(values.get("vega", 0.0)),
                    ltp_change_pct=float(values.get("ltp_change_pct", 0.0)),
                )
            )

    if not contracts:
        raise ValueError("No usable CE/PE option contracts were found in the CSV.")
    return contracts


_NSE_EXPORT_HEADERS = {
    "strike": 11,
    "ce_oi": 1,
    "ce_oi_change": 2,
    "ce_volume": 3,
    "ce_iv": 4,
    "ce_ltp": 5,
    "ce_change": 6,
    "ce_bid": 8,
    "ce_ask": 9,
    "pe_bid": 13,
    "pe_ask": 14,
    "pe_change": 16,
    "pe_ltp": 17,
    "pe_iv": 18,
    "pe_volume": 19,
    "pe_oi_change": 20,
    "pe_oi": 21,
}


def _numeric_export_value(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "—", "NA", "N/A"}:
        return 0.0
    parsed = pd.to_numeric(text, errors="coerce")
    return float(parsed) if pd.notna(parsed) else 0.0


def is_nse_option_chain_export(dataframe: pd.DataFrame) -> bool:
    """Detect the NSE web-export format with a grouping row plus a 23-column header row."""
    if dataframe.shape[1] < 22 or dataframe.shape[0] < 2:
        return False

    header_values = {
        _normalise_header(value)
        for value in dataframe.iloc[1].tolist()
    }
    return "strike" in header_values and "oi" in header_values and "chng in oi" in header_values


def parse_nse_option_chain_export(
    dataframe: pd.DataFrame,
    *,
    expiry: str = "",
) -> list[OptionContract]:
    """Parse an NSE-style option-chain CSV exported with two header rows.

    The parser intentionally preserves real LTP, bid/ask, OI, change-in-OI,
    volume and IV. Missing values represented by '-' become zero. This path
    never creates synthetic premiums.
    """
    if not is_nse_option_chain_export(dataframe):
        raise ValueError(
            "This file is not a supported NSE option-chain export. "
            "Expected the two-row CALLS/NO TEXT/PUTS layout with a STRIKE column."
        )

    contracts: list[OptionContract] = []
    for _, row in dataframe.iloc[2:].iterrows():
        raw_strike = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["strike"]])
        try:
            strike = validate_option_strike("NIFTY", raw_strike)
        except (TypeError, ValueError):
            continue

        ce_ltp = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["ce_ltp"]])
        ce_bid = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["ce_bid"]])
        ce_ask = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["ce_ask"]])
        ce_volume = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["ce_volume"]])
        ce_oi = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["ce_oi"]])
        ce_oi_change = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["ce_oi_change"]])
        ce_iv = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["ce_iv"]])
        ce_change = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["ce_change"]])

        pe_ltp = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["pe_ltp"]])
        pe_bid = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["pe_bid"]])
        pe_ask = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["pe_ask"]])
        pe_volume = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["pe_volume"]])
        pe_oi = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["pe_oi"]])
        pe_oi_change = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["pe_oi_change"]])
        pe_iv = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["pe_iv"]])
        pe_change = _numeric_export_value(row.iloc[_NSE_EXPORT_HEADERS["pe_change"]])

        for option_type, values in (
            (
                "CE",
                (ce_ltp, ce_bid, ce_ask, ce_volume, ce_oi, ce_oi_change, ce_iv, ce_change),
            ),
            (
                "PE",
                (pe_ltp, pe_bid, pe_ask, pe_volume, pe_oi, pe_oi_change, pe_iv, pe_change),
            ),
        ):
            ltp, bid, ask, volume, open_interest, oi_change, iv, ltp_change = values
            if ltp <= 0 and open_interest <= 0 and volume <= 0:
                continue

            contracts.append(
                OptionContract(
                    symbol=f"{option_type}_{int(strike) if strike.is_integer() else strike:g}",
                    expiry=expiry,
                    strike=strike,
                    option_type=option_type,
                    ltp=ltp,
                    bid=bid,
                    ask=ask,
                    volume=volume,
                    open_interest=open_interest,
                    oi_change=oi_change,
                    implied_volatility=iv,
                    ltp_change_pct=ltp_change,
                )
            )

    if not contracts:
        raise ValueError("No usable CE/PE contracts were found in the NSE option-chain export.")
    return contracts
