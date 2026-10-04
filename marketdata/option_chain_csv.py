from __future__ import annotations

import re
import pandas as pd

from features.option_chain import OptionContract

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
            strike = float(row[strike_column])
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
