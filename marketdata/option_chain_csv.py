from __future__ import annotations

import re

import pandas as pd

from features.option_chain import OptionContract


_FIELD_MAP = {
    "calls built up": ("CE", "built_up"),
    "calls vega": ("CE", "vega"),
    "calls theta": ("CE", "theta"),
    "calls delta": ("CE", "delta"),
    "calls volume": ("CE", "volume"),
    "calls chg in oi": ("CE", "oi_change"),
    "calls oi": ("CE", "open_interest"),
    "calls iv": ("CE", "implied_volatility"),
    "calls ltp (chg %)": ("CE", "ltp_change_pct"),
    "puts built up": ("PE", "built_up"),
    "puts vega": ("PE", "vega"),
    "puts theta": ("PE", "theta"),
    "puts delta": ("PE", "delta"),
    "puts volume": ("PE", "volume"),
    "puts chg in oi": ("PE", "oi_change"),
    "puts oi": ("PE", "open_interest"),
    "puts iv": ("PE", "implied_volatility"),
    "puts ltp (chg %)": ("PE", "ltp_change_pct"),
}


def _normalise_header(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def is_option_chain_snapshot(columns: object) -> bool:
    """Return True for the wide CE/PE snapshot format exported by broker tools."""
    normalized = {_normalise_header(column) for column in columns}
    return "strike" in normalized and (
        "calls oi" in normalized or "puts oi" in normalized
    )


def parse_option_chain_csv(dataframe: pd.DataFrame) -> list[OptionContract]:
    """Convert a wide CE/PE option-chain snapshot into normalized contracts.

    This is intentionally an option-chain *snapshot* parser. It does not turn
    option-chain rows into OHLCV candles because a snapshot has no candle
    timestamp/open/high/low/close and therefore cannot support a valid
    historical candle backtest.
    """
    if not is_option_chain_snapshot(dataframe.columns):
        raise ValueError(
            "This file is not a supported option-chain snapshot. "
            "Expected Strike plus Calls/Puts OI columns."
        )

    lookup = {_normalise_header(column): column for column in dataframe.columns}
    strike_column = lookup["strike"]
    contracts: list[OptionContract] = []

    for _, row in dataframe.iterrows():
        try:
            strike = float(row[strike_column])
        except (TypeError, ValueError):
            continue

        for option_type in ("CE", "PE"):
            values = {}
            for header, (side, field) in _FIELD_MAP.items():
                if side != option_type or header not in lookup:
                    continue
                value = pd.to_numeric(row[lookup[header]], errors="coerce")
                values[field] = float(value) if pd.notna(value) else 0.0

            # The export does not contain bid/ask/expiry/symbol. Using LTP as
            # both sides avoids inventing a spread and makes it explicit that
            # order-book rules cannot be evaluated from this snapshot.
            ltp = values.get("ltp", 0.0)
            if not any(
                key in values
                for key in ("ltp_change_pct", "volume", "open_interest")
            ):
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
                    volume=values.get("volume", 0.0),
                    open_interest=values.get("open_interest", 0.0),
                    oi_change=values.get("oi_change", 0.0),
                    implied_volatility=values.get("implied_volatility", 0.0),
                    built_up=values.get("built_up", ""),
                    delta=values.get("delta", 0.0),
                    theta=values.get("theta", 0.0),
                    vega=values.get("vega", 0.0),
                    ltp_change_pct=values.get("ltp_change_pct", 0.0),
                )
            )

    if not contracts:
        raise ValueError("No usable CE/PE option contracts were found in the CSV.")

    return contracts
