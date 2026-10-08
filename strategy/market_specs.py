from __future__ import annotations

from dataclasses import dataclass

# Current option strike intervals used by the dashboard's research chain.
# Live broker chains must use the broker/exchange-provided strike values rather
# than this table. MCX updates these parameters via contract specifications
# and circulars, so this mapping is intentionally isolated in one module.
OPTION_STRIKE_INTERVALS: dict[str, float] = {
    "NIFTY": 50.0,
    "BANKNIFTY": 100.0,
    "CRUDEOIL": 50.0,
    "CRUDEOILM": 50.0,
    "NATURALGAS": 5.0,
    "NATGASMINI": 5.0,
    "COPPER": 10.0,
    "GOLD": 500.0,
    "GOLDM": 500.0,
    "SILVER": 1000.0,
    "SILVERM": 1000.0,
}


def get_option_strike_step(instrument: str) -> float:
    """Return the configured option strike interval for an instrument."""
    key = str(instrument or "").strip().upper()
    try:
        return OPTION_STRIKE_INTERVALS[key]
    except KeyError as exc:
        raise ValueError(
            f"No option strike interval configured for instrument '{instrument}'."
        ) from exc


def round_to_strike(value: float, step: float) -> float:
    """Round a positive underlying value to the nearest valid strike."""
    price = float(value)
    increment = float(step)
    if price < 0 or increment <= 0:
        raise ValueError("price must be non-negative and step must be positive")
    return float(int(price / increment + 0.5) * increment)


@dataclass(frozen=True)
class FuturesContractSpec:
    instrument: str
    lot_size: int
    point_value: float
    tick_size: float
    margin_per_lot: float


def validate_futures_contract_spec(spec: FuturesContractSpec) -> None:
    if not spec.instrument or spec.lot_size <= 0 or spec.point_value <= 0 or spec.tick_size <= 0 or spec.margin_per_lot <= 0:
        raise ValueError("Futures contract specification requires positive instrument, lot_size, point_value, tick_size and margin_per_lot")


def validate_option_strike(
    instrument: str,
    strike: float,
    *,
    symbol_strike: float | None = None,
) -> float:
    """Validate a broker option strike without silently rescaling it.

    The broker contract identifier is authoritative when it contains a strike,
    but an obviously impossible NIFTY value is rejected rather than divided by
    the strike interval. This prevents malformed payloads/CSV columns from
    becoming tradable prices.
    """
    name = str(instrument or "").strip().upper()
    value = float(symbol_strike if symbol_strike and symbol_strike > 0 else strike)
    if value <= 0:
        raise ValueError(f"Invalid {name} option strike: {value}")
    if name == "NIFTY":
        step = OPTION_STRIKE_INTERVALS["NIFTY"]
        if value > 100_000 or abs((value / step) - round(value / step)) > 1e-9:
            raise ValueError(f"Invalid NIFTY option strike: {value}")
    return value
