from __future__ import annotations

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
