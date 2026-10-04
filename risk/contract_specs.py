from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractSpec:
    instrument: str
    lot_size: int
    tick_size: float
    tick_value: float
    multiplier: float = 1.0


# Defaults are intentionally explicit and overridable. Broker/exchange master data
# should replace these values before live trading; unknown contracts must be blocked.
KNOWN_CONTRACTS: dict[str, ContractSpec] = {
    "NIFTY": ContractSpec("NIFTY", lot_size=1, tick_size=0.05, tick_value=0.05),
}


def validate_contract_spec(spec: ContractSpec) -> None:
    if spec.lot_size <= 0 or spec.tick_size <= 0 or spec.tick_value <= 0 or spec.multiplier <= 0:
        raise ValueError("Contract specification values must be positive")


def risk_per_lot(stop_distance: float, spec: ContractSpec) -> float:
    validate_contract_spec(spec)
    if stop_distance <= 0:
        raise ValueError("stop_distance must be positive")
    ticks = stop_distance / spec.tick_size
    return ticks * spec.tick_value * spec.lot_size * spec.multiplier


def lots_for_risk(risk_amount: float, stop_distance: float, spec: ContractSpec) -> int:
    if risk_amount <= 0:
        return 0
    per_lot = risk_per_lot(stop_distance, spec)
    return max(0, int(risk_amount // per_lot))


def quantity_for_risk(risk_amount: float, stop_distance: float, spec: ContractSpec) -> int:
    return lots_for_risk(risk_amount, stop_distance, spec) * spec.lot_size
