from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from features.option_chain import OptionContract


@dataclass(frozen=True)
class OptionMarketContext:
    """Point-in-time option-chain context used as a directional confirmation gate.

    PCR is total put OI / total call OI.
    PCE is put premium exposure / call premium exposure, where exposure is
    LTP × OI. It is deliberately named PCE to distinguish it from PCR.
    Delta and gamma exposure are populated only when the source supplies
    contract Greeks; they are never inferred from missing expiry information.
    """

    pcr_oi: float | None
    pcr_volume: float | None
    pce: float | None
    net_delta: float | None
    gamma_exposure: float | None
    direction: str
    quality: str
    reasons: tuple[str, ...]


def build_option_context(contracts: Iterable[OptionContract]) -> OptionMarketContext:
    contracts = list(contracts)
    if not contracts:
        return OptionMarketContext(None, None, None, None, None, "WAIT", "RED", ("No option-chain contracts",))

    calls = [c for c in contracts if c.option_type == "CE"]
    puts = [c for c in contracts if c.option_type == "PE"]
    call_oi = sum(max(0.0, c.open_interest) for c in calls)
    put_oi = sum(max(0.0, c.open_interest) for c in puts)
    call_vol = sum(max(0.0, c.volume) for c in calls)
    put_vol = sum(max(0.0, c.volume) for c in puts)
    call_premium_exposure = sum(max(0.0, c.ltp) * max(0.0, c.open_interest) for c in calls)
    put_premium_exposure = sum(max(0.0, c.ltp) * max(0.0, c.open_interest) for c in puts)

    pcr_oi = put_oi / call_oi if call_oi > 0 else None
    pcr_volume = put_vol / call_vol if call_vol > 0 else None
    pce = put_premium_exposure / call_premium_exposure if call_premium_exposure > 0 else None

    greeks_available = all(hasattr(c, "gamma") for c in contracts)
    delta_values = [getattr(c, "delta", None) for c in contracts]
    gamma_values = [getattr(c, "gamma", None) for c in contracts]
    delta_available = all(v is not None for v in delta_values)
    gamma_available = all(v is not None for v in gamma_values)

    net_delta = None
    gamma_exposure = None
    if delta_available:
        net_delta = sum(
            c.open_interest * c.delta * (1.0 if c.option_type == "CE" else 1.0)
            for c in contracts
        )
    if gamma_available:
        gamma_exposure = sum(
            c.open_interest * c.gamma * (1.0 if c.option_type == "CE" else -1.0)
            for c in contracts
        )

    reasons = []
    if pcr_oi is not None:
        reasons.append(f"PCR(OI)={pcr_oi:.2f}")
    if pcr_volume is not None:
        reasons.append(f"PCR(volume)={pcr_volume:.2f}")
    if pce is not None:
        reasons.append(f"PCE={pce:.2f}")
    if not delta_available:
        reasons.append("Delta exposure unavailable from source")
    if not gamma_available:
        reasons.append("Gamma exposure unavailable from source")

    quality = "GREEN"
    if pcr_oi is None or pce is None:
        quality = "RED"
    elif not delta_available or not gamma_available:
        quality = "YELLOW"

    bullish = pcr_oi is not None and pce is not None and pcr_oi > 1.05 and pce > 1.05
    bearish = pcr_oi is not None and pce is not None and pcr_oi < 0.95 and pce < 0.95
    direction = "BUY" if bullish else "SELL" if bearish else "WAIT"
    return OptionMarketContext(pcr_oi, pcr_volume, pce, net_delta, gamma_exposure, direction, quality, tuple(reasons))
