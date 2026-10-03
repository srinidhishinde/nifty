from dataclasses import dataclass
from datetime import datetime

import numpy as np

from features.option_chain import (
    OptionContract,
    OptionAnalysis,
    analyze_option,
)
from strategy.ce_pe_selector import (
    MarketContext,
)


@dataclass(frozen=True)
class MarketSnapshot:
    instrument: str
    timeframe: str
    spot: float
    vwap: float
    context: MarketContext
    ce: OptionAnalysis
    pe: OptionAnalysis
    ml_probability: float


def _classify_trend(
    prices: np.ndarray,
) -> str:

    if len(prices) < 20:
        return "NEUTRAL"

    fast = float(np.mean(prices[-5:]))
    slow = float(np.mean(prices[-20:]))

    difference_pct = (
        (fast - slow)
        / slow
        * 100
    )

    if difference_pct > 0.08:
        return "BULLISH"

    if difference_pct < -0.08:
        return "BEARISH"

    return "NEUTRAL"


def _classify_momentum(
    prices: np.ndarray,
) -> str:

    if len(prices) < 6:
        return "NEUTRAL"

    change_pct = (
        (prices[-1] - prices[-6])
        / prices[-6]
        * 100
    )

    if change_pct > 0.10:
        return "POSITIVE"

    if change_pct < -0.10:
        return "NEGATIVE"

    return "NEUTRAL"


def _classify_volatility(
    prices: np.ndarray,
) -> str:

    if len(prices) < 20:
        return "NORMAL"

    returns = np.diff(prices) / prices[:-1]

    volatility = float(
        np.std(returns[-20:]) * 100
    )

    if volatility > 0.45:
        return "HIGH"

    if volatility < 0.15:
        return "LOW"

    return "NORMAL"


def _build_option(
    option_type: str,
    strike: float,
    spot: float,
    bias: float,
) -> OptionAnalysis:

    distance = abs(strike - spot)

    intrinsic = (
        max(spot - strike, 0)
        if option_type == "CE"
        else max(strike - spot, 0)
    )

    time_value = max(
        20.0,
        120.0 - distance * 0.8,
    )

    ltp = intrinsic + time_value

    volume = max(
        1000.0,
        25000.0 - distance * 100,
    )

    open_interest = 50000.0

    oi_change = bias * 5000.0

    iv = 18.0 + abs(bias) * 8.0

    contract = OptionContract(
        symbol=f"DEMO-{option_type}-{strike:.0f}",
        expiry="DEMO",
        strike=float(strike),
        option_type=option_type,
        ltp=float(ltp),
        bid=float(ltp - 1),
        ask=float(ltp + 1),
        volume=float(volume),
        open_interest=float(open_interest),
        oi_change=float(oi_change),
        implied_volatility=float(iv),
    )

    return analyze_option(contract)


def generate_snapshot(
    instrument: str,
    timeframe: str,
    seed: int | None = None,
) -> MarketSnapshot:

    rng = np.random.default_rng(seed)

    base_price = {
        "NIFTY": 25000.0,
        "CRUDEOIL": 6500.0,
        "NATURALGAS": 300.0,
        "COPPER": 1000.0,
        "SILVER": 95000.0,
        "GOLD": 125000.0,
    }.get(instrument, 25000.0)

    returns = rng.normal(
        0,
        base_price * 0.0008,
        100,
    )

    prices = base_price + np.cumsum(returns)

    spot = float(prices[-1])

    vwap = float(
        np.mean(prices[-30:])
    )

    trend = _classify_trend(prices)

    momentum = _classify_momentum(prices)

    volatility = _classify_volatility(prices)

    if spot > vwap * 1.0003:
        price_vs_vwap = "ABOVE"

    elif spot < vwap * 0.9997:
        price_vs_vwap = "BELOW"

    else:
        price_vs_vwap = "AT"

    context = MarketContext(
        timeframe=timeframe,
        trend=trend,
        momentum=momentum,
        price_vs_vwap=price_vs_vwap,
        volatility_regime=volatility,
    )

    # Synthetic probability for development only.
    directional_score = 0.50

    if trend == "BULLISH":
        directional_score += 0.08

    elif trend == "BEARISH":
        directional_score -= 0.08

    if momentum == "POSITIVE":
        directional_score += 0.07

    elif momentum == "NEGATIVE":
        directional_score -= 0.07

    directional_score += float(
        rng.normal(0, 0.04)
    )

    ml_probability = max(
        0.05,
        min(0.95, directional_score),
    )

    # Strike increment for the development feed.
    step = (
        50.0
        if instrument == "NIFTY"
        else 100.0
    )

    atm = round(
        spot / step
    ) * step

    ce_bias = (
        (ml_probability - 0.5)
        * 2
    )

    pe_bias = -ce_bias

    ce = _build_option(
        "CE",
        atm,
        spot,
        ce_bias,
    )

    pe = _build_option(
        "PE",
        atm,
        spot,
        pe_bias,
    )

    return MarketSnapshot(
        instrument=instrument,
        timeframe=timeframe,
        spot=spot,
        vwap=vwap,
        context=context,
        ce=ce,
        pe=pe,
        ml_probability=ml_probability,
    )
