from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable


def _finite(value: object, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def imbalance_ratio(
    bid_volume: float,
    ask_volume: float,
    liquidity_threshold: float,
    history: Iterable[float] = (),
    window: int = 5,
) -> float:
    total = max(0.0, _finite(bid_volume)) + max(0.0, _finite(ask_volume))
    if total < max(0.0, _finite(liquidity_threshold)):
        return 0.0
    raw = clamp((_finite(bid_volume) - _finite(ask_volume)) / (total + 1e-6), -1.0, 1.0)
    values = [clamp(_finite(v), -1.0, 1.0) for v in list(history)[-max(1, window - 1):]]
    return clamp(float(median(values + [raw])), -1.0, 1.0)


def infer_aggressor(
    last_trade_price: float,
    last_trade_direction: str | None,
    best_bid: float,
    best_ask: float,
    last_ticks: Iterable[float] = (),
    spread_width: float | None = None,
) -> str:
    ticks = [_finite(v) for v in last_ticks if _finite(v) > 0]
    if len(ticks) < 20:
        direction = str(last_trade_direction or "").lower()
        if direction in {"buy", "buyer", "b", "1"}:
            return "buyer"
        if direction in {"sell", "seller", "s", "-1"}:
            return "seller"
        return "neutral"

    spread = max(0.0, _finite(spread_width, _finite(best_ask) - _finite(best_bid)))
    mean = sum(ticks) / len(ticks)
    variance = sum((x - mean) ** 2 for x in ticks) / max(1, len(ticks) - 1)
    std = math.sqrt(max(0.0, variance))
    # EMA-like stabilization using the latest observation window.
    tick_volatility = 0.5 * std + 0.5 * abs(ticks[-1] - ticks[-2])
    tolerance = min(max(spread / 2.0, tick_volatility), 2.0 * spread) if spread > 0 else tick_volatility
    mid = (_finite(best_bid) + _finite(best_ask)) / 2.0
    price = _finite(last_trade_price)
    if price > mid + tolerance:
        return "buyer"
    if price < mid - tolerance:
        return "seller"
    return "neutral"


class AdaptiveMicroWeight:
    def __init__(self, path: str | Path = "logs/microstructure_weight.json", default: float = 0.15):
        self.path = Path(path)
        self.default = clamp(default, 0.10, 0.20)

    def load(self, key: str = "default") -> float:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return clamp(_finite(payload.get(key), self.default), 0.10, 0.20)
        except (FileNotFoundError, json.JSONDecodeError, AttributeError):
            return self.default

    def update(self, volatility_band: float, key: str = "default") -> float:
        previous = self.load(key)
        adjustment = _finite(volatility_band) * 0.1 - previous
        candidate = clamp(previous + adjustment, 0.10, 0.20)
        candidate = clamp(candidate, previous - 0.05, previous + 0.05)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {}
        payload[key] = round(candidate, 6)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        return candidate


def dynamic_imbalance_threshold(imbalance_history: Iterable[float], volatility_band: float) -> tuple[float, float]:
    values = [abs(clamp(_finite(v), -1.0, 1.0)) for v in imbalance_history]
    base_percentile = 80.0
    threshold_percentile = clamp(round(base_percentile + _finite(volatility_band) * 10.0), 70.0, 90.0)
    if len(values) < 20:
        # Avoid a noisy percentile trigger on tiny samples.
        return threshold_percentile, 1.0
    ordered = sorted(values)
    rank = (threshold_percentile / 100.0) * (len(ordered) - 1)
    lo, hi = int(math.floor(rank)), int(math.ceil(rank))
    trigger = ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)
    return threshold_percentile, clamp(trigger, 0.05, 1.0)


def microstructure_features(
    *,
    bid_volume: float = 0.0,
    ask_volume: float = 0.0,
    liquidity_threshold: float = float("inf"),
    imbalance_history: Iterable[float] = (),
    last_trade_price: float = 0.0,
    last_trade_direction: str | None = None,
    best_bid: float = 0.0,
    best_ask: float = 0.0,
    last_ticks: Iterable[float] = (),
    spread_width: float | None = None,
    volatility_band: float = 1.0,
) -> dict:
    history = list(imbalance_history)
    ratio = imbalance_ratio(bid_volume, ask_volume, liquidity_threshold, history)
    aggressor = infer_aggressor(
        last_trade_price, last_trade_direction, best_bid, best_ask, last_ticks, spread_width
    )
    percentile, trigger = dynamic_imbalance_threshold(history + [ratio], volatility_band)
    triggered = abs(ratio) >= trigger and abs(ratio) > 0.0
    return {
        "imbalance_ratio": round(ratio, 6),
        "aggressor": aggressor,
        "volatility_band": round(clamp(_finite(volatility_band), 0.0, 10.0), 6),
        "micro_weight": 0.15,
        "threshold_percentile": round(percentile, 2),
        "imbalance_trigger": round(trigger, 6),
        "imbalance_triggered": bool(triggered),
    }
