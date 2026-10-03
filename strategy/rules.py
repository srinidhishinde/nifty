from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from features.technical.indicators import add_indicators


RULE_NAMES = (
    "rsi_oversold_buy",
    "macd_bullish",
    "bollinger_breakout",
    "ema_cross",
    "vwap_reversion",
    "order_book_imbalance",
    "sentiment_negative",
)


@dataclass(frozen=True)
class RuleSignal:
    rule: str
    direction: str
    reason: str


@dataclass(frozen=True)
class StrategySignal:
    direction: str
    rules: tuple[str, ...]
    reasons: tuple[str, ...]
    stop_loss: float
    target: float
    atr: float
    valid: bool


@dataclass(frozen=True)
class StrategyConfig:
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    volume_ma_period: int = 20
    macd_signal_period: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    ema_fast: int = 20
    ema_slow: int = 50
    vwap_deviation_pct: float = 2.0
    bid_ask_ratio_max: float = 0.5
    spread_widening_factor: float = 1.25
    sentiment_negative_threshold: float = -0.5
    stop_loss_pct: float = 0.015
    target_roi_pct: float = 0.40
    min_rules_for_signal: int = 1
    atr_risk_multiplier: float = 1.0


def _value(row: pd.Series, *names: str, default: float | None = None) -> float | None:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            try:
                return float(row[name])
            except (TypeError, ValueError):
                return default
    return default


def _previous(row: pd.Series, name: str, previous: pd.Series | None) -> float | None:
    if previous is None or name not in previous.index or pd.isna(previous[name]):
        return None
    try:
        return float(previous[name])
    except (TypeError, ValueError):
        return None


def evaluate_rules(
    row: pd.Series,
    previous: pd.Series | None = None,
    config: StrategyConfig | None = None,
) -> list[RuleSignal]:
    """Evaluate the seven pseudocode rules for one completed candle."""
    cfg = config or StrategyConfig()
    close = _value(row, "close")
    if close is None:
        return []

    signals: list[RuleSignal] = []

    rsi = _value(row, "RSI")
    volume = _value(row, "volume", default=0.0) or 0.0
    volume_ma = _value(row, "VOLUME_MA20")
    if rsi is not None and rsi < cfg.rsi_oversold and volume_ma is not None and volume > volume_ma:
        signals.append(RuleSignal("rsi_oversold_buy", "BUY", "RSI(14) < 30 with volume above MA(20)"))

    macd = _value(row, "MACD")
    macd_signal = _value(row, "MACD_SIGNAL", "Signal")
    if macd is not None and macd_signal is not None and macd > macd_signal and macd > 0:
        signals.append(RuleSignal("macd_bullish", "BUY", "MACD is above signal and zero"))

    upper = _value(row, "BB_UPPER", "UpperBand")
    sentiment = _value(row, "sentiment", "news_sentiment", "Sentiment")
    if upper is not None and sentiment is not None and close > upper and sentiment > 0:
        signals.append(RuleSignal("bollinger_breakout", "BUY", "Price above upper Bollinger Band with positive sentiment"))

    fast = _value(row, "EMA20")
    slow = _value(row, "EMA50")
    prev_fast = _previous(row, "EMA20", previous)
    prev_slow = _previous(row, "EMA50", previous)
    if None not in (fast, slow, prev_fast, prev_slow) and prev_fast <= prev_slow and fast > slow:
        signals.append(RuleSignal("ema_cross", "BUY", "EMA(20) crossed above EMA(50)"))

    vwap = _value(row, "VWAP")
    if vwap is not None and vwap > 0:
        deviation = (close - vwap) / vwap
        if deviation <= -cfg.vwap_deviation_pct / 100:
            signals.append(RuleSignal("vwap_reversion", "BUY", "Price is more than 2% below VWAP"))
        elif deviation >= cfg.vwap_deviation_pct / 100:
            signals.append(RuleSignal("vwap_reversion", "SELL", "Price is more than 2% above VWAP"))

    bid = _value(row, "bid", "best_bid", "Bid")
    ask = _value(row, "ask", "best_ask", "Ask")
    spread = _value(row, "spread", "Spread")
    previous_spread = _value(previous, "spread", "Spread") if previous is not None else None
    if bid is not None and ask is not None and ask > 0:
        ratio = bid / ask
        current_spread = spread if spread is not None else ask - bid
        widening = (
            previous_spread is not None
            and previous_spread > 0
            and current_spread > previous_spread * cfg.spread_widening_factor
        )
        if ratio < cfg.bid_ask_ratio_max and widening:
            signals.append(RuleSignal("order_book_imbalance", "SELL", "Bid/ask ratio below 0.5 with widening spread"))

    if sentiment is not None and sentiment < cfg.sentiment_negative_threshold:
        signals.append(RuleSignal("sentiment_negative", "SELL", "News sentiment below -0.5"))

    return signals


def generate_signal(
    data: pd.DataFrame,
    config: StrategyConfig | None = None,
) -> StrategySignal:
    """Evaluate the latest completed candle and calculate SL/TP."""
    if data.empty:
        return StrategySignal("WAIT", (), (), 0.0, 0.0, 0.0, False)

    cfg = config or StrategyConfig()
    enriched = add_indicators(data) if "MACD" not in data.columns else data.copy()
    row = enriched.iloc[-1]
    previous = enriched.iloc[-2] if len(enriched) > 1 else None
    rules = evaluate_rules(row, previous, cfg)

    if not rules:
        return StrategySignal("WAIT", (), (), 0.0, 0.0, 0.0, False)

    buys = [r for r in rules if r.direction == "BUY"]
    sells = [r for r in rules if r.direction == "SELL"]
    if buys and sells:
        return StrategySignal("WAIT", tuple(r.rule for r in rules), tuple(r.reason for r in rules), 0.0, 0.0, float(_value(row, "ATR", default=0.0) or 0.0), False)

    direction = "BUY" if buys else "SELL"
    close = float(row["close"])
    atr = float(_value(row, "ATR", default=0.0) or 0.0)

    if direction == "BUY":
        stop_loss = close * (1.0 - cfg.stop_loss_pct)
        target = close * (1.0 + cfg.target_roi_pct)
    else:
        stop_loss = close * (1.0 + cfg.stop_loss_pct)
        target = close * (1.0 - cfg.target_roi_pct)

    return StrategySignal(
        direction=direction,
        rules=tuple(r.rule for r in rules),
        reasons=tuple(r.reason for r in rules),
        stop_loss=round(stop_loss, 2),
        target=round(target, 2),
        atr=atr,
        valid=len(rules) >= cfg.min_rules_for_signal,
    )


def rank_rule_performance(trades: pd.DataFrame) -> pd.DataFrame:
    """Return per-rule trade count, win rate and ROI, sorted by net ROI."""
    if trades.empty:
        return pd.DataFrame(columns=["rule", "trades", "wins", "win_rate_pct", "net_pnl", "roi_pct"])

    rows = []
    for _, trade in trades.iterrows():
        rule_names = trade.get("rules")
        if not rule_names:
            rule_names = str(trade.get("rule", "")).split("|")
        if isinstance(rule_names, str):
            rule_names = [rule_names]
        for rule in rule_names:
            if rule:
                rows.append({
                    "rule": rule,
                    "pnl": float(trade.get("pnl", 0.0) or 0.0),
                    "roi_pct": float(trade.get("roi_pct", 0.0) or 0.0),
                })
    if not rows:
        return pd.DataFrame(columns=["rule", "trades", "wins", "win_rate_pct", "net_pnl", "roi_pct"])
    expanded = pd.DataFrame(rows)
    result = []
    for rule, group in expanded.groupby("rule"):
        pnl = pd.to_numeric(group["pnl"], errors="coerce").fillna(0.0)
        wins = int((pnl > 0).sum())
        result.append({
            "rule": rule,
            "trades": len(group),
            "wins": wins,
            "win_rate_pct": round(wins / len(group) * 100, 2),
            "net_pnl": round(float(pnl.sum()), 2),
            "roi_pct": round(float(group["roi_pct"].sum()), 2),
        })
    return pd.DataFrame(result).sort_values(["net_pnl", "win_rate_pct"], ascending=False).reset_index(drop=True)
