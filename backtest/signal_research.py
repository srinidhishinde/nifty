from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from features.technical.indicators import add_indicators
from strategy.rules import StrategyConfig, _generate_signal_from_enriched


@dataclass(frozen=True)
class SignalResearchResult:
    signals: pd.DataFrame
    validation: dict[str, Any]


def _normalise_frame(data: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if getattr(frame["timestamp"].dt, "tz", None) is None:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert("Asia/Kolkata")

    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=list(required))
    invalid = (
        (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (frame["volume"] < 0)
        | (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
    )
    if invalid.any():
        raise ValueError(f"Invalid OHLCV rows: {int(invalid.sum())}")

    return (
        frame.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


def _resolve_outcome(
    frame: pd.DataFrame,
    entry_index: int,
    direction: str,
    stop: float,
    target: float,
) -> tuple[float, str, pd.Timestamp]:
    entry_day = frame.iloc[entry_index].timestamp.date()
    for j in range(entry_index, len(frame)):
        row = frame.iloc[j]
        if row.timestamp.date() != entry_day:
            break
        if row.timestamp.time() > pd.Timestamp("15:40").time():
            break

        if direction == "BUY":
            stop_hit = float(row.low) <= stop
            target_hit = float(row.high) >= target
        else:
            stop_hit = float(row.high) >= stop
            target_hit = float(row.low) <= target

        # Conservative OHLC rule: when both are touched, stop wins.
        if stop_hit:
            return stop, "stop_loss", row.timestamp
        if target_hit:
            return target, "take_profit", row.timestamp

    last = frame[
        (frame["timestamp"].dt.date == entry_day)
        & (frame["timestamp"].dt.time <= pd.Timestamp("15:40").time())
    ]
    if last.empty:
        row = frame.iloc[entry_index]
    else:
        row = last.iloc[-1]
    return float(row.close), "market_close", row.timestamp


def run_signal_research(
    data: pd.DataFrame,
    config: StrategyConfig | None = None,
    evaluation_start: pd.Timestamp | None = None,
    evaluation_end: pd.Timestamp | None = None,
) -> SignalResearchResult:
    """Evaluate the rule engine on spot/index candles without pretending they are futures P&L.

    Signals are generated from a completed candle and use the next candle open as the
    theoretical entry. Outcomes are reported in index points/R, never rupees.
    """

    c = config or StrategyConfig()
    frame = add_indicators(_normalise_frame(data))

    start = pd.Timestamp(evaluation_start) if evaluation_start is not None else frame.timestamp.min()
    end = pd.Timestamp(evaluation_end) if evaluation_end is not None else frame.timestamp.max()
    if start.tzinfo is None:
        start = start.tz_localize("Asia/Kolkata")
    else:
        start = start.tz_convert("Asia/Kolkata")
    if end.tzinfo is None:
        end = end.tz_localize("Asia/Kolkata")
    else:
        end = end.tz_convert("Asia/Kolkata")

    rows: list[dict[str, Any]] = []
    counters = {
        "bars_considered": 0,
        "rule_trigger_bars": 0,
        "conflicting_signal_bars": 0,
        "qualified_signal_bars": 0,
        "rejected_warmup": 0,
        "rejected_atr": 0,
        "rejected_no_next_bar": 0,
        "rejected_overlap": 0,
    }

    active_until: pd.Timestamp | None = None
    for i in range(1, len(frame) - 1):
        row = frame.iloc[i]
        prev = frame.iloc[i - 1]
        ts = row.timestamp
        if ts < start or ts > end:
            continue
        if ts.time() < pd.Timestamp("09:15").time() or ts.time() > pd.Timestamp("15:40").time():
            continue

        counters["bars_considered"] += 1
        if active_until is not None and ts <= active_until:
            counters.setdefault("rejected_overlap", 0)
            counters["rejected_overlap"] += 1
            continue
        if i < c.min_history_bars:
            counters["rejected_warmup"] += 1
            continue

        signal = _generate_signal_from_enriched(frame.iloc[: i + 1], c)
        if signal.direction == "WAIT" or not signal.valid:
            continue

        direction = signal.direction
        signals = list(signal.rules)
        counters["rule_trigger_bars"] += 1
        atr = float(signal.atr)
        atr_pct = float(row.get("ATR_PCT", 0) or 0)
        if atr_pct > c.max_atr_pct:
            counters["rejected_atr"] += 1
            continue
        next_row = frame.iloc[i + 1]
        if next_row.timestamp.date() != ts.date():
            counters["rejected_no_next_bar"] += 1
            continue

        entry = float(next_row.open)
        risk_distance = max(
            entry * c.stop_loss_pct,
            atr * c.atr_stop_multiple if atr > 0 else 0,
        )
        reward_distance = max(
            entry * c.min_target_pct,
            atr * c.target_atr_multiple if atr > 0 else 0,
            risk_distance * c.min_reward_risk,
        )
        stop = entry - risk_distance if direction == "BUY" else entry + risk_distance
        target = entry + reward_distance if direction == "BUY" else entry - reward_distance
        exit_price, reason, exit_time = _resolve_outcome(
            frame, i + 1, direction, stop, target
        )
        signed_points = (
            exit_price - entry if direction == "BUY" else entry - exit_price
        )
        r_multiple = signed_points / risk_distance if risk_distance > 0 else 0.0
        counters["qualified_signal_bars"] += 1

        active_until = exit_time
        rows.append(
            {
                "signal_time": ts,
                "entry_time": next_row.timestamp,
                "exit_time": exit_time,
                "direction": direction,
                "confidence": round(signal.confidence, 2),
                "rules": "|".join(signals),
                "entry": round(entry, 2),
                "stop_loss": round(stop, 2),
                "target": round(target, 2),
                "exit": round(exit_price, 2),
                "exit_reason": reason,
                "risk_points": round(risk_distance, 2),
                "reward_points": round(reward_distance, 2),
                "points": round(signed_points, 2),
                "R": round(r_multiple, 3),
                "atr_pct": round(atr_pct * 100, 3),
            }
        )

    signals_df = pd.DataFrame(rows)
    if signals_df.empty:
        total = wins = losses = 0
        avg_r = total_r = 0.0
    else:
        r = pd.to_numeric(signals_df["R"], errors="coerce").fillna(0.0)
        total = len(signals_df)
        wins = int((r > 0).sum())
        losses = int((r < 0).sum())
        avg_r = float(r.mean())
        total_r = float(r.sum())

    validation = {
        "status": "PASS",
        "mode": "NIFTY_SPOT_SIGNAL_RESEARCH",
        "rows": len(frame),
        "evaluation_start": str(start),
        "evaluation_end": str(end),
        "trading_days": int(frame.timestamp.dt.date.nunique()),
        **counters,
        "signals": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / total * 100, 2) if total else 0.0,
        "average_R": round(avg_r, 3),
        "total_R": round(total_r, 3),
    }
    return SignalResearchResult(signals_df, validation)
