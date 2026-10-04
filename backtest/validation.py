from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from analysis.data_quality import assess_ohlcv


@dataclass(frozen=True)
class AcceptanceCriteria:
    """Explicit acceptance contract for the ₹1 lakh research profile."""

    starting_capital: float = 100_000.0
    min_trading_days: int = 100
    min_closed_trades: int = 30
    max_drawdown_pct: float = 12.0
    min_profit_factor: float = 1.10
    min_expectancy: float = 0.0
    max_monte_carlo_ruin_probability: float = 0.05
    stress_cost_multiplier: float = 2.0
    stress_slippage_bps: float = 10.0


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    purge_bars: int


def _normalise_timestamp(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="raise")
    if getattr(ts.dt, "tz", None) is None:
        return ts.dt.tz_localize("Asia/Kolkata")
    return ts.dt.tz_convert("Asia/Kolkata")


def trading_day_count(data: pd.DataFrame) -> int:
    if "timestamp" not in data.columns:
        raise ValueError("timestamp column is required")
    ts = _normalise_timestamp(data["timestamp"])
    return int(ts.dt.date.nunique())


def walk_forward_splits(
    data: pd.DataFrame,
    train_days: int = 60,
    validation_days: int = 20,
    test_days: int = 20,
    step_days: int = 20,
    purge_bars: int = 1,
) -> list[WalkForwardSplit]:
    """Create chronological train/validation/OOS windows.

    No random shuffling is used. The purge count documents the embargo between
    adjacent samples and should be increased when labels/features span bars.
    """
    if min(train_days, validation_days, test_days, step_days) <= 0:
        raise ValueError("walk-forward window sizes must be positive")
    if purge_bars < 0:
        raise ValueError("purge_bars cannot be negative")

    frame = data.copy()
    frame["timestamp"] = _normalise_timestamp(frame["timestamp"])
    days = pd.Series(frame["timestamp"].dt.date).drop_duplicates().sort_values().tolist()
    splits: list[WalkForwardSplit] = []
    cursor = 0
    while cursor + train_days + validation_days + test_days <= len(days):
        train = days[cursor : cursor + train_days]
        validation = days[cursor + train_days : cursor + train_days + validation_days]
        test = days[cursor + train_days + validation_days : cursor + train_days + validation_days + test_days]
        splits.append(
            WalkForwardSplit(
                str(train[0]), str(train[-1]),
                str(validation[0]), str(validation[-1]),
                str(test[0]), str(test[-1]),
                purge_bars,
            )
        )
        cursor += step_days
    return splits


def _max_drawdown_pct(pnl: pd.Series, starting_capital: float) -> float:
    if pnl.empty or starting_capital <= 0:
        return 0.0
    equity = starting_capital + pnl.cumsum()
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan) * 100
    return float(abs(dd.min())) if not dd.empty and pd.notna(dd.min()) else 0.0


def monte_carlo_drawdown(
    trade_pnl: pd.Series,
    starting_capital: float,
    simulations: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap trade outcomes to estimate drawdown and capital-loss risk.

    This is a robustness diagnostic, not a probability forecast of future
    market returns.
    """
    pnl = pd.to_numeric(trade_pnl, errors="coerce").dropna().to_numpy(dtype=float)
    if len(pnl) == 0:
        return {
            "simulations": 0,
            "median_final_equity": float(starting_capital),
            "p05_final_equity": float(starting_capital),
            "p95_final_equity": float(starting_capital),
            "probability_final_below_start": 1.0,
            "probability_max_drawdown_gt_20pct": 1.0,
        }
    rng = np.random.default_rng(seed)
    sample = rng.choice(pnl, size=(simulations, len(pnl)), replace=True)
    equity = starting_capital + np.cumsum(sample, axis=1)
    running_peak = np.maximum.accumulate(equity, axis=1)
    drawdown_pct = (equity - running_peak) / np.maximum(running_peak, 1e-9) * 100
    max_dd = np.abs(np.min(drawdown_pct, axis=1))
    final = equity[:, -1]
    return {
        "simulations": float(simulations),
        "median_final_equity": float(np.median(final)),
        "p05_final_equity": float(np.percentile(final, 5)),
        "p95_final_equity": float(np.percentile(final, 95)),
        "probability_final_below_start": float(np.mean(final < starting_capital)),
        "probability_max_drawdown_gt_20pct": float(np.mean(max_dd > 20.0)),
    }


def stress_trade_pnl(
    trades: pd.DataFrame,
    cost_multiplier: float = 2.0,
    slippage_bps: float = 10.0,
    point_value: float = 1.0,
) -> pd.DataFrame:
    """Apply transparent execution stress to already recorded trades.

    The stress is deliberately additive: doubled recorded costs plus an
    adverse notional-based slippage charge. It never improves a trade.
    """
    if trades.empty:
        return pd.DataFrame(columns=["stressed_pnl", "extra_cost", "stress_slippage"])
    required = {"pnl", "costs", "quantity", "entry_price", "exit_price"}
    if not required.issubset(trades.columns):
        # Capital-aware trades use the same fields under these names.
        missing = sorted(required - set(trades.columns))
        raise ValueError(f"Stress test requires trade columns: {missing}")
    out = trades.copy()
    qty = pd.to_numeric(out["quantity"], errors="coerce").fillna(0.0)
    entry = pd.to_numeric(out["entry_price"], errors="coerce").fillna(0.0)
    exit_price = pd.to_numeric(out["exit_price"], errors="coerce").fillna(0.0)
    costs = pd.to_numeric(out["costs"], errors="coerce").fillna(0.0)
    pnl = pd.to_numeric(out["pnl"], errors="coerce").fillna(0.0)
    extra_cost = costs.clip(lower=0.0) * max(cost_multiplier - 1.0, 0.0)
    notional = ((entry.abs() + exit_price.abs()) / 2.0) * qty * max(point_value, 0.0)
    stress_slippage = notional * max(slippage_bps, 0.0) / 10_000.0
    out["extra_cost"] = extra_cost.round(2)
    out["stress_slippage"] = stress_slippage.round(2)
    out["stressed_pnl"] = (pnl - extra_cost - stress_slippage).round(2)
    return out


def acceptance_report(
    data: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: Any,
    criteria: AcceptanceCriteria | None = None,
    point_value: float = 1.0,
) -> dict[str, Any]:
    """Return a machine-readable PASS/FAIL research acceptance report."""
    c = criteria or AcceptanceCriteria()
    days = trading_day_count(data)
    quality = assess_ohlcv(data)
    total_trades = int(getattr(metrics, "total_trades", len(trades)))
    pf = float(getattr(metrics, "profit_factor", 0.0) or 0.0)
    expectancy = float(getattr(metrics, "expectancy", 0.0) or 0.0)
    dd = float(getattr(metrics, "max_drawdown_pct", 0.0) or 0.0)

    stressed = stress_trade_pnl(
        trades,
        cost_multiplier=c.stress_cost_multiplier,
        slippage_bps=c.stress_slippage_bps,
        point_value=point_value,
    ) if not trades.empty else pd.DataFrame()

    mc = monte_carlo_drawdown(
        stressed["stressed_pnl"] if not stressed.empty else pd.Series(dtype=float),
        c.starting_capital,
    )
    checks = {
        "data_quality": quality.status in {"GREEN", "YELLOW"},
        "minimum_trading_days": days >= c.min_trading_days,
        "minimum_closed_trades": total_trades >= c.min_closed_trades,
        "max_drawdown": dd <= c.max_drawdown_pct,
        "profit_factor": pf >= c.min_profit_factor,
        "positive_expectancy": expectancy > c.min_expectancy,
        "stress_positive": (
            not stressed.empty and float(stressed["stressed_pnl"].sum()) > 0.0
        ),
        "monte_carlo_ruin": (
            float(mc["probability_final_below_start"])
            <= c.max_monte_carlo_ruin_probability
        ),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return {
        "status": "PASS" if not failures else "FAIL",
        "starting_capital": c.starting_capital,
        "trading_days": days,
        "closed_trades": total_trades,
        "metrics": {
            "profit_factor": pf,
            "expectancy": expectancy,
            "max_drawdown_pct": dd,
        },
        "quality": asdict(quality),
        "walk_forward_splits": [
            asdict(s) for s in walk_forward_splits(data)
        ],
        "stress": {
            "cost_multiplier": c.stress_cost_multiplier,
            "slippage_bps": c.stress_slippage_bps,
            "stressed_net_pnl": (
                float(stressed["stressed_pnl"].sum()) if not stressed.empty else 0.0
            ),
        },
        "monte_carlo": mc,
        "checks": checks,
        "failures": failures,
    }
