from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from backtest.rule_engine import RuleBacktestEngine


@dataclass(frozen=True)
class CapitalSimulationResult:
    starting_capital: float
    ending_capital: float
    net_profit: float
    return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    trading_days: int
    trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    worst_day: float
    best_day: float
    equity_curve: pd.DataFrame
    trades_frame: pd.DataFrame


def simulate_capital(
    data: pd.DataFrame,
    *,
    starting_capital: float = 100_000.0,
    days: int = 100,
    risk_pct_per_trade: float = 1.0,
    daily_loss_pct: float = 2.0,
    compounding: bool = True,
    commission_pct: float = 0.0005,
    slippage_pct: float = 0.0005,
    instrument: str = "NIFTY",
) -> CapitalSimulationResult:
    """Project a fixed starting capital through the strategy's historical trades.

    This is a scenario simulator, not a forecast. It uses only the supplied
    historical OHLCV data and scales each historical trade's R outcome to the
    simulated account. It never invents future market prices.
    """
    if starting_capital <= 0 or days <= 0 or risk_pct_per_trade <= 0:
        raise ValueError("starting_capital, days and risk_pct_per_trade must be positive")
    if daily_loss_pct < 0:
        raise ValueError("daily_loss_pct cannot be negative")

    frame = data.copy()
    if "timestamp" not in frame:
        raise ValueError("timestamp is required")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    available_days = frame["timestamp"].dt.date.nunique()
    if available_days < days:
        raise ValueError(f"Need at least {days} trading days; received {available_days}")

    selected_dates = sorted(frame["timestamp"].dt.date.unique())[-days:]
    evaluation_start = pd.Timestamp(selected_dates[0])
    evaluation_end = pd.Timestamp(selected_dates[-1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    risk_cash = starting_capital * risk_pct_per_trade / 100.0

    # Keep the full history in the engine so EMA/RSI/MACD/ATR/VWAP have
    # warm-up data. Only the final N trading days are scored.
    raw = RuleBacktestEngine(
        starting_capital=starting_capital,
        risk_per_trade=risk_cash,
        instrument=instrument,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        daily_loss_limit_pct=daily_loss_pct / 100.0,
    ).run(
        frame,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
    )

    trades = raw.trades.copy()
    if trades.empty:
        curve = pd.DataFrame({"date": selected_dates, "equity": [starting_capital] * len(selected_dates)})
        return CapitalSimulationResult(
            starting_capital, starting_capital, 0.0, 0.0, 0.0, 0.0,
            len(selected_dates), 0, 0, 0, 0.0, 0.0, 0.0, 0.0, curve, trades
        )

    trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], errors="coerce")
    base_risk = max(risk_cash, 1e-9)
    equity = float(starting_capital)
    daily = {}
    simulated_rows = []
    for _, trade in trades.sort_values("exit_time").iterrows():
        day = trade["exit_time"].date()
        prior_day_pnl = daily.get(day, 0.0)
        limit = equity * daily_loss_pct / 100.0
        if daily_loss_pct and prior_day_pnl <= -limit:
            continue
        scale = (equity / starting_capital) if compounding else 1.0
        pnl = float(trade["pnl"]) * scale
        if daily_loss_pct and prior_day_pnl + pnl < -limit:
            pnl = -limit - prior_day_pnl
        equity += pnl
        daily[day] = prior_day_pnl + pnl
        simulated_rows.append({**trade.to_dict(), "simulated_pnl": pnl, "simulated_equity": equity})

    sim = pd.DataFrame(simulated_rows)
    day_rows = []
    for day in selected_dates:
        pnl = float(daily.get(day, 0.0))
        day_rows.append({"date": day, "daily_pnl": pnl})
    curve = pd.DataFrame(day_rows)
    curve["equity"] = starting_capital + curve["daily_pnl"].cumsum()
    peak = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] - peak
    max_dd = abs(float(curve["drawdown"].min())) if not curve.empty else 0.0
    max_dd_pct = max_dd / float(curve["equity"].max()) * 100 if not curve.empty else 0.0

    wins = sim[sim["simulated_pnl"] > 0] if not sim.empty else sim
    losses = sim[sim["simulated_pnl"] < 0] if not sim.empty else sim
    gross_profit = float(wins["simulated_pnl"].sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses["simulated_pnl"].sum())) if not losses.empty else 0.0
    return CapitalSimulationResult(
        starting_capital=starting_capital,
        ending_capital=round(equity, 2),
        net_profit=round(equity - starting_capital, 2),
        return_pct=round((equity / starting_capital - 1) * 100, 2),
        max_drawdown=round(max_dd, 2),
        max_drawdown_pct=round(max_dd_pct, 2),
        trading_days=len(selected_dates),
        trades=len(sim),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate_pct=round(len(wins) / len(sim) * 100, 2) if len(sim) else 0.0,
        profit_factor=round(gross_profit / gross_loss, 3) if gross_loss else (float("inf") if gross_profit else 0.0),
        worst_day=round(min(daily.values()) if daily else 0.0, 2),
        best_day=round(max(daily.values()) if daily else 0.0, 2),
        equity_curve=curve,
        trades_frame=sim,
    )
