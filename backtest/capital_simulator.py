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


@dataclass(frozen=True)
class RollingSimulationResult:
    windows: pd.DataFrame
    window_days: int
    total_windows: int
    profitable_windows: int
    loss_windows: int
    profitable_window_pct: float
    median_ending_capital: float
    mean_ending_capital: float
    worst_ending_capital: float
    best_ending_capital: float
    median_return_pct: float
    worst_return_pct: float
    best_return_pct: float
    median_drawdown_pct: float
    worst_drawdown_pct: float


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
    if starting_capital <= 0 or days <= 0 or risk_pct_per_trade <= 0:
        raise ValueError("starting_capital, days and risk_pct_per_trade must be positive")
    if daily_loss_pct < 0:
        raise ValueError("daily_loss_pct cannot be negative")

    frame = data.copy()
    if "timestamp" not in frame:
        raise ValueError("timestamp is required")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    unique_days = sorted(frame["timestamp"].dt.date.unique())
    if len(unique_days) < days:
        raise ValueError(f"Need at least {days} trading days; received {len(unique_days)}")

    selected_dates = unique_days[-days:]
    evaluation_start = pd.Timestamp(selected_dates[0])
    evaluation_end = pd.Timestamp(selected_dates[-1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    risk_cash = starting_capital * risk_pct_per_trade / 100.0

    raw = RuleBacktestEngine(
        starting_capital=starting_capital,
        risk_per_trade=risk_cash,
        instrument=instrument,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        daily_loss_limit_pct=daily_loss_pct / 100.0,
    ).run(frame, evaluation_start=evaluation_start, evaluation_end=evaluation_end)

    trades = raw.trades.copy()
    if trades.empty:
        curve = pd.DataFrame({"date": selected_dates, "daily_pnl": 0.0, "equity": starting_capital})
        return CapitalSimulationResult(
            starting_capital, starting_capital, 0.0, 0.0, 0.0, 0.0,
            len(selected_dates), 0, 0, 0, 0.0, 0.0, 0.0, 0.0, curve, trades
        )

    trades["exit_time"] = pd.to_datetime(trades["exit_time"], errors="coerce")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], errors="coerce")
    equity = float(starting_capital)
    daily: dict = {}
    simulated_rows = []

    for _, trade in trades.sort_values("exit_time").iterrows():
        day = trade["exit_time"].date()
        prior_day_pnl = daily.get(day, 0.0)
        limit = equity * daily_loss_pct / 100.0
        if daily_loss_pct and prior_day_pnl <= -limit:
            continue
        scale = equity / starting_capital if compounding else 1.0
        pnl = float(trade["pnl"]) * scale
        if daily_loss_pct and prior_day_pnl + pnl < -limit:
            pnl = -limit - prior_day_pnl
        equity += pnl
        daily[day] = prior_day_pnl + pnl
        simulated_rows.append({**trade.to_dict(), "simulated_pnl": pnl, "simulated_equity": equity})

    sim = pd.DataFrame(simulated_rows)
    curve = pd.DataFrame({
        "date": selected_dates,
        "daily_pnl": [float(daily.get(day, 0.0)) for day in selected_dates],
    })
    curve["equity"] = starting_capital + curve["daily_pnl"].cumsum()
    peak = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] - peak
    max_dd = abs(float(curve["drawdown"].min())) if not curve.empty else 0.0
    max_dd_pct = max_dd / float(curve["equity"].max()) * 100 if not curve.empty else 0.0

    wins = sim[sim["simulated_pnl"] > 0]
    losses = sim[sim["simulated_pnl"] < 0]
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


def rolling_capital_simulation(
    data: pd.DataFrame,
    *,
    starting_capital: float = 100_000.0,
    window_days: int = 100,
    step_days: int = 20,
    risk_pct_per_trade: float = 1.0,
    daily_loss_pct: float = 2.0,
    compounding: bool = True,
    commission_pct: float = 0.0005,
    slippage_pct: float = 0.0005,
    instrument: str = "NIFTY",
) -> RollingSimulationResult:
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    dates = sorted(frame["timestamp"].dt.date.unique())
    if len(dates) < window_days:
        raise ValueError(f"Need at least {window_days} trading days; received {len(dates)}")

    rows = []
    for end_idx in range(window_days, len(dates) + 1, step_days):
        window_dates = dates[end_idx - window_days:end_idx]
        window = frame[frame["timestamp"].dt.date.isin(window_dates)]
        result = simulate_capital(
            window,
            starting_capital=starting_capital,
            days=window_days,
            risk_pct_per_trade=risk_pct_per_trade,
            daily_loss_pct=daily_loss_pct,
            compounding=compounding,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            instrument=instrument,
        )
        rows.append({
            "start_date": window_dates[0],
            "end_date": window_dates[-1],
            "starting_capital": starting_capital,
            "ending_capital": result.ending_capital,
            "net_profit": result.net_profit,
            "return_pct": result.return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "trades": result.trades,
            "win_rate_pct": result.win_rate_pct,
            "profit_factor": result.profit_factor,
        })

    # Always include the final window when step size does not land exactly on it.
    if dates[-window_days:] and (not rows or rows[-1]["end_date"] != dates[-1]):
        window_dates = dates[-window_days:]
        window = frame[frame["timestamp"].dt.date.isin(window_dates)]
        result = simulate_capital(
            window,
            starting_capital=starting_capital,
            days=window_days,
            risk_pct_per_trade=risk_pct_per_trade,
            daily_loss_pct=daily_loss_pct,
            compounding=compounding,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            instrument=instrument,
        )
        rows.append({
            "start_date": window_dates[0],
            "end_date": window_dates[-1],
            "starting_capital": starting_capital,
            "ending_capital": result.ending_capital,
            "net_profit": result.net_profit,
            "return_pct": result.return_pct,
            "max_drawdown_pct": result.max_drawdown_pct,
            "trades": result.trades,
            "win_rate_pct": result.win_rate_pct,
            "profit_factor": result.profit_factor,
        })

    windows = pd.DataFrame(rows)
    returns = windows["return_pct"]
    return RollingSimulationResult(
        windows=windows,
        window_days=window_days,
        total_windows=len(windows),
        profitable_windows=int((returns > 0).sum()),
        loss_windows=int((returns < 0).sum()),
        profitable_window_pct=round(float((returns > 0).mean() * 100), 2),
        median_ending_capital=round(float(windows["ending_capital"].median()), 2),
        mean_ending_capital=round(float(windows["ending_capital"].mean()), 2),
        worst_ending_capital=round(float(windows["ending_capital"].min()), 2),
        best_ending_capital=round(float(windows["ending_capital"].max()), 2),
        median_return_pct=round(float(returns.median()), 2),
        worst_return_pct=round(float(returns.min()), 2),
        best_return_pct=round(float(returns.max()), 2),
        median_drawdown_pct=round(float(windows["max_drawdown_pct"].median()), 2),
        worst_drawdown_pct=round(float(windows["max_drawdown_pct"].max()), 2),
    )
