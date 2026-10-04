from dataclasses import dataclass
import math
import pandas as pd


@dataclass
class BacktestMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    gross_profit: float
    gross_loss: float
    net_pnl: float
    max_drawdown: float
    max_drawdown_pct: float
    average_trade: float
    profit_factor: float
    return_pct: float
    expectancy: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0


def calculate_max_drawdown(equity: pd.Series) -> tuple[float, float]:
    if equity.empty:
        return 0.0, 0.0
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = abs(float(drawdown.min()))
    max_equity = float(running_max.max())
    return max_drawdown, (max_drawdown / max_equity * 100) if max_equity else 0.0


def calculate_metrics(trades: pd.DataFrame, starting_capital: float) -> BacktestMetrics:
    if trades.empty:
        return BacktestMetrics(0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
    winners, losers = pnl[pnl > 0], pnl[pnl < 0]
    total = len(pnl)
    gross_profit, gross_loss = float(winners.sum()), float(abs(losers.sum()))
    equity = starting_capital + pnl.cumsum()
    mdd, mdd_pct = calculate_max_drawdown(equity)
    trade_returns = pnl / float(starting_capital) if starting_capital else pnl * 0.0
    sharpe = float(trade_returns.mean() / trade_returns.std(ddof=1) * math.sqrt(total)) if total > 1 and trade_returns.std(ddof=1) > 0 else 0.0
    downside = trade_returns[trade_returns < 0]
    sortino = float(trade_returns.mean() / downside.std(ddof=1) * math.sqrt(total)) if len(downside) > 1 and downside.std(ddof=1) > 0 else 0.0
    calmar = float((pnl.sum() / starting_capital) / (mdd / starting_capital)) if starting_capital and mdd > 0 else 0.0
    return BacktestMetrics(
        total_trades=total,
        winning_trades=len(winners),
        losing_trades=len(losers),
        win_rate_pct=round(len(winners)/total*100,2),
        gross_profit=round(gross_profit,2),
        gross_loss=round(gross_loss,2),
        net_pnl=round(float(pnl.sum()),2),
        max_drawdown=round(mdd,2),
        max_drawdown_pct=round(mdd_pct,2),
        average_trade=round(float(pnl.mean()),2),
        profit_factor=round(gross_profit/gross_loss,3) if gross_loss else math.inf,
        return_pct=round(float(pnl.sum())/starting_capital*100,3) if starting_capital else 0.0,
        expectancy=round(float(pnl.mean()),2),
        average_win=round(float(winners.mean()),2) if not winners.empty else 0.0,
        average_loss=round(float(losers.mean()),2) if not losers.empty else 0.0,
        sharpe=round(sharpe,3),
        sortino=round(sortino,3),
        calmar=round(calmar,3),
    )
