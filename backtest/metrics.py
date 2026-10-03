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


def calculate_max_drawdown(equity: pd.Series) -> tuple[float, float]:

    if equity.empty:
        return 0.0, 0.0

    running_max = equity.cummax()

    drawdown = equity - running_max

    max_drawdown = abs(drawdown.min())

    max_equity = running_max.max()

    if max_equity == 0:
        return max_drawdown, 0.0

    max_drawdown_pct = (
        max_drawdown / max_equity
    ) * 100

    return max_drawdown, max_drawdown_pct


def calculate_metrics(
    trades: pd.DataFrame,
    starting_capital: float,
) -> BacktestMetrics:

    if trades.empty:

        return BacktestMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            net_pnl=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            average_trade=0.0,
            profit_factor=0.0,
            return_pct=0.0,
        )

    pnl = trades["pnl"].astype(float)

    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]

    total_trades = len(pnl)

    winning_trades = len(winners)

    losing_trades = len(losers)

    win_rate = (
        winning_trades / total_trades * 100
    )

    gross_profit = winners.sum()

    gross_loss = abs(losers.sum())

    net_pnl = pnl.sum()

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else math.inf
    )

    equity = (
        starting_capital
        + pnl.cumsum()
    )

    max_drawdown, max_drawdown_pct = (
        calculate_max_drawdown(equity)
    )

    average_trade = pnl.mean()

    return_pct = (
        net_pnl / starting_capital * 100
    )

    return BacktestMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate_pct=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_pnl=net_pnl,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        average_trade=average_trade,
        profit_factor=profit_factor,
        return_pct=return_pct,
    )