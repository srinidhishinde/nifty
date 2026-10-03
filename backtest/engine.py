from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backtest.costs import (
    TradingCosts,
    apply_buy_cost,
    apply_sell_cost,
    calculate_transaction_cost,
)


@dataclass
class Trade:

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp

    symbol: str
    direction: str

    entry_price: float
    exit_price: float

    quantity: int

    pnl: float

    reason: str


class BacktestEngine:

    def __init__(
        self,
        starting_capital: float = 300000,
        max_risk_per_trade: float = 15000,
        costs: Optional[TradingCosts] = None,
    ):

        self.starting_capital = starting_capital

        self.max_risk_per_trade = (
            max_risk_per_trade
        )

        self.costs = costs or TradingCosts()

    def run(
        self,
        data: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:

        required = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "signal",
        }

        missing = required - set(data.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        data = data.copy()

        data["timestamp"] = pd.to_datetime(
            data["timestamp"]
        )

        data = data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        trades: list[Trade] = []

        position = None

        for _, row in data.iterrows():

            signal = str(row["signal"]).upper()

            price = float(row["close"])

            timestamp = row["timestamp"]

            # ------------------------------------
            # Entry
            # ------------------------------------

            if position is None:

                if signal not in {"CE", "PE"}:
                    continue

                entry_price = apply_buy_cost(
                    price,
                    self.costs,
                )

                position = {
                    "entry_time": timestamp,
                    "entry_price": entry_price,
                    "direction": signal,
                    "quantity": 1,
                }

                continue

            # ------------------------------------
            # Exit / reversal
            # ------------------------------------

            if signal == position["direction"]:
                continue

            exit_price = apply_sell_cost(
                price,
                self.costs,
            )

            entry_price = position["entry_price"]

            quantity = position["quantity"]

            if position["direction"] == "CE":

                pnl = (
                    exit_price - entry_price
                ) * quantity

            else:

                pnl = (
                    entry_price - exit_price
                ) * quantity

            transaction_cost = (
                calculate_transaction_cost(
                    entry_price,
                    exit_price,
                    quantity,
                    self.costs,
                )
            )

            pnl -= transaction_cost

            trades.append(
                Trade(
                    entry_time=position["entry_time"],
                    exit_time=timestamp,
                    symbol=symbol,
                    direction=position["direction"],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    pnl=pnl,
                    reason="signal_reversal",
                )
            )

            position = None

            # Don't immediately re-enter on the
            # same candle.
            continue

        # ----------------------------------------
        # Close remaining position
        # ----------------------------------------

        if position is not None and not data.empty:

            row = data.iloc[-1]

            exit_price = apply_sell_cost(
                float(row["close"]),
                self.costs,
            )

            entry_price = position["entry_price"]

            quantity = position["quantity"]

            if position["direction"] == "CE":

                pnl = (
                    exit_price - entry_price
                ) * quantity

            else:

                pnl = (
                    entry_price - exit_price
                ) * quantity

            transaction_cost = (
                calculate_transaction_cost(
                    entry_price,
                    exit_price,
                    quantity,
                    self.costs,
                )
            )

            pnl -= transaction_cost

            trades.append(
                Trade(
                    entry_time=position["entry_time"],
                    exit_time=row["timestamp"],
                    symbol=symbol,
                    direction=position["direction"],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    pnl=pnl,
                    reason="end_of_data",
                )
            )

        if not trades:

            return pd.DataFrame(
                columns=[
                    "entry_time",
                    "exit_time",
                    "symbol",
                    "direction",
                    "entry_price",
                    "exit_price",
                    "quantity",
                    "pnl",
                    "reason",
                ]
            )

        return pd.DataFrame(
            [trade.__dict__ for trade in trades]
        )