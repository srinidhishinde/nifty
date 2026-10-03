from dataclasses import dataclass
from typing import Literal

from features.historical_schema import (
    ChainSnapshot,
)


Side = Literal["CE", "PE"]


@dataclass(frozen=True)
class ContractTrade:

    instrument: str

    option_type: Side

    strike: float

    entry_time: object

    exit_time: object

    entry_price: float

    exit_price: float

    quantity: int

    gross_pnl: float

    transaction_cost: float

    net_pnl: float


@dataclass(frozen=True)
class ReplayResult:

    trades: tuple[ContractTrade, ...]

    starting_capital: float

    ending_capital: float

    total_pnl: float

    win_rate: float

    max_drawdown: float


class ContractReplayEngine:

    def __init__(
        self,
        starting_capital: float,
        max_loss_per_trade: float,
        transaction_cost_rate: float = 0.0005,
    ):

        self.starting_capital = (
            starting_capital
        )

        self.max_loss_per_trade = (
            max_loss_per_trade
        )

        self.transaction_cost_rate = (
            transaction_cost_rate
        )

    def run(
        self,
        snapshots: list[ChainSnapshot],
        hold_bars: int = 3,
    ) -> ReplayResult:

        if len(snapshots) <= hold_bars:

            return ReplayResult(
                trades=(),
                starting_capital=self.starting_capital,
                ending_capital=self.starting_capital,
                total_pnl=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
            )

        trades: list[ContractTrade] = []

        equity = self.starting_capital

        peak_equity = equity

        max_drawdown = 0.0

        # --------------------------------------------------
        # Replay only information available at entry time.
        # --------------------------------------------------

        for index in range(
            0,
            len(snapshots) - hold_bars,
            hold_bars,
        ):

            entry_snapshot = snapshots[index]

            exit_snapshot = snapshots[
                index + hold_bars
            ]

            options = (
                entry_snapshot.options
            )

            if not options:

                continue

            # Deterministic paper-test contract:
            # ATM CE for even bars, ATM PE for odd bars.
            #
            # The real signal engine will replace this
            # selection later.
            option_type: Side = (
                "CE"
                if index % 2 == 0
                else "PE"
            )

            candidates = [
                option
                for option in options
                if option.option_type
                == option_type
            ]

            if not candidates:

                continue

            selected = min(
                candidates,
                key=lambda option: abs(
                    option.strike
                    - entry_snapshot.underlying.price
                ),
            )

            matching_exit = [
                option
                for option in exit_snapshot.options
                if (
                    option.option_type
                    == selected.option_type
                    and option.strike
                    == selected.strike
                )
            ]

            if not matching_exit:

                continue

            exit_contract = (
                matching_exit[0]
            )

            entry_price = (
                selected.ask
            )

            exit_price = (
                exit_contract.bid
            )

            if entry_price <= 0:

                continue

            # Fixed risk sizing.
            risk_budget = min(
                self.max_loss_per_trade,
                equity * 0.02,
            )

            quantity = max(
                1,
                int(
                    risk_budget
                    / entry_price
                ),
            )

            gross_pnl = (
                exit_price
                - entry_price
            ) * quantity

            turnover = (
                entry_price
                + exit_price
            ) * quantity

            transaction_cost = (
                turnover
                * self.transaction_cost_rate
            )

            net_pnl = (
                gross_pnl
                - transaction_cost
            )

            # Safety cap for a paper-trade replay.
            net_pnl = max(
                net_pnl,
                -self.max_loss_per_trade,
            )

            equity += net_pnl

            peak_equity = max(
                peak_equity,
                equity,
            )

            drawdown = (
                peak_equity
                - equity
            )

            max_drawdown = max(
                max_drawdown,
                drawdown,
            )

            trades.append(
                ContractTrade(
                    instrument=selected.instrument,
                    option_type=selected.option_type,
                    strike=selected.strike,
                    entry_time=entry_snapshot.timestamp,
                    exit_time=exit_snapshot.timestamp,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    gross_pnl=round(
                        gross_pnl,
                        2,
                    ),
                    transaction_cost=round(
                        transaction_cost,
                        2,
                    ),
                    net_pnl=round(
                        net_pnl,
                        2,
                    ),
                )
            )

        total_pnl = (
            equity
            - self.starting_capital
        )

        wins = sum(
            trade.net_pnl > 0
            for trade in trades
        )

        win_rate = (
            (wins / len(trades)) * 100
            if trades
            else 0.0
        )

        return ReplayResult(
            trades=tuple(trades),
            starting_capital=self.starting_capital,
            ending_capital=round(
                equity,
                2,
            ),
            total_pnl=round(
                total_pnl,
                2,
            ),
            win_rate=round(
                win_rate,
                2,
            ),
            max_drawdown=round(
                max_drawdown,
                2,
            ),
        )
