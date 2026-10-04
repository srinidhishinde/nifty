from dataclasses import dataclass


@dataclass(frozen=True)
class TradeLevels:
    entry: float
    stop_loss: float
    target: float
    risk_per_unit: float
    reward_per_unit: float
    risk_reward: float

    @property
    def reward_risk_ratio(self) -> float:
        """
        Compatibility alias used by the strategy tests.

        reward_risk_ratio and risk_reward represent the same
        reward-to-risk calculation.
        """
        return self.risk_reward


class TradeLevelCalculator:
    """
    Calculates entry, stop-loss and target levels for an option trade.

    Default configuration is intentionally conservative and must be
    overridden by the unified volatility/contract risk engine for live or
    evidence-grade backtests. This class remains a compatibility calculator.

    Therefore:
        Entry       = 100
        Stop loss   = 80
        Target      = 140
        Risk        = 20
        Reward      = 40
        Reward/Risk = 2.0
    """

    def __init__(
        self,
        stop_loss_pct: float = 10.0,
        target_pct: float = 20.0,
        minimum_risk_reward: float = 1.8,
    ) -> None:

        if stop_loss_pct <= 0:
            raise ValueError(
                "stop_loss_pct must be greater than zero"
            )

        if target_pct <= 0:
            raise ValueError(
                "target_pct must be greater than zero"
            )

        if minimum_risk_reward <= 0:
            raise ValueError(
                "minimum_risk_reward must be greater than zero"
            )

        self.stop_loss_pct = float(stop_loss_pct)
        self.target_pct = float(target_pct)
        self.minimum_risk_reward = float(
            minimum_risk_reward
        )

    def calculate(
        self,
        direction: str,
        entry: float,
        strike: float | None = None,
    ) -> TradeLevels:

        # Strike is accepted because the strategy API supplies it.
        # Premium-based SL/target calculation does not require it.
        del strike

        entry = float(entry)

        if entry <= 0:
            raise ValueError(
                "entry must be greater than zero"
            )

        direction = direction.upper().strip()

        if direction not in {
            "CE",
            "PE",
            "BUY",
            "SELL",
        }:
            raise ValueError(
                f"Unsupported direction: {direction}"
            )

        stop_fraction = (
            self.stop_loss_pct / 100.0
        )

        target_fraction = (
            self.target_pct / 100.0
        )

        stop_loss = entry * (
            1.0 - stop_fraction
        )

        target = entry * (
            1.0 + target_fraction
        )

        risk_per_unit = (
            entry - stop_loss
        )

        reward_per_unit = (
            target - entry
        )

        if risk_per_unit <= 0:
            raise ValueError(
                "Calculated risk must be positive"
            )

        risk_reward = (
            reward_per_unit
            / risk_per_unit
        )

        if (
            risk_reward
            < self.minimum_risk_reward
        ):
            raise ValueError(
                "Calculated risk/reward is below "
                "the configured minimum"
            )

        return TradeLevels(
            entry=round(entry, 2),
            stop_loss=round(stop_loss, 2),
            target=round(target, 2),
            risk_per_unit=round(
                risk_per_unit,
                2,
            ),
            reward_per_unit=round(
                reward_per_unit,
                2,
            ),
            risk_reward=round(
                risk_reward,
                2,
            ),
        )
