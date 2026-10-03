from dataclasses import dataclass


@dataclass(frozen=True)
class StrikeLevel:
    strike: float
    distance: int


class StrikeSelector:
    """
    Selects ATM and nearby option strikes.

    Compatibility:
    - build_levels(strikes, spot)
    - build_levels(strikes, atm_index, spot)

    The existing tests use the two-argument form:
        StrikeSelector.build_levels(strikes, spot)
    """

    @staticmethod
    def build_levels(
        strikes: list[float],
        spot_or_atm_index: float | int,
        spot: float | None = None,
    ) -> list[tuple[float, int]]:
        """
        Return strikes ordered by distance from ATM.

        Example:
            strikes = [24900, 25000, 25100, 25200]
            spot = 25040

        Result:
            [(25000.0, 0), (25100.0, 1), (24900.0, 1), ...]
        """

        if not strikes:
            return []

        # Two-argument form:
        # build_levels(strikes, spot)
        if spot is None:
            spot_value = float(spot_or_atm_index)

        # Three-argument compatibility form:
        # build_levels(strikes, atm_index, spot)
        else:
            spot_value = float(spot)

        normalized = sorted(
            {float(strike) for strike in strikes}
        )

        if not normalized:
            return []

        # Find the strike closest to spot.
        atm_strike = min(
            normalized,
            key=lambda strike: (
                abs(strike - spot_value),
                strike,
            ),
        )

        # Distance is expressed in strike steps from ATM.
        levels: list[tuple[float, int]] = []

        for strike in normalized:
            distance = abs(
                normalized.index(strike)
                - normalized.index(atm_strike)
            )

            levels.append(
                (float(strike), distance)
            )

        # ATM first, then nearest strikes.
        levels.sort(
            key=lambda item: (
                item[1],
                abs(item[0] - spot_value),
                item[0],
            )
        )

        return levels

    @staticmethod
    def select(
        strikes: list[float],
        spot: float,
        atm_count: int = 2,
        otm_count: int = 5,
    ) -> dict[str, list[float]]:
        """
        Return ATM and OTM strike groups.

        This is intended for the later option-selection layer.

        atm_count:
            Number of strikes closest to spot.

        otm_count:
            Number of additional strikes on each relevant side.
        """

        normalized = sorted(
            {float(strike) for strike in strikes}
        )

        if not normalized:
            return {
                "atm": [],
                "otm": [],
            }

        levels = StrikeSelector.build_levels(
            normalized,
            spot,
        )

        atm = [
            strike
            for strike, distance in levels
            if distance == 0
        ][:max(1, atm_count)]

        if not atm:
            atm = [
                min(
                    normalized,
                    key=lambda x: abs(x - spot)
                )
            ]

        atm_strike = atm[0]

        # OTM means strikes away from ATM.
        otm = [
            strike
            for strike in normalized
            if strike != atm_strike
        ]

        otm.sort(
            key=lambda strike: (
                abs(strike - atm_strike),
                strike,
            )
        )

        return {
            "atm": atm,
            "otm": otm[:max(0, otm_count)],
        }
