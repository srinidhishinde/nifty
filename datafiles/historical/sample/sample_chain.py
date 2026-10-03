from datetime import datetime, timedelta

from features.historical_schema import (
    ChainSnapshot,
    OptionSnapshot,
    UnderlyingSnapshot,
)


def create_sample_chain(
    rows: int = 120,
) -> list[ChainSnapshot]:

    result: list[ChainSnapshot] = []

    start = datetime(
        2026,
        1,
        5,
        9,
        15,
    )

    base_price = 25000.0

    for i in range(rows):

        timestamp = start + timedelta(
            minutes=5 * i
        )

        # Deterministic movement for testing.
        movement = (
            ((i % 20) - 10) * 2.5
        )

        price = (
            base_price
            + movement
            + i * 1.2
        )

        underlying = UnderlyingSnapshot(
            timestamp=timestamp,
            instrument="NIFTY",
            price=price,
            open=price - 5,
            high=price + 15,
            low=price - 15,
            close=price,
            volume=100000 + i * 100,
            vwap=price - 2,
        )

        options = []

        for offset in (-100, 0, 100):

            strike = round(
                price / 100
            ) * 100 + offset

            for option_type in ("CE", "PE"):

                if option_type == "CE":

                    intrinsic = max(
                        price - strike,
                        0,
                    )

                else:

                    intrinsic = max(
                        strike - price,
                        0,
                    )

                premium = (
                    80
                    + intrinsic * 0.35
                    + (i % 10)
                )

                options.append(
                    OptionSnapshot(
                        timestamp=timestamp,
                        instrument="NIFTY",
                        expiry="2026-01-08",
                        strike=strike,
                        option_type=option_type,
                        underlying_price=price,
                        option_ltp=round(
                            premium,
                            2,
                        ),
                        bid=round(
                            premium - 1,
                            2,
                        ),
                        ask=round(
                            premium + 1,
                            2,
                        ),
                        volume=10000 + i * 10,
                        open_interest=50000 + i * 100,
                        oi_change=100 + (i % 5),
                        implied_volatility=18 + (i % 5),
                    )
                )

        result.append(
            ChainSnapshot(
                timestamp=timestamp,
                underlying=underlying,
                options=tuple(options),
            )
        )

    return result
