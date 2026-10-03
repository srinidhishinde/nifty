from config.settings import settings


def calculate_position_size(
    entry: float,
    stop_loss: float,
    lot_size: int
) -> int:

    risk_per_unit = abs(entry - stop_loss)

    if risk_per_unit <= 0:
        return 0

    quantity = int(
        settings.max_loss_per_trade
        / risk_per_unit
    )

    lots = quantity // lot_size

    return max(lots, 0)


def validate_risk(
    estimated_loss: float,
    daily_loss: float,
    trades_today: int
) -> bool:

    if estimated_loss > settings.max_loss_per_trade:
        return False

    if daily_loss >= settings.max_daily_loss:
        return False

    if trades_today >= settings.max_trades_per_day:
        return False

    return True
