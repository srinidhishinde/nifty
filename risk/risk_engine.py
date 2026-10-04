from config.settings import settings


def _risk_budget(account_equity: float) -> float:
    if account_equity <= 0:
        return 0.0
    return float(account_equity) * float(settings.risk_fraction)


def _daily_loss_limit(account_equity: float) -> float:
    if account_equity <= 0:
        return 0.0
    return float(account_equity) * float(settings.max_daily_loss_fraction)


def calculate_position_size(entry: float, stop_loss: float, lot_size: int, account_equity: float) -> int:
    if lot_size <= 0 or account_equity <= 0:
        return 0
    risk_per_unit = abs(entry - stop_loss)
    if risk_per_unit <= 0:
        return 0
    quantity = int(_risk_budget(account_equity) / risk_per_unit)
    return max(quantity // lot_size, 0) * lot_size


def validate_risk(
    estimated_loss: float,
    daily_loss: float,
    trades_today: int,
    account_equity: float,
) -> bool:
    if account_equity <= 0:
        return False
    if estimated_loss > _risk_budget(account_equity):
        return False
    if daily_loss >= _daily_loss_limit(account_equity):
        return False
    if trades_today >= settings.max_trades_per_day:
        return False
    return True
