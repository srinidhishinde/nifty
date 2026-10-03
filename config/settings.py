from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=False,
    extra="ignore",
)

# --------------------------------------------------
# Application
# --------------------------------------------------

app_name: str = "AI Derivatives Terminal"

environment: str = "uat"

# --------------------------------------------------
# Trading
# --------------------------------------------------

starting_capital: float = 300000.0

max_loss_per_trade: float = 15000.0

max_trades_per_day: int = 15

max_daily_loss: float = 30000.0

paper_trading: bool = True

live_trading_enabled: bool = False

# --------------------------------------------------
# Strategy
# --------------------------------------------------

minimum_signal_confidence: float = 60.0

minimum_signal_edge: float = 10.0

# --------------------------------------------------
# Data
# --------------------------------------------------

default_timeframe: str = "5m"

# --------------------------------------------------
# Kotak Neo
#
# These are mapped directly from your .env names.
# --------------------------------------------------

kotak_api_key: str = Field(
    default="",
    validation_alias="NEO_CONSUMER_KEY",
)

kotak_mobile_number: str = Field(
    default="",
    validation_alias="NEO_MOBILE",
)

kotak_ucc: str = Field(
    default="",
    validation_alias="NEO_UCC",
)

kotak_mpin: str = Field(
    default="",
    validation_alias="NEO_MPIN",
)

# Optional values if you later use them.
kotak_api_secret: str = ""

kotak_access_token: str = ""

kotak_totp_secret: str = ""

# --------------------------------------------------
# Execution safety
# --------------------------------------------------

allow_order_submission: bool = False

def live_trading_allowed(self) -> bool:

    return (
        self.environment.lower() == "production"
        and self.live_trading_enabled
        and not self.paper_trading
        and self.allow_order_submission
    )


settings = Settings()