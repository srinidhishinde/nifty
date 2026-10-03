from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    app_name: str = "AI Derivatives Terminal"
    environment: str = Field(default="paper", validation_alias="APP_ENV")

    starting_capital: float = Field(default=300000.0, validation_alias="CAPITAL")
    max_loss_per_trade: float = Field(default=15000.0, validation_alias="MAX_RISK_PER_TRADE")
    max_trades_per_day: int = Field(default=15, validation_alias="MAX_TRADES_PER_DAY")
    max_daily_loss: float = Field(default=30000.0, validation_alias="MAX_DAILY_LOSS")
    paper_trading: bool = Field(default=True, validation_alias="PAPER_TRADING")
    live_trading_enabled: bool = Field(default=False, validation_alias="LIVE_TRADING_ENABLED")

    minimum_signal_confidence: float = 60.0
    minimum_signal_edge: float = 10.0
    default_timeframe: str = "5m"

    neo_consumer_key: str = Field(default="", validation_alias="NEO_CONSUMER_KEY")
    neo_mobile: str = Field(default="", validation_alias="NEO_MOBILE")
    neo_ucc: str = Field(default="", validation_alias="NEO_UCC")
    neo_mpin: str = Field(default="", validation_alias="NEO_MPIN")

    kotak_api_key: str = Field(default="", validation_alias="NEO_CONSUMER_KEY")
    kotak_mobile_number: str = Field(default="", validation_alias="NEO_MOBILE")
    kotak_ucc: str = Field(default="", validation_alias="NEO_UCC")
    kotak_mpin: str = Field(default="", validation_alias="NEO_MPIN")

    kotak_api_secret: str = ""
    kotak_access_token: str = ""
    kotak_totp_secret: str = ""

    allow_order_submission: bool = Field(default=False, validation_alias="ALLOW_ORDER_SUBMISSION")

    def live_trading_allowed(self) -> bool:
        return (
            self.environment.lower() == "production"
            and self.live_trading_enabled
            and not self.paper_trading
            and self.allow_order_submission
        )


settings = Settings()
