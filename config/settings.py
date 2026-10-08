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

    starting_capital: float = Field(default=0.0, validation_alias="CAPITAL")
    runtime_equity_required: bool = Field(default=True, validation_alias="RUNTIME_EQUITY_REQUIRED")
    max_notional_fraction: float = Field(default=0.10, validation_alias="MAX_NOTIONAL_FRACTION")
    max_option_spread_pct: float = Field(default=0.05, validation_alias="MAX_OPTION_SPREAD_PCT")
    option_quote_max_age_seconds: float = Field(default=60.0, validation_alias="OPTION_QUOTE_MAX_AGE_SECONDS")
    candle_max_age_minutes_5m: float = Field(default=10.0, validation_alias="CANDLE_MAX_AGE_MINUTES_5M")
    candle_max_age_minutes_15m: float = Field(default=20.0, validation_alias="CANDLE_MAX_AGE_MINUTES_15M")
    risk_fraction: float = Field(default=0.01, validation_alias="RISK_FRACTION")
    max_trades_per_day: int = Field(default=5, validation_alias="MAX_TRADES_PER_DAY")
    max_daily_loss_fraction: float = Field(default=0.02, validation_alias="MAX_DAILY_LOSS_FRACTION")
    daily_target_pct: float = Field(default=0.0, validation_alias="DAILY_TARGET_PCT")
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

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-6-luna", validation_alias="OPENAI_MODEL")
    neo_nifty_neosymbol: str = Field(default="", validation_alias="NEO_NIFTY_NEOSYMBOL")
    mcx_capture_symbols: str = Field(default="CRUDEOIL,GOLD,SILVER", validation_alias="MCX_CAPTURE_SYMBOLS")
    whatsapp_alerts_enabled: bool = Field(default=False, validation_alias="WHATSAPP_ALERTS_ENABLED")
    whatsapp_api_token: str = Field(default="", validation_alias="WHATSAPP_API_TOKEN")
    whatsapp_phone_number_id: str = Field(default="", validation_alias="WHATSAPP_PHONE_NUMBER_ID")
    whatsapp_graph_version: str = Field(default="v23.0", validation_alias="WHATSAPP_GRAPH_VERSION")
    whatsapp_alert_interval_minutes: int = Field(default=5, validation_alias="WHATSAPP_ALERT_INTERVAL_MINUTES")
    whatsapp_duplicate_suppression_minutes: int = Field(default=15, validation_alias="WHATSAPP_DUPLICATE_SUPPRESSION_MINUTES")
    market_data_root: str = Field(default="data/market", validation_alias="MARKET_DATA_ROOT")
    capture_nifty_enabled: bool = Field(default=True, validation_alias="CAPTURE_NIFTY_ENABLED")
    capture_mcx_enabled: bool = Field(default=True, validation_alias="CAPTURE_MCX_ENABLED")

    allow_order_submission: bool = Field(default=False, validation_alias="ALLOW_ORDER_SUBMISSION")

    def live_trading_allowed(self) -> bool:
        return (
            self.environment.lower() == "production"
            and self.live_trading_enabled
            and not self.paper_trading
            and self.allow_order_submission
        )


settings = Settings()
