from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    trading_env: str = "development"
    trading_demo_mode: bool = True
    trading_execution_mode: Literal["internal-paper", "alpaca-paper"] = "internal-paper"
    trading_starting_cash: float = 100_000.0
    trading_symbols: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
    )
    trading_decision_interval_seconds: float = 3.0
    trading_max_position_pct: float = 0.05
    trading_max_gross_exposure_pct: float = 0.60
    trading_max_long_position_pct: float = 0.05
    trading_max_short_position_pct: float = 0.03
    trading_max_gross_long_exposure_pct: float = 0.60
    trading_max_gross_short_exposure_pct: float = 0.30
    trading_initial_margin_requirement: float = 0.50
    trading_maintenance_margin_requirement: float = 0.30
    trading_easy_to_borrow_symbols: Annotated[list[str], NoDecode] = Field(
        default_factory=list
    )
    trading_borrow_status_max_age_seconds: int = Field(default=300, gt=0)
    trading_borrow_rate_annual: float = 0.03
    trading_dividend_replacement_rate_annual: float = 0.02
    trading_slippage_bps: float = Field(default=2.0, ge=0)
    trading_commission_bps: float = Field(default=0.0, ge=0)
    trading_regulatory_sell_fee_bps: float = Field(default=0.05, ge=0)
    trading_market_impact_stress_bps: float = Field(default=2.0, ge=0)
    trading_margin_interest_rate_annual: float = Field(default=0.08, ge=0)
    trading_pln_to_usd_conversion_bps: float = Field(default=0.0, ge=0)
    trading_pln_to_usd_conversion_amount_usd: float = Field(default=0.0, ge=0)
    trading_estimated_polish_capital_gains_rate: float = Field(
        default=0.19,
        ge=0,
        le=1,
    )
    trading_max_daily_loss_pct: float = 0.01
    trading_max_drawdown_pct: float = 0.08
    trading_min_confidence: float = 0.65
    trading_max_spread_bps: float = 35.0
    trading_max_data_age_seconds: int = 30
    trading_max_trades_per_day: int = 10
    trading_order_fill_timeout_seconds: float = 15.0
    trading_order_poll_interval_seconds: float = 0.25
    trading_control_api_key: str | None = None
    trading_database_path: str = ".trading/events.db"
    trading_strategy_mode: Literal["explainable", "champion"] = "explainable"
    trading_model_registry_path: str = ".trading/models"
    trading_entity_catalog_path: str | None = None
    trading_cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    alpaca_trading_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_base_url: str = "https://data.alpaca.markets"
    alpaca_data_stream_url: str = "wss://stream.data.alpaca.markets/v2/iex"
    alpaca_news_stream_url: str = "wss://stream.data.alpaca.markets/v1beta1/news"

    sec_user_agent: str | None = None
    sec_data_base_url: str = "https://data.sec.gov"

    @field_validator(
        "trading_symbols",
        "trading_cors_origins",
        "trading_easy_to_borrow_symbols",
        mode="before",
    )
    @classmethod
    def split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("trading_symbols", "trading_easy_to_borrow_symbols")
    @classmethod
    def normalize_symbols(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(symbol.upper() for symbol in value))

    @model_validator(mode="after")
    def validate_risk_parameters(self) -> Settings:
        fractions = {
            "trading_max_long_position_pct": self.trading_max_long_position_pct,
            "trading_max_short_position_pct": self.trading_max_short_position_pct,
            "trading_max_gross_long_exposure_pct": self.trading_max_gross_long_exposure_pct,
            "trading_max_gross_short_exposure_pct": self.trading_max_gross_short_exposure_pct,
            "trading_initial_margin_requirement": self.trading_initial_margin_requirement,
            "trading_maintenance_margin_requirement": (
                self.trading_maintenance_margin_requirement
            ),
        }
        if any(value <= 0 or value > 1 for value in fractions.values()):
            raise ValueError("position, exposure, and margin fractions must be in (0, 1]")
        if (
            self.trading_maintenance_margin_requirement
            > self.trading_initial_margin_requirement
        ):
            raise ValueError("maintenance margin cannot exceed initial margin")
        if self.trading_borrow_rate_annual < 0:
            raise ValueError("borrow rate cannot be negative")
        if self.trading_dividend_replacement_rate_annual < 0:
            raise ValueError("dividend replacement rate cannot be negative")
        return self

    def require_alpaca_credentials(self) -> tuple[str, str]:
        if not self.alpaca_api_key or not self.alpaca_api_secret:
            raise RuntimeError("Alpaca credentials are required for Alpaca data or paper mode")
        return self.alpaca_api_key, self.alpaca_api_secret

    def require_sec_user_agent(self) -> str:
        if not self.sec_user_agent:
            raise RuntimeError(
                "SEC_USER_AGENT is required and must include an application name and email"
            )
        return self.sec_user_agent


@lru_cache
def get_settings() -> Settings:
    return Settings()
