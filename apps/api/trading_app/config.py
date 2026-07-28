from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
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
    trading_max_daily_loss_pct: float = 0.01
    trading_max_drawdown_pct: float = 0.08
    trading_min_confidence: float = 0.65
    trading_max_spread_bps: float = 35.0
    trading_max_data_age_seconds: int = 30
    trading_max_trades_per_day: int = 10
    trading_database_path: str = ".trading/events.db"
    trading_strategy_mode: Literal["explainable", "champion"] = "explainable"
    trading_model_registry_path: str = ".trading/models"
    trading_cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    alpaca_trading_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_stream_url: str = "wss://stream.data.alpaca.markets/v2/iex"
    alpaca_news_stream_url: str = "wss://stream.data.alpaca.markets/v1beta1/news"

    @field_validator("trading_symbols", "trading_cors_origins", mode="before")
    @classmethod
    def split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("trading_symbols")
    @classmethod
    def normalize_symbols(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(symbol.upper() for symbol in value))

    def require_alpaca_credentials(self) -> tuple[str, str]:
        if not self.alpaca_api_key or not self.alpaca_api_secret:
            raise RuntimeError("Alpaca credentials are required for alpaca-paper mode")
        return self.alpaca_api_key, self.alpaca_api_secret


@lru_cache
def get_settings() -> Settings:
    return Settings()
