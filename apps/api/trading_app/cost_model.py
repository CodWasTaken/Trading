from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import Settings


class ExecutionCostConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    observed_spread_bps: float = Field(default=10.0, ge=0)
    slippage_bps: float = Field(default=2.0, ge=0)
    commission_bps: float = Field(default=0.0, ge=0)
    regulatory_sell_fee_bps: float = Field(default=0.05, ge=0)
    market_impact_stress_bps: float = Field(default=2.0, ge=0)
    borrow_rate_annual: float = Field(default=0.03, ge=0)
    margin_interest_rate_annual: float = Field(default=0.08, ge=0)
    dividend_replacement_rate_annual: float = Field(default=0.02, ge=0)

    def trade_cost_fraction(self, previous: float, current: float) -> float:
        delta = current - previous
        buy_turnover = max(0.0, delta)
        sell_turnover = max(0.0, -delta)
        common_bps = (
            self.observed_spread_bps / 2
            + self.slippage_bps
            + self.commission_bps
            + self.market_impact_stress_bps
        )
        return (
            buy_turnover * common_bps
            + sell_turnover * (common_bps + self.regulatory_sell_fee_bps)
        ) / 10_000

    def holding_cost_fraction(self, position: float, periods_per_year: float) -> float:
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        short = max(0.0, -position)
        leveraged = max(0.0, abs(position) - 1.0)
        return (
            short
            * (self.borrow_rate_annual + self.dividend_replacement_rate_annual)
            + leveraged * self.margin_interest_rate_annual
        ) / periods_per_year


class FundingCostConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    pln_to_usd_conversion_bps: float = Field(default=0.0, ge=0)
    conversion_amount_usd: float = Field(default=0.0, ge=0)

    @property
    def estimated_conversion_cost_usd(self) -> float:
        return self.conversion_amount_usd * self.pln_to_usd_conversion_bps / 10_000


class PolishTaxConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    estimated_capital_gains_rate: float = Field(default=0.19, ge=0, le=1)
    prior_loss_offset_usd: float = Field(default=0.0, ge=0)


class CostModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    name: str = "conservative-us-paper-v1"
    execution: ExecutionCostConfig = Field(default_factory=ExecutionCostConfig)
    funding: FundingCostConfig = Field(default_factory=FundingCostConfig)
    polish_tax: PolishTaxConfig = Field(default_factory=PolishTaxConfig)

    @model_validator(mode="after")
    def validate_schema(self) -> CostModelConfig:
        if self.schema_version != 1:
            raise ValueError("Unsupported cost model schema_version")
        return self

    @property
    def manifest_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def load_cost_model(path: str | Path | None) -> CostModelConfig:
    if path is None:
        return CostModelConfig()
    return CostModelConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def legacy_cost_model(transaction_cost_bps: float) -> CostModelConfig:
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative")
    return CostModelConfig(
        name="legacy-single-rate",
        execution=ExecutionCostConfig(
            observed_spread_bps=0,
            slippage_bps=transaction_cost_bps,
            commission_bps=0,
            regulatory_sell_fee_bps=0,
            market_impact_stress_bps=0,
            borrow_rate_annual=0,
            margin_interest_rate_annual=0,
            dividend_replacement_rate_annual=0,
        ),
    )


def cost_model_from_settings(
    settings: Settings,
    *,
    observed_spread_bps: float = 0.0,
    slippage_bps: float | None = None,
) -> CostModelConfig:
    return CostModelConfig(
        name="runtime-paper-costs-v1",
        execution=ExecutionCostConfig(
            observed_spread_bps=observed_spread_bps,
            slippage_bps=(
                settings.trading_slippage_bps
                if slippage_bps is None
                else slippage_bps
            ),
            commission_bps=settings.trading_commission_bps,
            regulatory_sell_fee_bps=settings.trading_regulatory_sell_fee_bps,
            market_impact_stress_bps=settings.trading_market_impact_stress_bps,
            borrow_rate_annual=settings.trading_borrow_rate_annual,
            margin_interest_rate_annual=settings.trading_margin_interest_rate_annual,
            dividend_replacement_rate_annual=(
                settings.trading_dividend_replacement_rate_annual
            ),
        ),
        funding=FundingCostConfig(
            pln_to_usd_conversion_bps=(
                settings.trading_pln_to_usd_conversion_bps
            ),
            conversion_amount_usd=(
                settings.trading_pln_to_usd_conversion_amount_usd
            ),
        ),
        polish_tax=PolishTaxConfig(
            estimated_capital_gains_rate=(
                settings.trading_estimated_polish_capital_gains_rate
            )
        ),
    )
