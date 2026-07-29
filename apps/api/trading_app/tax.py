from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from .cost_model import CostModelConfig, PolishTaxConfig


class EstimatedPolishTaxReport(BaseModel):
    schema_version: int = 1
    report_kind: str = "estimated_polish_capital_gains_tax"
    status: str = "estimate_not_tax_advice"
    generated_at: datetime
    currency: str = "USD"
    realized_pnl_before_explicit_costs: float
    explicit_execution_and_financing_costs: float
    estimated_taxable_gain: float
    estimated_tax: float
    realized_profit_after_costs_before_tax: float
    estimated_realized_profit_after_tax: float
    pre_tax_strategy_equity: float
    estimated_after_tax_equity: float
    funding_fx_cost: float
    estimated_after_tax_and_funding_equity: float
    assumptions: dict[str, object]
    limitations: list[str]


def estimate_polish_tax(
    *,
    starting_equity: float,
    current_equity: float,
    realized_pnl_before_explicit_costs: float,
    explicit_execution_and_financing_costs: float,
    funding_fx_cost: float,
    config: PolishTaxConfig,
) -> EstimatedPolishTaxReport:
    if starting_equity <= 0:
        raise ValueError("starting_equity must be positive")
    if explicit_execution_and_financing_costs < 0 or funding_fx_cost < 0:
        raise ValueError("cost inputs cannot be negative")
    realized_after_costs = (
        realized_pnl_before_explicit_costs
        - explicit_execution_and_financing_costs
    )
    taxable = max(0.0, realized_after_costs - config.prior_loss_offset_usd)
    estimated_tax = taxable * config.estimated_capital_gains_rate
    return EstimatedPolishTaxReport(
        generated_at=datetime.now(UTC),
        realized_pnl_before_explicit_costs=realized_pnl_before_explicit_costs,
        explicit_execution_and_financing_costs=(
            explicit_execution_and_financing_costs
        ),
        estimated_taxable_gain=taxable,
        estimated_tax=estimated_tax,
        realized_profit_after_costs_before_tax=realized_after_costs,
        estimated_realized_profit_after_tax=realized_after_costs - estimated_tax,
        pre_tax_strategy_equity=current_equity,
        estimated_after_tax_equity=current_equity - estimated_tax,
        funding_fx_cost=funding_fx_cost,
        estimated_after_tax_and_funding_equity=(
            current_equity - estimated_tax - funding_fx_cost
        ),
        assumptions={
            **config.model_dump(mode="json"),
            "tax_base_proxy": "realized_pnl_less_execution_and_financing_costs",
            "funding_fx_treatment": "separate_cash_flow_cost",
        },
        limitations=[
            "This is a simplified estimate, not tax advice.",
            "Polish tax treatment depends on the taxpayer, account, FX conversion dates, "
            "deductibility, loss carryforwards, and current law.",
            "Unrealized PnL is excluded from the taxable-gain proxy.",
            "Tax is reported separately and is never deducted from individual trades.",
        ],
    )


def write_estimated_polish_tax_report(
    output: str | Path,
    *,
    starting_equity: float,
    current_equity: float,
    realized_pnl_before_explicit_costs: float,
    explicit_execution_and_financing_costs: float,
    cost_model: CostModelConfig,
) -> EstimatedPolishTaxReport:
    report = estimate_polish_tax(
        starting_equity=starting_equity,
        current_equity=current_equity,
        realized_pnl_before_explicit_costs=realized_pnl_before_explicit_costs,
        explicit_execution_and_financing_costs=(
            explicit_execution_and_financing_costs
        ),
        funding_fx_cost=cost_model.funding.estimated_conversion_cost_usd,
        config=cost_model.polish_tax,
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
