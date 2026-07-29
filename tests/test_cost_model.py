import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from trading_app.broker import InternalPaperBroker
from trading_app.cost_model import (
    CostModelConfig,
    ExecutionCostConfig,
    FundingCostConfig,
    PolishTaxConfig,
    load_cost_model,
)
from trading_app.domain import Fill, Order, Quote, Side
from trading_app.portfolio import Portfolio
from trading_app.tax import estimate_polish_tax, write_estimated_polish_tax_report


def order(side: Side = Side.SELL) -> Order:
    return Order(
        proposal_id=uuid4(),
        symbol="AAPL",
        side=side,
        quantity=10,
        requested_price=100,
    )


def test_versioned_cost_manifest_has_stable_hash(tmp_path) -> None:
    source = tmp_path / "costs.json"
    source.write_text(
        json.dumps(CostModelConfig().model_dump(mode="json")),
        encoding="utf-8",
    )
    first = load_cost_model(source)
    second = load_cost_model(source)
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.schema_version == 1


@pytest.mark.asyncio
async def test_paper_fill_separates_embedded_and_cash_execution_costs() -> None:
    costs = ExecutionCostConfig(
        observed_spread_bps=10,
        slippage_bps=2,
        commission_bps=1,
        regulatory_sell_fee_bps=0.5,
        market_impact_stress_bps=3,
    )
    quote = Quote(symbol="AAPL", bid=99.95, ask=100.05)
    fill = await InternalPaperBroker(cost_config=costs).execute(
        order(Side.SELL),
        quote,
    )

    assert fill.observed_spread_cost == pytest.approx(0.5)
    assert fill.slippage_cost > 0
    assert fill.commission > 0
    assert fill.regulatory_fees > 0
    assert fill.market_impact_stress_cost > 0
    assert fill.cash_fees == pytest.approx(
        fill.commission + fill.regulatory_fees + fill.market_impact_stress_cost
    )

    portfolio = Portfolio(100_000)
    portfolio.apply_fill(fill)
    snapshot = portfolio.snapshot()
    assert snapshot.execution_costs == pytest.approx(fill.total_execution_cost)
    assert snapshot.cash_execution_fees == pytest.approx(fill.cash_fees)
    assert snapshot.cash == pytest.approx(
        100_000 + fill.quantity * fill.price - fill.cash_fees
    )


def test_borrow_margin_and_dividend_costs_are_separate() -> None:
    start = datetime(2025, 1, 2, 15, tzinfo=UTC)
    short = Portfolio(
        100_000,
        borrow_rate_annual=0.365,
        dividend_replacement_rate_annual=0.365,
    )
    short.apply_fill(
        Fill(
            order_id=uuid4(),
            symbol="AAPL",
            side=Side.SELL,
            quantity=10,
            price=100,
            slippage_bps=0,
        )
    )
    short.accrue_financing(start)
    short.accrue_financing(start + timedelta(days=1))
    assert short.snapshot().borrow_costs == pytest.approx(1)
    assert short.snapshot().dividend_replacement_costs == pytest.approx(1)

    leveraged = Portfolio(1_000, margin_interest_rate_annual=0.365)
    leveraged.apply_fill(
        Fill(
            order_id=uuid4(),
            symbol="AAPL",
            side=Side.BUY,
            quantity=20,
            price=100,
            slippage_bps=0,
        )
    )
    leveraged.accrue_financing(start)
    leveraged.accrue_financing(start + timedelta(days=1))
    assert leveraged.snapshot().margin_interest_costs == pytest.approx(1)


def test_pln_usd_conversion_is_explicit_funding_cash_flow() -> None:
    funding = FundingCostConfig(
        pln_to_usd_conversion_bps=25,
        conversion_amount_usd=20_000,
    )
    portfolio = Portfolio(100_000)
    portfolio.apply_fill(
        Fill(
            order_id=uuid4(),
            symbol="AAPL",
            side=Side.BUY,
            quantity=10,
            price=100,
            slippage_bps=0,
        )
    )
    assert portfolio.snapshot().funding_fx_costs == 0

    charged = portfolio.apply_funding_conversion(
        funding.conversion_amount_usd,
        funding.pln_to_usd_conversion_bps,
    )
    assert charged == pytest.approx(50)
    assert portfolio.snapshot().funding_fx_costs == pytest.approx(50)


def test_polish_tax_is_separate_estimate_and_never_a_trade_fee(tmp_path) -> None:
    cost_model = CostModelConfig(
        funding=FundingCostConfig(
            pln_to_usd_conversion_bps=25,
            conversion_amount_usd=20_000,
        ),
        polish_tax=PolishTaxConfig(estimated_capital_gains_rate=0.19),
    )
    report = estimate_polish_tax(
        starting_equity=100_000,
        current_equity=110_000,
        realized_pnl_before_explicit_costs=10_000,
        explicit_execution_and_financing_costs=1_000,
        funding_fx_cost=cost_model.funding.estimated_conversion_cost_usd,
        config=cost_model.polish_tax,
    )
    assert report.status == "estimate_not_tax_advice"
    assert report.estimated_taxable_gain == pytest.approx(9_000)
    assert report.estimated_tax == pytest.approx(1_710)
    assert report.pre_tax_strategy_equity == 110_000
    assert report.estimated_after_tax_equity == pytest.approx(108_290)
    assert report.estimated_after_tax_and_funding_equity == pytest.approx(108_240)
    assert any("not tax advice" in item for item in report.limitations)

    output = tmp_path / "polish-tax-estimate.json"
    written = write_estimated_polish_tax_report(
        output,
        starting_equity=100_000,
        current_equity=110_000,
        realized_pnl_before_explicit_costs=10_000,
        explicit_execution_and_financing_costs=1_000,
        cost_model=cost_model,
    )
    assert written.estimated_tax == report.estimated_tax
    assert json.loads(output.read_text())["status"] == "estimate_not_tax_advice"
