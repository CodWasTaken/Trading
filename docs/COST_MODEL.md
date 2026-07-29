# Cost model and Polish tax estimate

The canonical initial paper assumptions are in
`config/costs/conservative-us-paper-v1.json`. Copy and version that file for a
new experiment; do not edit a manifest after results have been produced. Model
registry records, holdout reports, monitoring reports, and replay reports retain
the normalized manifest and its SHA-256.

## Execution and financing

Pre-tax strategy performance includes:

- observed bid/ask spread;
- slippage;
- commission where configured;
- regulatory sell-side fees;
- market-impact stress;
- borrow fees;
- margin interest;
- dividend-replacement costs.

Spread and slippage are embedded in fill price. Commission, regulatory fees, and
impact stress are explicit paper cash deductions. Borrow, margin interest, and
dividend replacement accrue with elapsed time. Reports keep each component
separate so an assumption can be audited without reconstructing a blended fee.

## Funding and PLN→USD conversion

Currency conversion is a funding cash flow. Configure both the converted USD
amount and conversion basis points, then explicitly record the conversion. It
is not charged merely because an order was filled and is never multiplied by
turnover. A zero conversion amount disables the estimated funding cost.

## Estimated Polish tax

`GET /v1/reports/polish-tax-estimate` and replay reports provide:

- pre-tax strategy equity after execution and financing costs;
- an estimated taxable realized-gain proxy;
- estimated tax;
- estimated after-tax equity;
- funding/FX cost shown separately.

Tax is never deducted from individual trades or the pre-tax equity curve. The
calculation is intentionally simplified, excludes unrealized PnL from its
taxable-gain proxy, and is labelled `estimate_not_tax_advice`. Actual Polish tax
depends on current law, the taxpayer and account, FX conversion dates,
deductibility, documentation, and loss carryforwards. Use a qualified Polish tax
professional for filing decisions.
