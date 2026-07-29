from __future__ import annotations

from collections import defaultdict
from datetime import UTC, date, datetime

from .domain import (
    Fill,
    PortfolioSnapshot,
    Position,
    PositionEffect,
    PositionTransition,
    Quote,
    Side,
)

_EPSILON = 1e-9


def classify_position_effect(
    old_quantity: float,
    side: Side,
    quantity: float,
) -> PositionEffect:
    delta = quantity if side == Side.BUY else -quantity
    new_quantity = old_quantity + delta
    if abs(old_quantity) <= _EPSILON:
        return (
            PositionEffect.OPEN_LONG
            if delta > 0
            else PositionEffect.OPEN_SHORT
        )
    if old_quantity > 0:
        if delta > 0:
            return PositionEffect.INCREASE_LONG
        if new_quantity > _EPSILON:
            return PositionEffect.PARTIAL_CLOSE_LONG
        if abs(new_quantity) <= _EPSILON:
            return PositionEffect.CLOSE_LONG
        return PositionEffect.REVERSE_LONG_TO_SHORT
    if delta < 0:
        return PositionEffect.INCREASE_SHORT
    if new_quantity < -_EPSILON:
        return PositionEffect.PARTIAL_COVER_SHORT
    if abs(new_quantity) <= _EPSILON:
        return PositionEffect.COVER_SHORT
    return PositionEffect.REVERSE_SHORT_TO_LONG


class Portfolio:
    def __init__(
        self,
        starting_cash: float,
        *,
        initial_margin_requirement: float = 0.50,
        maintenance_margin_requirement: float = 0.30,
        borrow_rate_annual: float = 0.0,
        dividend_replacement_rate_annual: float = 0.0,
        margin_interest_rate_annual: float = 0.0,
    ) -> None:
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if not 0 < maintenance_margin_requirement <= initial_margin_requirement <= 1:
            raise ValueError("margin requirements must satisfy 0 < maintenance <= initial <= 1")
        if (
            borrow_rate_annual < 0
            or dividend_replacement_rate_annual < 0
            or margin_interest_rate_annual < 0
        ):
            raise ValueError("short financing rates cannot be negative")
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.initial_margin_requirement = initial_margin_requirement
        self.maintenance_margin_requirement = maintenance_margin_requirement
        self.borrow_rate_annual = borrow_rate_annual
        self.dividend_replacement_rate_annual = dividend_replacement_rate_annual
        self.margin_interest_rate_annual = margin_interest_rate_annual
        self._quantity: dict[str, float] = defaultdict(float)
        self._average_price: dict[str, float] = defaultdict(float)
        self._last_price: dict[str, float] = {}
        self._realized_pnl: dict[str, float] = defaultdict(float)
        self._long_realized_pnl = 0.0
        self._short_realized_pnl = 0.0
        self.borrow_costs = 0.0
        self.dividend_replacement_costs = 0.0
        self.margin_interest_costs = 0.0
        self.execution_costs = 0.0
        self.cash_execution_fees = 0.0
        self.funding_fx_costs = 0.0
        self.transitions: list[PositionTransition] = []
        self.trades_today = 0
        self.day_start_equity = starting_cash
        self.peak_equity = starting_cash
        self._session_date: date | None = None
        self._last_financing_time: datetime | None = None

    def mark(self, quote: Quote) -> None:
        self._last_price[quote.symbol] = quote.mid

    def quantity(self, symbol: str) -> float:
        return self._quantity.get(symbol, 0.0)

    def accrue_financing(self, as_of: datetime) -> float:
        timestamp = as_of.astimezone(UTC)
        if self._last_financing_time is None:
            self._last_financing_time = timestamp
            return 0.0
        elapsed_seconds = (timestamp - self._last_financing_time).total_seconds()
        if elapsed_seconds <= 0:
            return 0.0
        year_fraction = elapsed_seconds / (365.0 * 24 * 60 * 60)
        gross_short = sum(
            abs(quantity)
            * self._last_price.get(symbol, self._average_price[symbol])
            for symbol, quantity in self._quantity.items()
            if quantity < 0
        )
        borrow = gross_short * self.borrow_rate_annual * year_fraction
        replacement = (
            gross_short * self.dividend_replacement_rate_annual * year_fraction
        )
        margin_interest = (
            max(0.0, -self.cash)
            * self.margin_interest_rate_annual
            * year_fraction
        )
        total = borrow + replacement + margin_interest
        self.cash -= total
        self.borrow_costs += borrow
        self.dividend_replacement_costs += replacement
        self.margin_interest_costs += margin_interest
        self._last_financing_time = timestamp
        return total

    def apply_dividend(self, symbol: str, amount_per_share: float) -> float:
        if amount_per_share < 0:
            raise ValueError("dividend amount cannot be negative")
        quantity = self._quantity.get(symbol, 0.0)
        if quantity >= 0:
            return 0.0
        replacement = abs(quantity) * amount_per_share
        self.cash -= replacement
        self.dividend_replacement_costs += replacement
        return replacement

    def apply_funding_conversion(
        self,
        amount_usd: float,
        conversion_cost_bps: float,
    ) -> float:
        if amount_usd < 0 or conversion_cost_bps < 0:
            raise ValueError("funding conversion inputs cannot be negative")
        cost = amount_usd * conversion_cost_bps / 10_000
        self.cash -= cost
        self.funding_fx_costs += cost
        return cost

    def rollover_session(self, as_of: datetime) -> bool:
        """Reset daily controls once when market-data dates advance."""
        session_date = as_of.date()
        if self._session_date is None:
            self._session_date = session_date
            return False
        if session_date <= self._session_date:
            return False
        equity = self._current_equity()
        self._session_date = session_date
        self.day_start_equity = equity
        self.trades_today = 0
        self.peak_equity = max(self.peak_equity, equity)
        return True

    def apply_fill(self, fill: Fill) -> PositionTransition:
        symbol = fill.symbol
        old_quantity = self._quantity[symbol]
        delta = fill.quantity if fill.side == Side.BUY else -fill.quantity
        new_quantity = old_quantity + delta
        effect = classify_position_effect(old_quantity, fill.side, fill.quantity)
        realized = 0.0

        if old_quantity * delta < 0:
            closed_quantity = min(abs(old_quantity), abs(delta))
            realized = (
                closed_quantity
                * (fill.price - self._average_price[symbol])
                * (1.0 if old_quantity > 0 else -1.0)
            )
            self._realized_pnl[symbol] += realized
            if old_quantity > 0:
                self._long_realized_pnl += realized
            else:
                self._short_realized_pnl += realized

        if abs(new_quantity) <= _EPSILON:
            self._quantity.pop(symbol, None)
            self._average_price.pop(symbol, None)
            new_quantity = 0.0
        elif abs(old_quantity) <= _EPSILON or old_quantity * new_quantity < 0:
            self._quantity[symbol] = new_quantity
            self._average_price[symbol] = fill.price
        elif old_quantity * delta > 0:
            old_notional = abs(old_quantity) * self._average_price[symbol]
            added_notional = abs(delta) * fill.price
            self._quantity[symbol] = new_quantity
            self._average_price[symbol] = (
                old_notional + added_notional
            ) / abs(new_quantity)
        else:
            self._quantity[symbol] = new_quantity

        self.cash -= delta * fill.price
        self.cash -= fill.cash_fees
        self.execution_costs += fill.total_execution_cost
        self.cash_execution_fees += fill.cash_fees
        self._last_price[symbol] = fill.price
        self.trades_today += 1
        transition = PositionTransition(
            symbol=symbol,
            side=fill.side,
            quantity=fill.quantity,
            effect=effect,
            previous_quantity=old_quantity,
            resulting_quantity=new_quantity,
            realized_pnl=realized,
        )
        self.transitions.append(transition)
        return transition

    def position_value(self, symbol: str) -> float:
        return self._quantity.get(symbol, 0.0) * self._last_price.get(symbol, 0.0)

    def _current_equity(self) -> float:
        return self.cash + sum(
            quantity * self._last_price.get(symbol, self._average_price[symbol])
            for symbol, quantity in self._quantity.items()
        )

    def snapshot(self) -> PortfolioSnapshot:
        positions = [
            Position(
                symbol=symbol,
                quantity=quantity,
                average_price=self._average_price[symbol],
                last_price=self._last_price.get(symbol, self._average_price[symbol]),
                realized_pnl=self._realized_pnl[symbol],
            )
            for symbol, quantity in sorted(self._quantity.items())
            if abs(quantity) > _EPSILON
        ]
        gross_long = sum(
            position.market_value for position in positions if position.quantity > 0
        )
        gross_short = sum(
            abs(position.market_value)
            for position in positions
            if position.quantity < 0
        )
        gross = gross_long + gross_short
        net = gross_long - gross_short
        equity = self.cash + net
        self.peak_equity = max(self.peak_equity, equity)
        daily_pnl = equity - self.day_start_equity
        drawdown = 0.0 if self.peak_equity <= 0 else (self.peak_equity - equity) / self.peak_equity
        long_unrealized = sum(
            position.unrealized_pnl
            for position in positions
            if position.quantity > 0
        )
        short_unrealized = sum(
            position.unrealized_pnl
            for position in positions
            if position.quantity < 0
        )
        maintenance_required = gross * self.maintenance_margin_requirement
        return PortfolioSnapshot(
            cash=self.cash,
            equity=equity,
            gross_exposure=gross,
            gross_long_exposure=gross_long,
            gross_short_exposure=gross_short,
            net_exposure=net,
            buying_power=max(0.0, equity / self.initial_margin_requirement - gross),
            maintenance_margin_required=maintenance_required,
            margin_excess=equity - maintenance_required,
            realized_pnl=self._long_realized_pnl + self._short_realized_pnl,
            unrealized_pnl=long_unrealized + short_unrealized,
            long_realized_pnl=self._long_realized_pnl,
            short_realized_pnl=self._short_realized_pnl,
            long_unrealized_pnl=long_unrealized,
            short_unrealized_pnl=short_unrealized,
            borrow_costs=self.borrow_costs,
            dividend_replacement_costs=self.dividend_replacement_costs,
            margin_interest_costs=self.margin_interest_costs,
            execution_costs=self.execution_costs,
            cash_execution_fees=self.cash_execution_fees,
            funding_fx_costs=self.funding_fx_costs,
            daily_pnl=daily_pnl,
            drawdown=drawdown,
            positions=positions,
            trades_today=self.trades_today,
        )
