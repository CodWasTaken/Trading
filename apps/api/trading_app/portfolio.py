from __future__ import annotations

from collections import defaultdict

from .domain import Fill, PortfolioSnapshot, Position, Quote, Side


class Portfolio:
    def __init__(self, starting_cash: float) -> None:
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self._quantity: dict[str, float] = defaultdict(float)
        self._average_price: dict[str, float] = defaultdict(float)
        self._last_price: dict[str, float] = {}
        self.trades_today = 0
        self.day_start_equity = starting_cash
        self.peak_equity = starting_cash

    def mark(self, quote: Quote) -> None:
        self._last_price[quote.symbol] = quote.mid

    def apply_fill(self, fill: Fill) -> None:
        symbol = fill.symbol
        quantity = fill.quantity
        old_quantity = self._quantity[symbol]

        if fill.side == Side.BUY:
            new_quantity = old_quantity + quantity
            old_cost = old_quantity * self._average_price[symbol]
            self._average_price[symbol] = (old_cost + quantity * fill.price) / new_quantity
            self._quantity[symbol] = new_quantity
            self.cash -= quantity * fill.price
        else:
            sell_quantity = min(quantity, max(0.0, old_quantity))
            self._quantity[symbol] = old_quantity - sell_quantity
            self.cash += sell_quantity * fill.price
            if self._quantity[symbol] <= 1e-9:
                self._quantity.pop(symbol, None)
                self._average_price.pop(symbol, None)

        self._last_price[symbol] = fill.price
        self.trades_today += 1

    def position_value(self, symbol: str) -> float:
        return self._quantity.get(symbol, 0.0) * self._last_price.get(symbol, 0.0)

    def snapshot(self) -> PortfolioSnapshot:
        positions = [
            Position(
                symbol=symbol,
                quantity=quantity,
                average_price=self._average_price[symbol],
                last_price=self._last_price.get(symbol, self._average_price[symbol]),
            )
            for symbol, quantity in sorted(self._quantity.items())
            if abs(quantity) > 1e-9
        ]
        gross = sum(abs(position.market_value) for position in positions)
        equity = self.cash + sum(position.market_value for position in positions)
        self.peak_equity = max(self.peak_equity, equity)
        daily_pnl = equity - self.day_start_equity
        drawdown = 0.0 if self.peak_equity <= 0 else (self.peak_equity - equity) / self.peak_equity
        return PortfolioSnapshot(
            cash=self.cash,
            equity=equity,
            gross_exposure=gross,
            daily_pnl=daily_pnl,
            drawdown=drawdown,
            positions=positions,
            trades_today=self.trades_today,
        )
