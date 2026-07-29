from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from .borrow import BorrowProvider, BorrowState, ConfiguredBorrowProvider
from .config import Settings
from .domain import DecisionStatus, PortfolioSnapshot, Quote, RiskDecision, Side, SignalProposal


def _utc_now() -> datetime:
    return datetime.now(UTC)


class RiskEngine:
    def __init__(
        self,
        settings: Settings,
        *,
        clock: Callable[[], datetime] | None = None,
        borrow_provider: BorrowProvider | None = None,
    ) -> None:
        self.settings = settings
        self.kill_switch = False
        self._seen_proposals: set[str] = set()
        self._clock = clock or _utc_now
        self.borrow_provider = borrow_provider or ConfiguredBorrowProvider(
            settings.trading_easy_to_borrow_symbols,
            clock=self._clock,
            annual_rate=settings.trading_borrow_rate_annual,
        )

    def set_kill_switch(self, enabled: bool) -> None:
        self.kill_switch = enabled

    def evaluate(
        self,
        proposal: SignalProposal,
        portfolio: PortfolioSnapshot,
        quote: Quote,
    ) -> RiskDecision:
        reasons: list[str] = []
        proposal_key = str(proposal.id)

        if self.kill_switch:
            reasons.append("kill_switch_enabled")
        if proposal_key in self._seen_proposals:
            reasons.append("duplicate_proposal")
        if proposal.confidence < self.settings.trading_min_confidence:
            reasons.append("confidence_below_threshold")
        if quote.spread_bps > self.settings.trading_max_spread_bps:
            reasons.append("spread_above_threshold")

        age = (self._clock() - quote.knowledge_time).total_seconds()
        if age > self.settings.trading_max_data_age_seconds:
            reasons.append("stale_market_data")
        if portfolio.trades_today >= self.settings.trading_max_trades_per_day:
            reasons.append("daily_trade_limit")
        if portfolio.equity <= 0:
            reasons.append("non_positive_equity")
        elif portfolio.daily_pnl / portfolio.equity <= -self.settings.trading_max_daily_loss_pct:
            reasons.append("daily_loss_limit")
        if portfolio.drawdown >= self.settings.trading_max_drawdown_pct:
            reasons.append("drawdown_limit")

        self._seen_proposals.add(proposal_key)
        if reasons:
            return self._rejection(proposal, reasons)

        current = next(
            (
                position.market_value
                for position in portfolio.positions
                if position.symbol == proposal.symbol
            ),
            0.0,
        )
        closing_capacity = (
            abs(min(current, 0.0))
            if proposal.side == Side.BUY
            else max(current, 0.0)
        )
        opening_capacity = self._opening_capacity(
            proposal.side,
            portfolio,
            current,
            closing_capacity,
        )
        approved = min(proposal.target_notional, closing_capacity + opening_capacity)

        if approved <= 0:
            reason = (
                "margin_buying_power_exhausted"
                if self._buying_power(portfolio) <= 0
                else "exposure_limit"
            )
            return self._rejection(proposal, [reason])

        short_open_notional = 0.0
        if proposal.side == Side.SELL:
            short_open_notional = max(0.0, approved - max(current, 0.0))
        if short_open_notional > 1e-9:
            borrow = self.borrow_provider.status(proposal.symbol)
            if borrow is None:
                return self._rejection(proposal, ["borrow_status_missing"])
            borrow_age = (self._clock() - borrow.checked_at).total_seconds()
            if (
                borrow_age < 0
                or borrow_age > self.settings.trading_borrow_status_max_age_seconds
            ):
                return self._rejection(proposal, ["borrow_status_not_current"])
            if borrow.state != BorrowState.EASY_TO_BORROW:
                return self._rejection(
                    proposal,
                    [
                        "borrow_unavailable"
                        if borrow.state in {BorrowState.UNAVAILABLE, BorrowState.RECALLED}
                        else "borrow_not_easy_to_borrow"
                    ],
                )
            short_quantity = short_open_notional / quote.bid
            if not borrow.permits(short_quantity):
                return self._rejection(proposal, ["borrow_quantity_unavailable"])

        resized = approved + 1e-9 < proposal.target_notional
        return RiskDecision(
            proposal_id=proposal.id,
            status=DecisionStatus.RESIZED if resized else DecisionStatus.APPROVED,
            approved_notional=approved,
            reasons=["position_resized"] if resized else ["all_checks_passed"],
        )

    def _opening_capacity(
        self,
        side: Side,
        portfolio: PortfolioSnapshot,
        current: float,
        closing_capacity: float,
    ) -> float:
        if side == Side.BUY:
            per_position_limit = portfolio.equity * min(
                self.settings.trading_max_position_pct,
                self.settings.trading_max_long_position_pct,
            )
            position_room = max(0.0, per_position_limit - max(current, 0.0))
            aggregate_room = max(
                0.0,
                portfolio.equity
                * min(
                    self.settings.trading_max_gross_exposure_pct,
                    self.settings.trading_max_gross_long_exposure_pct,
                )
                - portfolio.gross_long_exposure,
            )
        else:
            per_position_limit = (
                portfolio.equity * self.settings.trading_max_short_position_pct
            )
            position_room = max(0.0, per_position_limit - abs(min(current, 0.0)))
            aggregate_room = max(
                0.0,
                portfolio.equity
                * self.settings.trading_max_gross_short_exposure_pct
                - portfolio.gross_short_exposure,
            )
        total_gross_room = max(
            0.0,
            portfolio.equity * self.settings.trading_max_gross_exposure_pct
            - max(0.0, portfolio.gross_exposure - closing_capacity),
        )
        return min(
            position_room,
            aggregate_room,
            total_gross_room,
            self._buying_power(portfolio) + closing_capacity,
        )

    def _buying_power(self, portfolio: PortfolioSnapshot) -> float:
        if portfolio.buying_power is not None:
            return max(0.0, portfolio.buying_power)
        return max(
            0.0,
            portfolio.equity / self.settings.trading_initial_margin_requirement
            - portfolio.gross_exposure,
        )

    @staticmethod
    def _rejection(
        proposal: SignalProposal,
        reasons: list[str],
    ) -> RiskDecision:
        return RiskDecision(
            proposal_id=proposal.id,
            status=DecisionStatus.REJECTED,
            approved_notional=0,
            reasons=reasons,
        )
