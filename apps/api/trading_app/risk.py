from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

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
    ) -> None:
        self.settings = settings
        self.kill_switch = False
        self._seen_proposals: set[str] = set()
        self._clock = clock or _utc_now

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
        if proposal.side == Side.SELL:
            current = next(
                (
                    position.market_value
                    for position in portfolio.positions
                    if position.symbol == proposal.symbol
                ),
                0.0,
            )
            if current <= 0:
                reasons.append("short_selling_disabled")

        self._seen_proposals.add(proposal_key)
        if reasons:
            return RiskDecision(
                proposal_id=proposal.id,
                status=DecisionStatus.REJECTED,
                approved_notional=0,
                reasons=reasons,
            )

        max_position = portfolio.equity * self.settings.trading_max_position_pct
        current_position = next(
            (
                abs(position.market_value)
                for position in portfolio.positions
                if position.symbol == proposal.symbol
            ),
            0.0,
        )
        if proposal.side == Side.SELL:
            # Closing or reducing a long position lowers exposure, so buy-side capacity limits
            # must not prevent the exit. Short selling remains blocked above.
            approved = min(proposal.target_notional, current_position)
        else:
            position_room = max(0.0, max_position - current_position)
            gross_room = max(
                0.0,
                portfolio.equity * self.settings.trading_max_gross_exposure_pct
                - portfolio.gross_exposure,
            )
            approved = min(proposal.target_notional, position_room, gross_room)

        if approved <= 0:
            return RiskDecision(
                proposal_id=proposal.id,
                status=DecisionStatus.REJECTED,
                approved_notional=0,
                reasons=["exposure_limit"],
            )

        resized = approved + 1e-9 < proposal.target_notional
        return RiskDecision(
            proposal_id=proposal.id,
            status=DecisionStatus.RESIZED if resized else DecisionStatus.APPROVED,
            approved_notional=approved,
            reasons=["position_resized"] if resized else ["all_checks_passed"],
        )
