from __future__ import annotations

from collections import defaultdict, deque
from typing import Protocol

from .domain import NewsEvent, Quote, Side, SignalProposal


class Strategy(Protocol):
    def on_quote(
        self, quote: Quote, news: list[NewsEvent], equity: float
    ) -> SignalProposal | None: ...


class ExplainableCatalystStrategy:
    """Reference strategy for the vertical slice, not a production alpha claim."""

    def __init__(
        self,
        minimum_combined: float = 0.16,
        repeat_combined: float = 0.30,
    ) -> None:
        if minimum_combined <= 0:
            raise ValueError("minimum_combined must be positive")
        if repeat_combined < minimum_combined:
            raise ValueError("repeat_combined must not be below minimum_combined")
        self.minimum_combined = minimum_combined
        self.repeat_combined = repeat_combined
        self._prices: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=12))
        self._last_direction: dict[str, Side] = {}

    def on_quote(
        self, quote: Quote, news: list[NewsEvent], equity: float
    ) -> SignalProposal | None:
        prices = self._prices[quote.symbol]
        prices.append(quote.mid)
        if len(prices) < 5:
            return None

        momentum = (prices[-1] / prices[0]) - 1
        relevant = news[:5]
        if relevant:
            weighted = [
                item.sentiment * item.novelty * item.source_quality for item in relevant
            ]
            news_score = sum(weighted) / len(weighted)
            top = relevant[0]
            news_age_seconds = max(
                0.0, (quote.knowledge_time - top.knowledge_time).total_seconds()
            )
            news_event_type = top.event_type
            news_source = top.source
        else:
            news_score = 0.0
            news_age_seconds = 999_999.0
            news_event_type = "none"
            news_source = "none"

        combined = 8.0 * momentum + 0.65 * news_score
        confidence = min(0.95, 0.50 + abs(combined))
        if abs(combined) < self.minimum_combined:
            return None

        side = Side.BUY if combined > 0 else Side.SELL
        if (
            self._last_direction.get(quote.symbol) == side
            and abs(combined) < self.repeat_combined
        ):
            return None
        self._last_direction[quote.symbol] = side

        target_notional = max(
            100.0, equity * min(0.05, 0.01 + abs(combined) * 0.05)
        )
        rationale = [
            f"short-term momentum={momentum:.3%}",
            f"news catalyst score={news_score:.3f}",
            f"combined signal={combined:.3f}",
        ]
        if relevant:
            rationale.append(f"latest headline: {relevant[0].headline}")

        return SignalProposal(
            symbol=quote.symbol,
            side=side,
            confidence=confidence,
            expected_return=combined / 10,
            target_notional=target_notional,
            reference_price=quote.mid,
            rationale=rationale,
            feature_snapshot={
                "momentum": momentum,
                "news_score": news_score,
                "news_age_seconds": news_age_seconds,
                "news_event_type": news_event_type,
                "news_source": news_source,
                "spread_bps": quote.spread_bps,
                "combined": combined,
                "signal_threshold": self.minimum_combined,
            },
        )
