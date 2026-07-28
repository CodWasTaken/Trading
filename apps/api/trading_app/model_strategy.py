from __future__ import annotations

import math
from collections import defaultdict, deque
from statistics import pstdev

from .domain import NewsEvent, Quote, Side, SignalProposal
from .research import RidgeReturnModel


class ChampionModelStrategy:
    """Live inference wrapper for a registered point-in-time return model."""

    SUPPORTED_FEATURES = ("momentum", "news_score", "volatility", "spread_bps")

    def __init__(self, model: RidgeReturnModel, minimum_edge: float = 0.0005) -> None:
        unsupported = set(model.feature_names) - set(self.SUPPORTED_FEATURES)
        if unsupported:
            raise ValueError(
                f"Champion features {sorted(unsupported)!r} are not available in live inference"
            )
        if not {"momentum", "news_score", "volatility"}.issubset(model.feature_names):
            raise ValueError(
                "Champion must include momentum, news_score, and volatility"
            )
        self.model = model
        self.minimum_edge = minimum_edge
        self._prices: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=20))
        self._last_direction: dict[str, Side] = {}

    def on_quote(
        self, quote: Quote, news: list[NewsEvent], equity: float
    ) -> SignalProposal | None:
        prices = self._prices[quote.symbol]
        prices.append(quote.mid)
        if len(prices) < 20:
            return None
        returns = [
            prices[index] / prices[index - 1] - 1 for index in range(1, len(prices))
        ]
        momentum = prices[-1] / prices[0] - 1
        volatility = pstdev(returns) if len(returns) > 1 else 0.0
        weighted_news = [
            item.sentiment * item.novelty * item.source_quality for item in news[:5]
        ]
        news_score = sum(weighted_news) / len(weighted_news) if weighted_news else 0.0
        available_features = {
            "momentum": momentum,
            "news_score": news_score,
            "volatility": volatility,
            "spread_bps": quote.spread_bps,
        }
        features = tuple(available_features[name] for name in self.model.feature_names)
        prediction = self.model.predict(features)
        if abs(prediction) <= self.minimum_edge:
            return None
        side = Side.BUY if prediction > 0 else Side.SELL
        if (
            self._last_direction.get(quote.symbol) == side
            and abs(prediction) < self.minimum_edge * 2
        ):
            return None
        self._last_direction[quote.symbol] = side
        confidence = min(
            0.95, 0.50 + abs(prediction) / max(self.minimum_edge * 8, 1e-9)
        )
        target_pct = min(0.05, 0.01 + math.sqrt(abs(prediction)) * 0.10)
        return SignalProposal(
            symbol=quote.symbol,
            side=side,
            confidence=confidence,
            expected_return=prediction,
            target_notional=max(100.0, equity * target_pct),
            reference_price=quote.mid,
            rationale=[
                f"champion predicted return={prediction:.4%}",
                f"momentum={momentum:.4%}",
                f"news score={news_score:.3f}",
                f"volatility={volatility:.4%}",
            ],
            feature_snapshot={
                **available_features,
                "model_prediction": prediction,
                "champion_model": True,
            },
        )
