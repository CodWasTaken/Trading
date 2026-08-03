from __future__ import annotations

import math
from collections import defaultdict, deque
from statistics import pstdev

from .domain import NewsEvent, Quote, Side, SignalProposal
from .modeling import ContextTradingModel, RidgeModelAdapter, TradingModel
from .research import RidgeReturnModel


class ChampionModelStrategy:
    """Live inference wrapper for a hash-verified registered paper model."""

    SUPPORTED_FEATURES = ("momentum", "news_score", "volatility", "spread_bps")

    def __init__(
        self,
        model: TradingModel | RidgeReturnModel,
        minimum_edge: float = 0.0005,
    ) -> None:
        # Registry schema v4 and older tests may still hand us the legacy ridge
        # implementation directly. Preserve that stable API while routing all
        # inference through the governed model contract.
        governed_model: TradingModel = (
            RidgeModelAdapter(model) if isinstance(model, RidgeReturnModel) else model
        )
        capabilities = governed_model.capabilities
        unsupported = set(capabilities.feature_names) - set(self.SUPPORTED_FEATURES)
        if unsupported:
            raise ValueError(
                f"Model features {sorted(unsupported)!r} are not available in live inference"
            )
        if capabilities.task != "return_regression" or not capabilities.live_compatible:
            raise ValueError("Only live-compatible return-regression models may trade")
        if capabilities.required_history > 20:
            raise ValueError("Model requires more live feature history than the runtime supports")
        if minimum_edge <= 0:
            raise ValueError("minimum_edge must be positive")
        self.model = governed_model
        self.minimum_edge = minimum_edge
        self._prices: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=20))
        self._contexts: dict[str, deque[tuple[float, ...]]] = defaultdict(
            lambda: deque(maxlen=20)
        )
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
        relevant = news[:5]
        weighted_news = [
            item.sentiment * item.novelty * item.source_quality for item in relevant
        ]
        news_score = sum(weighted_news) / len(weighted_news) if weighted_news else 0.0
        available_features = {
            "momentum": momentum,
            "news_score": news_score,
            "volatility": volatility,
            "spread_bps": quote.spread_bps,
        }
        features = tuple(
            available_features[name] for name in self.model.capabilities.feature_names
        )
        context = self._contexts[quote.symbol]
        context.append(features)
        required = self.model.capabilities.required_history
        if len(context) < required:
            return None
        if isinstance(self.model, ContextTradingModel):
            prediction = self.model.predict_context(tuple(context))
        else:
            prediction = self.model.predict(features)
        if not math.isfinite(prediction):
            return None
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
        capabilities = self.model.capabilities
        return SignalProposal(
            symbol=quote.symbol,
            side=side,
            confidence=confidence,
            expected_return=prediction,
            target_notional=max(100.0, equity * target_pct),
            reference_price=quote.mid,
            rationale=[
                f"{capabilities.implementation} predicted return={prediction:.4%}",
                f"momentum={momentum:.4%}",
                f"news score={news_score:.3f}",
                f"volatility={volatility:.4%}",
            ],
            feature_snapshot={
                **available_features,
                "news_event_type": relevant[0].event_type if relevant else "none",
                "news_source": relevant[0].source if relevant else "none",
                "model_prediction": prediction,
                "signal_threshold": self.minimum_edge,
                "model_family": capabilities.family,
                "model_implementation": capabilities.implementation,
                "paper_only": True,
            },
        )
