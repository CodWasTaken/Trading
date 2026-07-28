from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from statistics import pstdev

from .domain import NewsEvent, Quote
from .research import FeatureRow


FEATURE_NAMES = ("momentum", "news_score", "volatility", "spread_bps")


class PointInTimeDatasetBuilder:
    """Build labels using only observations known at each feature timestamp."""

    def __init__(
        self,
        *,
        lookback_quotes: int = 20,
        forecast_quotes: int = 5,
        news_window: timedelta = timedelta(hours=24),
    ) -> None:
        if lookback_quotes < 3:
            raise ValueError("lookback_quotes must be at least three")
        if forecast_quotes < 1:
            raise ValueError("forecast_quotes must be positive")
        self.lookback_quotes = lookback_quotes
        self.forecast_quotes = forecast_quotes
        self.news_window = news_window

    def build(self, quotes: list[Quote], news: list[NewsEvent]) -> list[FeatureRow]:
        quotes_by_symbol: dict[str, list[Quote]] = defaultdict(list)
        news_by_symbol: dict[str, list[NewsEvent]] = defaultdict(list)
        for quote in quotes:
            quotes_by_symbol[quote.symbol].append(quote)
        for item in news:
            news_by_symbol[item.symbol].append(item)
        rows: list[FeatureRow] = []
        for symbol, symbol_quotes in quotes_by_symbol.items():
            ordered_quotes = sorted(symbol_quotes, key=lambda item: item.knowledge_time)
            ordered_news = sorted(
                news_by_symbol.get(symbol, []), key=lambda item: item.knowledge_time
            )
            news_cursor = 0
            visible_news: list[NewsEvent] = []
            last_index = len(ordered_quotes) - self.forecast_quotes
            for index in range(self.lookback_quotes, last_index):
                current = ordered_quotes[index]
                while (
                    news_cursor < len(ordered_news)
                    and ordered_news[news_cursor].knowledge_time <= current.knowledge_time
                ):
                    visible_news.append(ordered_news[news_cursor])
                    news_cursor += 1
                cutoff = current.knowledge_time - self.news_window
                visible_news = [
                    item for item in visible_news if item.knowledge_time >= cutoff
                ]
                window = ordered_quotes[index - self.lookback_quotes : index + 1]
                mids = [quote.mid for quote in window]
                returns = [
                    mids[position] / mids[position - 1] - 1
                    for position in range(1, len(mids))
                ]
                momentum = mids[-1] / mids[0] - 1
                volatility = pstdev(returns) if len(returns) > 1 else 0.0
                weighted_news = [
                    item.sentiment * item.novelty * item.source_quality
                    for item in visible_news
                ]
                news_score = (
                    sum(weighted_news) / len(weighted_news) if weighted_news else 0.0
                )
                future = ordered_quotes[index + self.forecast_quotes]
                target_return = future.mid / current.mid - 1
                rows.append(
                    FeatureRow(
                        timestamp=current.knowledge_time,
                        symbol=symbol,
                        features=(
                            momentum,
                            news_score,
                            volatility,
                            current.spread_bps,
                        ),
                        target_return=target_return,
                    )
                )
        return sorted(rows, key=lambda item: item.timestamp)
