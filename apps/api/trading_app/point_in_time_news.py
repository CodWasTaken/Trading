from __future__ import annotations

from datetime import UTC, datetime

from .domain import NewsEvent
from .news_intelligence import NewsIntelligence


class PointInTimeNewsIntelligence(NewsIntelligence):
    """Apply dated entity relationships using the event's knowledge-time boundary."""

    def enrich(
        self,
        *,
        symbol: str,
        headline: str,
        source: str,
        event_time: datetime,
        knowledge_time: datetime | None = None,
        summary: str = "",
    ) -> NewsEvent | None:
        event = super().enrich(
            symbol=symbol,
            headline=headline,
            source=source,
            event_time=event_time,
            knowledge_time=knowledge_time,
            summary=summary,
        )
        if event is None:
            return None

        as_of = knowledge_time or datetime.now(UTC)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        as_of = as_of.astimezone(UTC)
        combined_text = f"{headline.strip()} {summary.strip()}".strip()
        return event.model_copy(
            update={
                "entities": self.entity_catalog.link(
                    symbol,
                    combined_text,
                    as_of=as_of,
                )
            }
        )
