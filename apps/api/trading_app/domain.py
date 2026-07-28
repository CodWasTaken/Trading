from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field


def utc_now() -> datetime:
    return datetime.now(UTC)


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class DecisionStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    RESIZED = "resized"
    REJECTED = "rejected"
    FILLED = "filled"


class Quote(BaseModel):
    symbol: str
    bid: float = Field(gt=0)
    ask: float = Field(gt=0)
    event_time: datetime = Field(default_factory=utc_now)
    knowledge_time: datetime = Field(default_factory=utc_now)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> float:
        return ((self.ask - self.bid) / self.mid) * 10_000


class NewsEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    symbol: str
    headline: str
    source: str
    sentiment: float = Field(ge=-1, le=1)
    novelty: float = Field(ge=0, le=1)
    source_quality: float = Field(ge=0, le=1)
    event_type: str = "other"
    event_time: datetime = Field(default_factory=utc_now)
    knowledge_time: datetime = Field(default_factory=utc_now)


class SignalProposal(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    symbol: str
    side: Side
    confidence: float = Field(ge=0, le=1)
    expected_return: float
    target_notional: float = Field(gt=0)
    reference_price: float = Field(gt=0)
    rationale: list[str]
    feature_snapshot: dict[str, float | str | bool]
    created_at: datetime = Field(default_factory=utc_now)


class RiskDecision(BaseModel):
    proposal_id: UUID
    status: DecisionStatus
    approved_notional: float = Field(ge=0)
    reasons: list[str]
    checked_at: datetime = Field(default_factory=utc_now)


class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID
    symbol: str
    side: Side
    quantity: float = Field(gt=0)
    requested_price: float = Field(gt=0)
    status: str = "new"
    created_at: datetime = Field(default_factory=utc_now)


class Fill(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    order_id: UUID
    symbol: str
    side: Side
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    slippage_bps: float
    filled_at: datetime = Field(default_factory=utc_now)


class Position(BaseModel):
    symbol: str
    quantity: float
    average_price: float
    last_price: float

    @computed_field
    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @computed_field
    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.last_price - self.average_price)


class PortfolioSnapshot(BaseModel):
    cash: float
    equity: float
    gross_exposure: float
    daily_pnl: float
    drawdown: float
    positions: list[Position]
    trades_today: int
    as_of: datetime = Field(default_factory=utc_now)


class LiveEvent(BaseModel):
    type: str
    payload: dict[str, object]
    created_at: datetime = Field(default_factory=utc_now)
