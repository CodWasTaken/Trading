from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field


class BorrowState(StrEnum):
    EASY_TO_BORROW = "easy_to_borrow"
    HARD_TO_BORROW = "hard_to_borrow"
    UNAVAILABLE = "unavailable"
    RECALLED = "recalled"


class BorrowStatus(BaseModel):
    symbol: str
    state: BorrowState
    available_quantity: float | None = Field(default=None, ge=0)
    annual_rate: float | None = Field(default=None, ge=0)
    checked_at: datetime

    def permits(self, quantity: float) -> bool:
        if self.state != BorrowState.EASY_TO_BORROW:
            return False
        return self.available_quantity is None or quantity <= self.available_quantity + 1e-9


class BorrowProvider(Protocol):
    def status(self, symbol: str) -> BorrowStatus | None: ...


class StaticBorrowProvider:
    """Deterministic paper/replay borrow inventory; absent symbols fail closed."""

    def __init__(self, statuses: Mapping[str, BorrowStatus] | None = None) -> None:
        self._statuses = {
            symbol.upper(): status for symbol, status in (statuses or {}).items()
        }

    def status(self, symbol: str) -> BorrowStatus | None:
        return self._statuses.get(symbol.upper())


class ConfiguredBorrowProvider:
    """Paper-only ETB list. It never infers availability for an unlisted symbol."""

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        clock: Callable[[], datetime],
        annual_rate: float,
    ) -> None:
        self._symbols = {symbol.upper() for symbol in symbols}
        self._clock = clock
        self._annual_rate = annual_rate

    def status(self, symbol: str) -> BorrowStatus | None:
        if symbol.upper() not in self._symbols:
            return None
        return BorrowStatus(
            symbol=symbol.upper(),
            state=BorrowState.EASY_TO_BORROW,
            annual_rate=self._annual_rate,
            checked_at=self._clock(),
        )


class BorrowRecall(BaseModel):
    symbol: str
    recalled_quantity: float | None = Field(default=None, gt=0)
    effective_at: datetime
    reason: str = "lender_recall"


class ForcedCoverInstruction(BaseModel):
    symbol: str
    quantity: float = Field(gt=0)
    effective_at: datetime
    reason: str


def forced_cover_for(
    recall: BorrowRecall,
    signed_position_quantity: float,
) -> ForcedCoverInstruction | None:
    if signed_position_quantity >= 0:
        return None
    quantity = abs(signed_position_quantity)
    if recall.recalled_quantity is not None:
        quantity = min(quantity, recall.recalled_quantity)
    return ForcedCoverInstruction(
        symbol=recall.symbol,
        quantity=quantity,
        effective_at=recall.effective_at,
        reason=recall.reason,
    )
