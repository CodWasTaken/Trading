from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PromotionGateConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    name: str = "strict-paper-promotion-v1"

    minimum_calibration_folds: int = Field(default=8, ge=8)
    minimum_calibration_observations: int = Field(default=1000, ge=1)
    minimum_calibration_net_return: float = 0.0
    minimum_calibration_excess_return: float = 0.0
    minimum_calibration_sharpe: float = 0.50
    maximum_calibration_drawdown: float = Field(default=0.15, gt=0, le=1)

    maximum_holdout_finalists: int = Field(default=3, ge=1, le=3)
    minimum_holdout_observations: int = Field(default=300, ge=1)
    minimum_holdout_net_return: float = 0.0
    minimum_holdout_excess_return: float = 0.0
    minimum_holdout_net_return_lower_bound: float = 0.0
    minimum_holdout_excess_return_lower_bound: float = 0.0
    maximum_holdout_drawdown: float = Field(default=0.15, gt=0, le=1)

    maximum_symbol_pnl_contribution: float = Field(default=0.20, gt=0, le=1)
    maximum_sector_pnl_contribution: float = Field(default=0.35, gt=0, le=1)

    maximum_unborrowable_short_orders: int = Field(default=0, ge=0)
    maximum_gross_short_exposure: float = Field(default=0.30, ge=0, le=1)
    maximum_single_short_position: float = Field(default=0.03, ge=0, le=1)

    minimum_news_sharpe_delta: float | None = None
    minimum_holdout_sharpe: float | None = None

    @model_validator(mode="after")
    def validate_schema(self) -> PromotionGateConfig:
        if self.schema_version != 1:
            raise ValueError("Unsupported promotion gate schema_version")
        return self

    @property
    def manifest_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def load_promotion_gates(path: str | Path | None) -> PromotionGateConfig:
    if path is None:
        return PromotionGateConfig()
    source = Path(path)
    if not source.is_absolute() and not source.is_file():
        repository_source = Path(__file__).resolve().parents[3] / source
        if repository_source.is_file():
            source = repository_source
    return PromotionGateConfig.model_validate_json(source.read_text(encoding="utf-8"))
