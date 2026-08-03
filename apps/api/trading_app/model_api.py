from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .auth import require_control_api_key
from .governed_models import GovernedModelRegistry
from .model_strategy import ChampionModelStrategy


class ModelSelectionRequest(BaseModel):
    version: str
    reason: str = Field(min_length=3, max_length=500)


def build_model_router(get_state: Callable[[], Any]) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models/catalog")
    async def model_catalogue(current: Any = Depends(get_state)) -> dict[str, object]:
        registry = GovernedModelRegistry(current.settings.trading_model_registry_path)
        return registry.catalogue(
            engine_running=current.engine.running,
            kill_switch=current.risk.kill_switch,
            runtime_feature_names=ChampionModelStrategy.SUPPORTED_FEATURES,
            universe_manifest_sha256=current.universe.manifest_sha256,
        )

    @router.get("/v1/models/compare")
    async def compare_models(
        versions: str = Query(min_length=1),
        current: Any = Depends(get_state),
    ) -> dict[str, object]:
        requested = [item.strip() for item in versions.split(",") if item.strip()]
        registry = GovernedModelRegistry(current.settings.trading_model_registry_path)
        try:
            return registry.compare_versions(requested)
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.post(
        "/v1/control/models/select",
        dependencies=[Depends(require_control_api_key)],
    )
    async def select_model(
        request: ModelSelectionRequest,
        current: Any = Depends(get_state),
    ) -> dict[str, object]:
        registry = GovernedModelRegistry(current.settings.trading_model_registry_path)
        try:
            record = registry.select_active_paper(
                request.version,
                reason=request.reason,
                engine_running=current.engine.running,
                kill_switch=current.risk.kill_switch,
                runtime_feature_names=ChampionModelStrategy.SUPPORTED_FEATURES,
                universe_manifest_sha256=current.universe.manifest_sha256,
            )
            model = registry.load_any(record.version)
            current.engine.strategy = ChampionModelStrategy(model)
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        current.model_registry = registry
        current.active_strategy = f"active-paper:{record.version}"
        return {
            "selected_version": record.version,
            "active_strategy": current.active_strategy,
            "execution_scope": "paper_only",
            "live_money_authorized": False,
            "restart_required": False,
        }

    return router
