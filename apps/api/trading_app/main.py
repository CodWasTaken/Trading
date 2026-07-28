from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import __version__
from .auth import require_control_api_key
from .broker import AlpacaPaperBroker, InternalPaperBroker
from .config import Settings, get_settings
from .engine import TradingEngine
from .model_registry import ModelRegistry
from .model_strategy import ChampionModelStrategy
from .persistence import SQLiteEventSink
from .portfolio import Portfolio
from .providers import AlpacaWebSocketFeed, DemoFeed
from .risk import RiskEngine
from .sec import SecEdgarClient
from .store import EventStore
from .strategy import ExplainableCatalystStrategy


class ControlRequest(BaseModel):
    enabled: bool


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.event_sink = SQLiteEventSink(settings.trading_database_path)
        self.store = EventStore(sink=self.event_sink)
        self.portfolio = Portfolio(settings.trading_starting_cash)
        self.risk = RiskEngine(settings)
        self.broker = (
            AlpacaPaperBroker(settings)
            if settings.trading_execution_mode == "alpaca-paper"
            else InternalPaperBroker()
        )
        registry = ModelRegistry(settings.trading_model_registry_path)
        champion = (
            registry.load_champion()
            if settings.trading_strategy_mode == "champion"
            else None
        )
        strategy = (
            ChampionModelStrategy(champion)
            if champion is not None
            else ExplainableCatalystStrategy()
        )
        self.model_registry = registry
        self.active_strategy = "champion" if champion is not None else "explainable"
        self.engine = TradingEngine(
            store=self.store,
            portfolio=self.portfolio,
            risk=self.risk,
            strategy=strategy,
            broker=self.broker,
        )
        self.engine_task: asyncio.Task[None] | None = None


settings = get_settings()
state = AppState(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if state.settings.trading_demo_mode:
        feed = DemoFeed(
            state.settings.trading_symbols,
            state.settings.trading_decision_interval_seconds,
        )
        state.engine_task = asyncio.create_task(state.engine.run_demo(feed.stream()))
    else:
        feed = AlpacaWebSocketFeed(state.settings)
        state.engine_task = asyncio.create_task(
            state.engine.run_streams(feed.stream_quotes(), feed.stream_news())
        )
    try:
        yield
    finally:
        await state.engine.stop()
        if state.engine_task:
            state.engine_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await state.engine_task
        if isinstance(state.broker, AlpacaPaperBroker):
            await state.broker.close()
        state.event_sink.close()


app = FastAPI(
    title="Trading Platform API",
    version=__version__,
    description="Risk-first AI-assisted paper trading API",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.trading_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_state() -> AppState:
    return state


StateDependency = Annotated[AppState, Depends(get_state)]


@app.get("/health")
async def health(current: StateDependency) -> dict[str, object]:
    feed_health = await current.store.health(
        current.settings.trading_max_data_age_seconds
    )
    return {
        "status": "ok" if feed_health["healthy"] else "degraded",
        "version": __version__,
        "environment": current.settings.trading_env,
        "demo_mode": current.settings.trading_demo_mode,
        "execution_mode": current.settings.trading_execution_mode,
        "engine_running": current.engine.running,
        "kill_switch": current.risk.kill_switch,
        "active_strategy": current.active_strategy,
        "feed": feed_health,
    }


@app.get("/v1/dashboard/summary")
async def dashboard_summary(current: StateDependency) -> dict[str, object]:
    portfolio = current.portfolio.snapshot()
    ledger = await current.store.snapshot()
    feed_health = await current.store.health(
        current.settings.trading_max_data_age_seconds
    )
    return {
        "version": __version__,
        "portfolio": portfolio.model_dump(mode="json"),
        "kill_switch": current.risk.kill_switch,
        "engine_running": current.engine.running,
        "symbols": current.settings.trading_symbols,
        "active_strategy": current.active_strategy,
        "model_registry": current.model_registry.summary(),
        "feed_health": feed_health,
        "recent": ledger,
    }


@app.get("/v1/portfolio")
async def portfolio(current: StateDependency) -> dict[str, object]:
    return current.portfolio.snapshot().model_dump(mode="json")


@app.get("/v1/events")
async def events(current: StateDependency) -> dict[str, object]:
    return await current.store.snapshot()


@app.get("/v1/models")
async def models(current: StateDependency) -> dict[str, object]:
    return current.model_registry.summary()


@app.get("/v1/reconciliation")
async def reconciliation(current: StateDependency) -> dict[str, object]:
    if not isinstance(current.broker, AlpacaPaperBroker):
        return {
            "mode": "internal-paper",
            "status": "not_applicable",
            "differences": [],
        }
    broker_positions = await current.broker.get_positions()
    internal = {
        position.symbol: position.quantity
        for position in current.portfolio.snapshot().positions
    }
    external = {position.symbol: position.quantity for position in broker_positions}
    differences = [
        {
            "symbol": symbol,
            "internal_quantity": internal.get(symbol, 0.0),
            "broker_quantity": external.get(symbol, 0.0),
            "difference": internal.get(symbol, 0.0) - external.get(symbol, 0.0),
        }
        for symbol in sorted(set(internal) | set(external))
        if abs(internal.get(symbol, 0.0) - external.get(symbol, 0.0)) > 1e-8
    ]
    return {
        "mode": "alpaca-paper",
        "status": "matched" if not differences else "mismatch",
        "differences": differences,
    }


@app.get("/v1/sec/{cik}/filings")
async def sec_filings(
    cik: str,
    current: StateDependency,
    forms: Annotated[str, Query(description="Comma-separated SEC form types")] = "10-K,10-Q,8-K",
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> dict[str, object]:
    try:
        user_agent = current.settings.require_sec_user_agent()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    client = SecEdgarClient(
        user_agent,
        base_url=current.settings.sec_data_base_url,
    )
    try:
        requested_forms = {part.strip() for part in forms.split(",") if part.strip()}
        filings = await client.recent_filings(cik, forms=requested_forms, limit=limit)
    finally:
        await client.close()
    return {
        "cik": str(cik).strip().removeprefix("CIK").zfill(10),
        "filings": [
            {**filing.model_dump(mode="json"), "filing_url": filing.filing_url}
            for filing in filings
        ],
    }


@app.post(
    "/v1/control/kill-switch",
    dependencies=[Depends(require_control_api_key)],
)
async def kill_switch(
    request: ControlRequest, current: StateDependency
) -> dict[str, bool]:
    current.risk.set_kill_switch(request.enabled)
    return {"kill_switch": current.risk.kill_switch}


@app.post(
    "/v1/control/pause",
    dependencies=[Depends(require_control_api_key)],
)
async def pause(current: StateDependency) -> dict[str, bool]:
    await current.engine.stop()
    return {"engine_running": current.engine.running}


@app.post(
    "/v1/control/resume",
    dependencies=[Depends(require_control_api_key)],
)
async def resume(current: StateDependency) -> dict[str, bool]:
    if current.engine.running:
        return {"engine_running": True}
    if not current.settings.trading_demo_mode:
        raise HTTPException(
            status_code=409, detail="Restart the service to resume external streams"
        )
    feed = DemoFeed(
        current.settings.trading_symbols,
        current.settings.trading_decision_interval_seconds,
    )
    current.engine_task = asyncio.create_task(current.engine.run_demo(feed.stream()))
    return {"engine_running": True}


@app.websocket("/v1/live/events")
async def live_events(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        async for event in state.store.subscribe():
            await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        return
