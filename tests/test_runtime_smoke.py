from __future__ import annotations

import httpx
import pytest

from trading_app import __version__
from trading_app.main import app
from trading_app.smoke_cli import smoke_application


def _summary(version: str) -> dict[str, object]:
    return {
        "application_version": version,
        "portfolio": {},
        "kill_switch": False,
        "engine_running": True,
        "symbols": ["AAPL"],
        "active_strategy": "explainable",
        "model_registry": {},
        "feed_health": {"healthy": True},
        "recent": {},
    }


def test_fastapi_metadata_uses_package_version() -> None:
    assert app.version == __version__


def test_runtime_smoke_validates_api_and_dashboard_contracts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.test" and request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "application_version": __version__,
                    "environment": "test",
                    "demo_mode": True,
                    "execution_mode": "internal-paper",
                    "engine_running": True,
                    "active_strategy": "explainable",
                    "feed": {"healthy": True},
                },
            )
        if request.url.host == "api.test" and request.url.path == "/openapi.json":
            return httpx.Response(200, json={"info": {"version": __version__}})
        if request.url.host == "api.test" and request.url.path == "/v1/dashboard/summary":
            return httpx.Response(200, json=_summary(__version__))
        if request.url.host == "dashboard.test" and request.url.path == "/":
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text="<html><body>Trading dashboard</body></html>",
            )
        return httpx.Response(404)

    report = smoke_application(
        "https://api.test",
        "https://dashboard.test",
        transport=httpx.MockTransport(handler),
    )

    assert report["passed"] is True
    assert report["dashboard_http_status"] == 200


def test_runtime_smoke_rejects_version_drift() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"application_version": "0.4.0", "engine_running": True})
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json={"info": {"version": "0.4.0"}})
        if request.url.path == "/v1/dashboard/summary":
            return httpx.Response(200, json=_summary("0.4.0"))
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<html></html>")

    with pytest.raises(RuntimeError, match="version"):
        smoke_application(
            "https://api.test",
            "https://dashboard.test",
            transport=httpx.MockTransport(handler),
        )
