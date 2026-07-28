from __future__ import annotations

import argparse
import json
from typing import Any

import httpx

from . import __version__


_REQUIRED_SUMMARY_KEYS = {
    "application_version",
    "portfolio",
    "kill_switch",
    "engine_running",
    "symbols",
    "active_strategy",
    "model_registry",
    "feed_health",
    "recent",
}


def _json_object(client: httpx.Client, url: str) -> dict[str, Any]:
    response = client.get(url)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return payload


def smoke_application(
    api_url: str,
    dashboard_url: str,
    *,
    timeout_seconds: float = 10.0,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, object]:
    api = api_url.rstrip("/")
    dashboard = dashboard_url.rstrip("/")
    with httpx.Client(
        timeout=timeout_seconds,
        follow_redirects=True,
        transport=transport,
    ) as client:
        health = _json_object(client, f"{api}/health")
        openapi = _json_object(client, f"{api}/openapi.json")
        summary = _json_object(client, f"{api}/v1/dashboard/summary")
        dashboard_response = client.get(dashboard)
        dashboard_response.raise_for_status()

    openapi_version = openapi.get("info", {}).get("version")
    checks = {
        "openapi_version_matches_package": openapi_version == __version__,
        "health_version_matches_package": health.get("application_version") == __version__,
        "summary_version_matches_package": summary.get("application_version") == __version__,
        "engine_running": health.get("engine_running") is True,
        "summary_contract_complete": _REQUIRED_SUMMARY_KEYS.issubset(summary),
        "dashboard_returns_html": "text/html"
        in dashboard_response.headers.get("content-type", "").lower(),
    }
    result: dict[str, object] = {
        "passed": all(checks.values()),
        "expected_version": __version__,
        "api_url": api,
        "dashboard_url": dashboard,
        "checks": checks,
        "health": {
            "status": health.get("status"),
            "environment": health.get("environment"),
            "demo_mode": health.get("demo_mode"),
            "execution_mode": health.get("execution_mode"),
            "engine_running": health.get("engine_running"),
            "active_strategy": health.get("active_strategy"),
            "feed": health.get("feed"),
        },
        "dashboard_http_status": dashboard_response.status_code,
    }
    if not result["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Runtime smoke checks failed: {', '.join(failed)}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a running trading API and dashboard stack"
    )
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--dashboard-url", default="http://localhost:3000")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    try:
        result = smoke_application(
            arguments.api_url,
            arguments.dashboard_url,
            timeout_seconds=arguments.timeout_seconds,
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "expected_version": __version__,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
