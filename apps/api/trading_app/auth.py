from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from .config import get_settings


async def require_control_api_key(
    x_trading_api_key: str | None = Header(default=None),
) -> None:
    """Protect state-changing controls when a key is configured.

    Local development remains frictionless when `TRADING_CONTROL_API_KEY` is unset. Any
    non-local deployment should configure it and terminate TLS at the application or proxy.
    """

    expected = get_settings().trading_control_api_key
    if expected is None:
        return
    if x_trading_api_key is None or not secrets.compare_digest(
        x_trading_api_key.encode(), expected.encode()
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing control API key")
