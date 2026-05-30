"""LLmThoughtLens live server — FastAPI dashboard + provider-compatible proxy.

Import the heavy FastAPI app lazily so ``import LLmThoughtLens`` never requires
the ``server`` extra.  Use :func:`create_app` or :func:`run_server`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(**kwargs: Any) -> FastAPI:
    """Build and return the FastAPI application (requires the ``server`` extra)."""
    from LLmThoughtLens.server.app import create_app as _create_app

    return _create_app(**kwargs)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    **kwargs: Any,
) -> None:
    """Launch the dashboard via uvicorn (requires the ``server`` extra)."""
    from LLmThoughtLens.server.app import run_server as _run_server

    _run_server(host=host, port=port, open_browser=open_browser, **kwargs)


__all__ = ["create_app", "run_server"]
