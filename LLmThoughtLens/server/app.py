"""FastAPI application factory for the ThoughtLens live dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from LLmThoughtLens import __version__
from LLmThoughtLens.server import (
    config_api,
    ingest_api,
    proxy,
    trace_api,
    whitebox_stream,
    xray,
)
from LLmThoughtLens.server.bus import get_bus

_STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    """Build the FastAPI app with all routers, the WebSocket, and static SPA."""
    app = FastAPI(
        title="LLmThoughtLens Live",
        version=__version__,
        description="Live interpretability dashboard + provider-compatible proxy.",
    )

    app.include_router(config_api.build_router())
    app.include_router(trace_api.build_router())
    app.include_router(whitebox_stream.build_router())
    app.include_router(xray.build_router())
    app.include_router(ingest_api.build_router())
    app.include_router(proxy.build_router())

    bus = get_bus()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        from LLmThoughtLens.providers.registry import available_providers

        return {
            "ok": True,
            "version": __version__,
            "providers": available_providers(),
            "subscribers": bus.n_subscribers,
        }

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            async for event in bus.subscribe():
                await websocket.send_json(event)
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001 — client vanished; end cleanly
            return

    # ---- Static SPA --------------------------------------------------
    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(_STATIC_DIR / "index.html"))
    else:  # pragma: no cover — only if packaging dropped the static dir

        @app.get("/")
        def index_missing() -> JSONResponse:
            return JSONResponse(
                {"error": "dashboard static assets not found", "static_dir": str(_STATIC_DIR)},
                status_code=500,
            )

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    """Launch the dashboard with uvicorn and (optionally) open the browser."""
    import uvicorn

    if open_browser:
        import threading
        import webbrowser

        url = f"http://{host}:{port}/"
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
