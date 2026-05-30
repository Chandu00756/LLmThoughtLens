"""Trace API — run a full interpretability trace and stream it to the dashboard.

``POST /api/trace`` runs ``Scope(provider).trace_full(prompt)`` in a worker
thread (it does blocking network / compute), publishes ``trace_started`` and
``trace_complete`` events to the :class:`EventBus`, and returns the same JSON
payload (``TraceResult.to_payload``) to the caller.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from LLmThoughtLens.server.bus import get_bus
from LLmThoughtLens.server.config_api import build_provider, load_server_config


class TraceRequest(BaseModel):
    prompt: str
    provider: str | None = None
    run_probes: bool = False
    top_k_features: int | None = None
    attribution_threshold: float | None = None


def _run_trace(req: TraceRequest) -> dict[str, Any]:
    from LLmThoughtLens.scope import Scope

    cfg = load_server_config()
    provider_name = req.provider or cfg.active_provider
    provider = build_provider(provider_name, cfg)
    scope = Scope(
        provider,
        top_k_features=req.top_k_features or cfg.top_k_features,
        attribution_threshold=(
            req.attribution_threshold
            if req.attribution_threshold is not None
            else cfg.attribution_threshold
        ),
        blackbox_budget=cfg.blackbox_budget,
    )
    result = scope.trace_full(req.prompt, run_probes=req.run_probes)
    payload = result.to_payload()
    payload["provider"] = provider_name
    payload["model"] = provider.model_id
    return payload


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api", tags=["trace"])
    bus = get_bus()

    @router.post("/trace")
    async def post_trace(req: TraceRequest) -> dict[str, Any]:
        bus.publish("trace_started", {"prompt": req.prompt, "provider": req.provider})
        try:
            payload = await run_in_threadpool(_run_trace, req)
        except Exception as exc:  # noqa: BLE001 — report the real failure
            err = {"error": f"{type(exc).__name__}: {exc}"}
            bus.publish("trace_error", err)
            return err
        bus.publish("trace_complete", payload)
        return payload

    return router
