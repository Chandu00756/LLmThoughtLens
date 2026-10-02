"""Trace API — run a full interpretability trace and stream it to the dashboard.

``POST /api/trace`` runs ``Scope(provider).trace_full(prompt)`` in a worker
thread (it does blocking network / compute), publishes ``trace_started`` and
``trace_complete`` events to the :class:`EventBus`, and returns the same JSON
payload (``TraceResult.to_payload``) to the caller.

Besides the prompt / provider / probe switches the request accepts the
user-facing attribution options (defaults: :data:`LLmThoughtLens.scope.ATTRIBUTION_DEFAULTS`
— ``attribution="auto"``, ``metric="logprob"``, ``attribution_nodes=10``,
no validation): ``attribution``, ``validate`` (real ablations of the top-k
nodes), ``metric``, ``target``, ``attribution_nodes``, plus the feature
``scoring`` and ``sae`` (``["release:sae_id", ...]``; omitted = the SAEs made
active through ``/api/sae/load``, HuggingFace traces only).  The payload's
``attribution`` block states the edge semantics and, when validated, the
faithfulness numbers (Spearman, Pearson, n, sign agreement + caveat).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from LLmThoughtLens.server.bus import get_bus
from LLmThoughtLens.server.config_api import build_provider, load_server_config


class TraceRequest(BaseModel):
    prompt: str
    provider: str | None = None
    run_probes: bool = False
    top_k_features: int | None = None
    attribution_threshold: float | None = None
    attribution: Literal["auto", "gradient", "activation_flow"] | None = None
    validate_k: int | None = Field(default=None, ge=0, le=100, alias="validate")
    scoring: Literal["centered", "l2"] | None = None
    metric: Literal["logprob", "logit", "logit_diff"] | None = None
    attribution_nodes: int | None = Field(default=None, ge=0, le=200)
    target: str | None = None
    sae: list[str] | None = None

    model_config = {"populate_by_name": True}


def _run_trace(req: TraceRequest) -> dict[str, Any]:
    from LLmThoughtLens.scope import Scope
    from LLmThoughtLens.server.sae_api import saes_for_trace

    cfg = load_server_config()
    provider_name = req.provider or cfg.active_provider
    provider = build_provider(provider_name, cfg)
    scope_kw: dict[str, Any] = {}
    if req.scoring is not None:
        scope_kw["scoring"] = req.scoring
    scope = Scope(
        provider,
        top_k_features=req.top_k_features or cfg.top_k_features,
        attribution_threshold=(
            req.attribution_threshold
            if req.attribution_threshold is not None
            else cfg.attribution_threshold
        ),
        blackbox_budget=cfg.blackbox_budget,
        **scope_kw,
    )
    saes, notes = saes_for_trace(provider, req.sae)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if saes:
            from LLmThoughtLens.scope import _warn_on_model_mismatch

            for _, sae in saes:
                _warn_on_model_mismatch(sae, provider)
            scope.attach_saes([sae for _, sae in saes])
        result = scope.trace_full(
            req.prompt,
            run_probes=req.run_probes,
            attribution=req.attribution,
            validate=req.validate_k,
            metric=req.metric,
            attribution_nodes=req.attribution_nodes,
            target=req.target or None,
        )
    payload = result.to_payload()
    payload["provider"] = provider_name
    payload["model"] = provider.model_id
    payload["notes"] = notes + [
        str(w.message) for w in caught if issubclass(w.category, (UserWarning, RuntimeWarning))
    ]
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
