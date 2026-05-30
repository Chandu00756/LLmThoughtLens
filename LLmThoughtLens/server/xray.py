"""Live LLM X-ray endpoint — thin HTTP wrapper over :mod:`LLmThoughtLens.xray_core`.

The real logit-lens computation lives in ``xray_core`` (no web dependency) so
the exact same loop powers both this endpoint and ``sdk.attach``.  This module
just loads a model (by HF id *or local weights path*) and streams events to the
:class:`EventBus`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from LLmThoughtLens.server.bus import get_bus
from LLmThoughtLens.xray_core import resolve_final_norm, run_xray_loop

# Re-exported for backwards-compatible imports/tests.
_resolve_final_norm = resolve_final_norm

__all__ = ["build_router", "XrayRequest", "_resolve_final_norm"]


class XrayRequest(BaseModel):
    # model_name may be a HuggingFace hub id OR a local weights folder / path.
    model_name: str = "gpt2"
    prompt: str = "The capital of France is"
    max_new_tokens: int = 12
    device: str = "auto"


def _stream_xray(req: XrayRequest) -> dict[str, Any]:
    """Load the model (hub id or local path) and run the shared X-ray loop."""
    from LLmThoughtLens.providers.huggingface_provider import HuggingFaceProvider

    bus = get_bus()
    provider = HuggingFaceProvider(
        model_name=req.model_name, device=req.device, capture_internals=True
    )
    provider._load()  # noqa: SLF001 — lazy loader; accepts hub ids and local paths
    return run_xray_loop(
        model=provider._model,  # noqa: SLF001
        tokenizer=provider._tokenizer,  # noqa: SLF001
        device=provider._device,  # noqa: SLF001
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        emit=bus.publish,
        model_label=req.model_name,
    )


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api/xray", tags=["xray"])

    @router.post("/stream")
    async def post_xray(req: XrayRequest) -> dict[str, Any]:
        async def _runner() -> None:
            try:
                await run_in_threadpool(_stream_xray, req)
            except Exception as exc:  # noqa: BLE001
                get_bus().publish("xray_error", {"error": f"{type(exc).__name__}: {exc}"})

        import asyncio

        asyncio.create_task(_runner())
        return {"started": True, "model": req.model_name}

    return router
