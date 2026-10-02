"""Steering API — baseline vs steered completion on a local white-box model.

``POST /api/steer`` builds a steering vector — from contrast prompts
(``positive`` / ``negative``, mean activation difference at ``layer``), or from
an SAE feature (``sae`` = ``"release:sae_id"`` + ``feature``) — and runs
:func:`~LLmThoughtLens.features.steering.steer_generate`.  The response is
``SteeringResult.to_dict()`` (baseline and steered text / tokens / logprobs,
teacher-forced ``kl_per_step``, promoted / suppressed tokens, evidence labels
``evidence_kind`` / ``method`` / ``effect_semantics`` / ``note``) plus the
vector's metadata and the model id.  ``steer_started`` / ``steer_complete`` /
``steer_error`` events go to the :class:`EventBus`.

Steering edits the residual stream, so it needs the HuggingFace provider (or a
``model_name`` = HF id / local weights path, like the X-ray).  Any other
provider returns HTTP 400 with ``kind="steering_unavailable"`` and a message
saying why (black-box API, or the mock's synthetic activations).
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from LLmThoughtLens.server.bus import get_bus
from LLmThoughtLens.server.config_api import build_provider, load_server_config

__all__ = ["SteerRequest", "build_router", "run_steer"]


class SteerRequest(BaseModel):
    prompt: str
    provider: str | None = None
    model_name: str | None = None
    device: str = "auto"
    positive: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)
    sae: str | None = None
    feature: int | None = Field(default=None, ge=0)
    layer: int | None = None
    site: Literal["resid_post", "resid_pre"] = "resid_post"
    position: Literal["last", "mean"] = "last"
    positions: Literal["all", "last", "prompt", "generated"] = "all"
    coeff: float = 1.0
    normalize: bool = False
    max_new_tokens: int = Field(default=20, ge=1, le=256)
    temperature: float = Field(default=0.0, ge=0.0)
    seed: int | None = None


class _BadRequest(ValueError):
    pass


def _provider_for(req: SteerRequest) -> Any:
    if req.model_name:
        from LLmThoughtLens.providers.huggingface_provider import HuggingFaceProvider

        return HuggingFaceProvider(model_name=req.model_name, device=req.device)
    cfg = load_server_config()
    return build_provider(req.provider or cfg.active_provider, cfg)


def run_steer(req: SteerRequest) -> dict[str, Any]:
    """Build the vector and run steered generation (blocking; call from a thread)."""
    from LLmThoughtLens.features.steering import SteeringVector, as_hooked, steer_generate

    provider = _provider_for(req)
    hooked = as_hooked(provider)  # SteeringUnavailableError for mock / black-box
    if req.sae:
        if req.feature is None:
            raise _BadRequest("an SAE steering vector needs 'feature' (the dictionary index)")
        from LLmThoughtLens.server.sae_api import get_sae

        vector = SteeringVector.from_sae_feature(
            get_sae(req.sae),
            int(req.feature),
            layer=req.layer,
            coeff=req.coeff,
            normalize=True,  # unit decoder direction: coeff is in residual units
            positions=req.positions,
        )
    else:
        positive = [p for p in req.positive if p.strip()]
        negative = [p for p in req.negative if p.strip()]
        if not positive or not negative:
            raise _BadRequest(
                "contrast steering needs at least one positive and one negative prompt "
                "(or pass 'sae' + 'feature')"
            )
        if req.layer is None:
            raise _BadRequest("contrast steering needs 'layer'")
        vector = SteeringVector.from_contrast(
            hooked,
            positive,
            negative,
            int(req.layer),
            req.site,
            req.position,
            coeff=req.coeff,
            normalize=req.normalize,
            positions=req.positions,
        )
    result = steer_generate(
        hooked,
        req.prompt,
        vector,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        seed=req.seed,
    )
    payload = result.to_dict()
    payload["vector"] = vector.metadata()
    payload["model"] = provider.model_id
    payload["n_layers"] = int(hooked.n_layers)
    return payload


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api", tags=["steering"])
    bus = get_bus()

    @router.post("/steer")
    async def post_steer(req: SteerRequest) -> Any:
        from LLmThoughtLens.features.steering import (
            SteeringMismatchError,
            SteeringUnavailableError,
        )

        bus.publish("steer_started", {"prompt": req.prompt, "layer": req.layer})
        try:
            payload = await run_in_threadpool(run_steer, req)
        except SteeringUnavailableError as exc:
            err = {"error": str(exc), "kind": "steering_unavailable"}
            bus.publish("steer_error", err)
            return JSONResponse(err, status_code=400)
        except (SteeringMismatchError, _BadRequest, ValueError, FileNotFoundError) as exc:
            err = {"error": str(exc), "kind": "invalid"}
            bus.publish("steer_error", err)
            return JSONResponse(err, status_code=400)
        except Exception as exc:  # noqa: BLE001 — report the real failure
            err = {"error": f"{type(exc).__name__}: {exc}", "kind": "failed"}
            bus.publish("steer_error", err)
            return JSONResponse(err, status_code=500)
        bus.publish("steer_complete", payload)
        return payload

    return router
