"""SAE API — list pretrained SAE releases, load one, and attach it to dashboard traces.

``GET  /api/sae``         known releases (no network), loaded SAEs and the active set.
``POST /api/sae/load``    load ``{release, sae_id}`` into the in-process registry
                          (Hugging Face cache only unless ``allow_download``) and,
                          by default, make it the active SAE for traces.
``POST /api/sae/detach``  clear the active set (traces go back to residual-site scoring).

Only ``release:sae_id`` specs are accepted — never server-side file paths — so
a page that can reach the local server cannot make it ``torch.load`` an
arbitrary file.  Active SAEs apply to white-box HuggingFace traces only
(:func:`saes_for_trace`); SAE features of a pretrained SAE are meaningful only
on the model it was trained on, and a mismatch is reported as a warning.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

__all__ = [
    "SAELoadRequest",
    "active_specs",
    "build_router",
    "get_sae",
    "parse_spec",
    "reset_registry",
    "saes_for_trace",
]

_LOCK = threading.Lock()
#: ``"release:sae_id"`` -> loaded SparseAutoencoder.
_REGISTRY: dict[str, Any] = {}
#: Specs attached to dashboard traces by default.
_ACTIVE: list[str] = []


class SAELoadRequest(BaseModel):
    release: str
    sae_id: str
    allow_download: bool = False
    attach: bool = True


def parse_spec(spec: str) -> tuple[str, str]:
    """``"release:sae_id"`` -> ``(release, sae_id)``; ``ValueError`` otherwise."""
    release, sep, sae_id = str(spec).partition(":")
    if not sep or not release.strip() or not sae_id.strip():
        raise ValueError(
            f"SAE spec {spec!r}: expected RELEASE:SAE_ID, e.g. "
            "gpt2-small-res-jb:blocks.6.hook_resid_pre"
        )
    return release.strip(), sae_id.strip()


def reset_registry() -> None:
    """Forget every loaded SAE and the active set (tests / memory)."""
    with _LOCK:
        _REGISTRY.clear()
        _ACTIVE.clear()


def active_specs() -> list[str]:
    with _LOCK:
        return list(_ACTIVE)


def get_sae(spec: str, *, allow_download: bool = False) -> Any:
    """The SAE for *spec*, loading it on first use (cache-only unless *allow_download*)."""
    release, sae_id = parse_spec(spec)
    key = f"{release}:{sae_id}"
    with _LOCK:
        if key in _REGISTRY:
            return _REGISTRY[key]
    from LLmThoughtLens.features.sae_loaders import from_pretrained

    try:
        sae = from_pretrained(release, sae_id, local_files_only=not allow_download)
    except FileNotFoundError as exc:
        if allow_download:
            raise
        raise FileNotFoundError(
            f"{key} is not in the local Hugging Face cache. Tick 'allow download' (pretrained "
            "SAEs are large, e.g. about 151 MB per GPT-2 layer) or run "
            f"`LLmThoughtLens sae inspect {key}` once. ({exc})"
        ) from None
    with _LOCK:
        _REGISTRY[key] = sae
    return sae


def _describe(key: str, sae: Any) -> dict[str, Any]:
    cfg = getattr(sae, "config", None)
    extra = getattr(cfg, "extra", None) or {}
    return {
        "key": key,
        "hook_name": getattr(cfg, "hook_name", None),
        "hook_layer": getattr(cfg, "hook_layer", None),
        "hook_site": getattr(cfg, "hook_site", None),
        "d_in": getattr(cfg, "input_dim", None),
        "d_sae": getattr(cfg, "dict_size", None),
        "architecture": getattr(cfg, "architecture", None),
        "hf_model": extra.get("hf_model"),
        "model_name": getattr(cfg, "model_name", None),
        "active": key in _ACTIVE,
    }


def _state() -> dict[str, Any]:
    from LLmThoughtLens.features.sae_loaders import list_pretrained

    with _LOCK:
        loaded = [_describe(k, v) for k, v in _REGISTRY.items()]
        active = list(_ACTIVE)
    return {"releases": list_pretrained(), "loaded": loaded, "active": active}


def saes_for_trace(
    provider: Any, requested: list[str] | None
) -> tuple[list[tuple[str, Any]], list[str]]:
    """``([(spec, sae)], notes)`` to attach to a trace on *provider*.

    *requested* ``None`` means "the active set"; an explicit list (even empty)
    overrides it.  SAEs need a HuggingFace model: for any other provider the
    active set is skipped with a note, and an explicit request raises
    ``ValueError``.
    """
    specs = active_specs() if requested is None else list(requested)
    if not specs:
        return [], []
    if getattr(provider, "name", "") != "huggingface":
        msg = (
            f"SAE features need a local HuggingFace model; the {provider.name!r} provider "
            + (
                "is black-box"
                if getattr(provider, "evidence_kind", "") == "black_box"
                else "has synthetic activations"
            )
        )
        if requested is not None:
            raise ValueError(msg)
        return [], [f"{msg}, so the active SAE(s) {specs} were not attached."]
    return [(spec, get_sae(spec)) for spec in specs], []


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api/sae", tags=["sae"])

    @router.get("")
    def get_state() -> dict[str, Any]:
        return _state()

    @router.post("/load")
    async def post_load(req: SAELoadRequest) -> Any:
        try:
            release, sae_id = parse_spec(f"{req.release}:{req.sae_id}")
            key = f"{release}:{sae_id}"
            sae = await run_in_threadpool(
                get_sae, key, allow_download=bool(req.allow_download)
            )
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc), "kind": "not_cached"}, status_code=404)
        except (ValueError, KeyError) as exc:
            return JSONResponse({"error": str(exc), "kind": "invalid"}, status_code=400)
        except Exception as exc:  # noqa: BLE001 — surface the real failure to the UI
            return JSONResponse(
                {"error": f"{type(exc).__name__}: {exc}", "kind": "load_failed"}, status_code=500
            )
        if req.attach:
            with _LOCK:
                _ACTIVE[:] = [key]
        return {"loaded": _describe(key, sae), **_state()}

    @router.post("/detach")
    def post_detach() -> dict[str, Any]:
        with _LOCK:
            _ACTIVE.clear()
        return _state()

    return router
