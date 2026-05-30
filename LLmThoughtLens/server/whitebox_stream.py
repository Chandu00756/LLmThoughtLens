"""White-box "thinking stream" — live per-layer activations from a local HF model.

Runs a real token-by-token greedy generation with ``output_hidden_states=True``
and publishes, for every generated token, the per-layer L2 norm of the residual
stream at the last position (the model's "thinking" as it unfolds).  After
generation it runs a full :class:`Scope` trace and publishes the complete
payload (features, attribution graph, residual-stream view data).

Everything here is real model computation — there is no synthetic data path.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from LLmThoughtLens.server.bus import get_bus


class WhiteboxRequest(BaseModel):
    model_name: str = "gpt2"
    prompt: str = "The capital of France is"
    max_new_tokens: int = 20
    device: str = "auto"


def _stream_whitebox(req: WhiteboxRequest) -> dict[str, Any]:
    """Blocking generation loop with real activation capture. Publishes events."""
    import torch

    from LLmThoughtLens.providers.huggingface_provider import HuggingFaceProvider
    from LLmThoughtLens.scope import Scope

    bus = get_bus()
    provider = HuggingFaceProvider(
        model_name=req.model_name, device=req.device, capture_internals=True
    )
    provider._load()  # noqa: SLF001 — within-package use of the lazy loader
    model = provider._model  # noqa: SLF001
    tokenizer = provider._tokenizer  # noqa: SLF001
    device = provider._device  # noqa: SLF001

    enc = tokenizer(req.prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    generated: list[int] = []

    bus.publish(
        "whitebox_started",
        {"model": req.model_name, "device": str(device), "prompt": req.prompt},
    )

    eos_id = tokenizer.eos_token_id
    with torch.no_grad():
        for step in range(int(req.max_new_tokens)):
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            hidden_states = out.hidden_states[1:]  # drop embedding layer
            # Per-layer L2 norm of the LAST position = "what the model is building".
            layer_norms = [float(hs[0, -1].to(torch.float32).norm().cpu()) for hs in hidden_states]
            logits = out.logits[0, -1].to(torch.float32)
            probs = torch.softmax(logits, dim=-1)
            top_p, top_i = torch.topk(probs, k=min(5, probs.shape[0]))
            next_id = int(top_i[0])
            next_tok = tokenizer.decode([next_id])
            top_tokens = [
                [tokenizer.decode([int(i)]), float(p)]
                for i, p in zip(top_i.tolist(), top_p.tolist(), strict=False)
            ]

            bus.publish(
                "whitebox_step",
                {
                    "step": step,
                    "token": next_tok,
                    "layer_norms": layer_norms,
                    "n_layers": len(layer_norms),
                    "top_tokens": top_tokens,
                },
            )
            generated.append(next_id)
            input_ids = torch.cat([input_ids, top_i[:1].view(1, 1)], dim=1)
            if eos_id is not None and next_id == eos_id:
                break

    completion = tokenizer.decode(generated) if generated else ""

    # Full attribution trace on the original prompt (real activations + SAE-less L2).
    scope = Scope(provider)
    result = scope.trace_full(req.prompt, run_probes=False)
    payload = result.to_payload()
    payload["completion"] = completion
    payload["provider"] = "huggingface"
    payload["model"] = provider.model_id
    bus.publish("trace_complete", payload)
    bus.publish("whitebox_complete", {"completion": completion, "n_steps": len(generated)})
    return {"completion": completion, "n_steps": len(generated)}


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api/whitebox", tags=["whitebox"])

    @router.post("/stream")
    async def post_stream(req: WhiteboxRequest) -> dict[str, Any]:
        # Kick off in a worker thread; progress flows over the WebSocket.
        async def _runner() -> None:
            try:
                await run_in_threadpool(_stream_whitebox, req)
            except Exception as exc:  # noqa: BLE001
                get_bus().publish("whitebox_error", {"error": f"{type(exc).__name__}: {exc}"})

        import asyncio

        asyncio.create_task(_runner())
        return {"started": True, "model": req.model_name}

    return router
