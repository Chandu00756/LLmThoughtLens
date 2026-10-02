"""xray_core — the logit-lens forward loop, with no web dependency.

Shared by the dashboard server (``server/xray.py``) and the SDK
(``sdk.attach``) so the same real computation drives both an HTTP endpoint and
an in-process model the user already has.  Depends only on ``torch`` +
``transformers`` (the ``huggingface`` extra), never on FastAPI.

The residual stream comes from :class:`~LLmThoughtLens.models.hooked.HookedModel`
(forward hooks on every block), so every layer — including the last — is the
true pre-final-norm residual, and the logit lens applies the final norm,
unembedding and any logit soft-capping exactly once.  (Reading HF's
``output_hidden_states`` instead would apply the final norm *twice* on the
last layer, because HF already normalises that entry.)  Models whose blocks
cannot be located fall back to ``output_hidden_states``; there the last entry
is treated as already normalised (the HF convention).

The transport is decoupled via an ``emit(kind, data)`` callback: the server
passes ``bus.publish``; the SDK passes a function that POSTs to a dashboard.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from LLmThoughtLens.models.families import resolve_final_norm
from LLmThoughtLens.models.hooked import infer_device

# The return value is ignored; ``Any`` lets both ``bus.publish`` (returns the
# event dict) and the SDK's push wrapper (returns None) satisfy this type.
EmitFn = Callable[[str, dict[str, Any]], Any]

__all__ = ["EmitFn", "infer_device", "resolve_final_norm", "run_xray_loop"]

#: Head-averaged attention is only shipped for prompts up to this length.
_MAX_ATTENTION_SEQ = 40


def _top3(tokenizer: Any, logits: Any) -> list[list[Any]]:
    import torch

    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    p, idx = torch.topk(probs, k=min(3, probs.shape[-1]))
    return [
        [tokenizer.decode([int(i)]), float(pp)]
        for i, pp in zip(idx.tolist(), p.tolist(), strict=False)
    ]


def run_xray_loop(
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt: str,
    max_new_tokens: int,
    emit: EmitFn,
    model_label: str = "model",
    *,
    hooked: Any = None,
) -> dict[str, Any]:
    """Generate token-by-token, emitting real logit-lens / activation events.

    For every generated token this emits an ``xray_step`` with:

    * ``logit_lens`` — each layer's last-position residual stream (``resid_post``)
      projected through the model's real final-norm + unembedding (top-3
      tokens).  Reading this column bottom→top shows the prediction forming
      inside the network; the last layer equals the model's own prediction.
    * ``grid`` — an ``(n_layers x n_tokens)`` residual-stream L2-norm map.
    * ``attention`` — the last layer's head-averaged attention (short prompts,
      eager attention only; ``[]`` otherwise).

    Parameters
    ----------
    model, tokenizer, device:
        The HF model to inspect (any supported family), its tokenizer and the
        input device.
    hooked:
        Optional ready-made :class:`HookedModel` over *model* (avoids
        re-resolving the architecture).

    Returns ``{"completion", "n_steps"}``.  All numbers come from a real forward
    pass; there is no synthetic path.
    """
    from LLmThoughtLens.models.families import UnsupportedArchitectureError
    from LLmThoughtLens.models.hooked import HookedModel

    hm = hooked
    if hm is None:
        try:
            hm = HookedModel(model, tokenizer, device=device)
        except UnsupportedArchitectureError:
            hm = None
    if hm is None:
        return _run_hidden_states_loop(
            model, tokenizer, device, prompt, max_new_tokens, emit, model_label
        )
    return _run_hooked_loop(hm, prompt, max_new_tokens, emit, model_label)


def _run_hooked_loop(
    hm: Any, prompt: str, max_new_tokens: int, emit: EmitFn, model_label: str
) -> dict[str, Any]:
    import torch

    tokenizer = hm.tokenizer
    emit(
        "xray_started",
        {
            "model": model_label,
            "device": str(hm.device),
            "prompt": prompt,
            "has_logit_lens": hm.unembed is not None,
            "has_final_norm": hm.final_norm is not None,
        },
    )

    ids = hm.tokenize(prompt).input_ids
    eos_id = getattr(tokenizer, "eos_token_id", None)
    generated: list[int] = []
    with torch.no_grad():
        for step in range(int(max_new_tokens)):
            res = hm.forward(ids, capture=True, capture_attentions=True)
            resid = res.resid_post  # (L, T, D) true residual, no final norm
            n_layers, seq = int(resid.shape[0]), int(resid.shape[1])

            logit_lens: list[dict[str, Any]] = []
            if hm.unembed is not None:
                lens_logits = hm.logit_lens(resid[:, -1, :])  # (L, V), final norm once
                logit_lens = [
                    {"layer": li, "top": _top3(tokenizer, lens_logits[li])}
                    for li in range(n_layers)
                ]

            grid = resid.to(torch.float32).norm(dim=-1).cpu().tolist()

            attention: list[list[float]] = []
            if res.attentions is not None and seq <= _MAX_ATTENTION_SEQ:
                attention = res.attentions[-1].to(torch.float32).mean(0).cpu().tolist()

            next_id = int(torch.argmax(res.logits[0, -1].to(torch.float32)))
            emit(
                "xray_step",
                {
                    "step": step,
                    "token": tokenizer.decode([next_id]),
                    "tokens": list(res.tokens),
                    "n_layers": n_layers,
                    "logit_lens": logit_lens,
                    "grid": grid,
                    "attention": attention,
                },
            )

            generated.append(next_id)
            nxt = torch.tensor([[next_id]], dtype=ids.dtype, device=ids.device)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and next_id == eos_id:
                break

    completion = tokenizer.decode(generated) if generated else ""
    emit("xray_complete", {"completion": completion, "n_steps": len(generated)})
    return {"completion": completion, "n_steps": len(generated)}


def _run_hidden_states_loop(
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt: str,
    max_new_tokens: int,
    emit: EmitFn,
    model_label: str,
) -> dict[str, Any]:
    """Fallback for models whose transformer blocks cannot be located.

    Reads ``output_hidden_states``; following the HF convention the last entry
    is already final-normed, so the final norm is applied to every layer
    except the last (no double norm).
    """
    import torch

    get_out = getattr(model, "get_output_embeddings", None)
    unembed = get_out() if callable(get_out) else None
    final_norm = resolve_final_norm(model)

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    generated: list[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)

    emit(
        "xray_started",
        {
            "model": model_label,
            "device": str(device),
            "prompt": prompt,
            "has_logit_lens": unembed is not None,
            "has_final_norm": final_norm is not None,
        },
    )

    with torch.no_grad():
        for step in range(int(max_new_tokens)):
            out = model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
            )
            hidden = out.hidden_states[1:]
            n_layers = len(hidden)
            seq = input_ids.shape[1]
            cur_tokens = [tokenizer.decode([int(t)]) for t in input_ids[0].tolist()]

            logit_lens: list[dict[str, Any]] = []
            if unembed is not None:
                for li in range(n_layers):
                    h = hidden[li][0, -1]
                    if final_norm is not None and li < n_layers - 1:
                        h = final_norm(h)
                    logit_lens.append({"layer": li, "top": _top3(tokenizer, unembed(h))})

            grid = [
                [float(hidden[li][0, t].to(torch.float32).norm().cpu()) for t in range(seq)]
                for li in range(n_layers)
            ]

            # ``out.attentions`` may be None OR an empty tuple when the model's
            # attention backend (e.g. SDPA) doesn't return weights even when
            # asked — guard against both so attach() works on any loaded model.
            attention: list[list[float]] = []
            if out.attentions and seq <= _MAX_ATTENTION_SEQ:
                attention = out.attentions[-1][0].to(torch.float32).mean(0).cpu().tolist()

            next_id = int(torch.argmax(out.logits[0, -1].to(torch.float32)))
            emit(
                "xray_step",
                {
                    "step": step,
                    "token": tokenizer.decode([next_id]),
                    "tokens": cur_tokens,
                    "n_layers": n_layers,
                    "logit_lens": logit_lens,
                    "grid": grid,
                    "attention": attention,
                },
            )

            generated.append(next_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
            if eos_id is not None and next_id == eos_id:
                break

    completion = tokenizer.decode(generated) if generated else ""
    emit("xray_complete", {"completion": completion, "n_steps": len(generated)})
    return {"completion": completion, "n_steps": len(generated)}
