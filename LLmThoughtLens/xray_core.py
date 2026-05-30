"""xray_core — the logit-lens forward loop, with no web dependency.

Shared by the dashboard server (``server/xray.py``) and the SDK
(``sdk.attach``) so the same real computation drives both an HTTP endpoint and
an in-process model the user already has.  Depends only on ``torch`` +
``transformers`` (the ``huggingface`` extra), never on FastAPI.

The transport is decoupled via an ``emit(kind, data)`` callback: the server
passes ``bus.publish``; the SDK passes a function that POSTs to a dashboard.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# The return value is ignored; ``Any`` lets both ``bus.publish`` (returns the
# event dict) and the SDK's push wrapper (returns None) satisfy this type.
EmitFn = Callable[[str, dict[str, Any]], Any]


def resolve_final_norm(model: Any) -> Any:
    """Return the model's final pre-unembedding norm module, or ``None``.

    Tries the common HF attribute paths (GPT-2, Llama/Mistral/Qwen, GPT-NeoX,
    OPT).  The final norm is applied before the unembedding for a faithful
    logit lens.
    """
    for path in (
        ("transformer", "ln_f"),  # GPT-2
        ("model", "norm"),  # Llama / Mistral / Qwen
        ("gpt_neox", "final_layer_norm"),  # GPT-NeoX
        ("model", "decoder", "final_layer_norm"),  # OPT
        ("transformer", "norm_f"),  # some others
    ):
        obj = model
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok:
            return obj
    return None


def infer_device(model: Any) -> Any:
    """Best-effort device of a loaded model's parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover — empty model
        import torch

        return torch.device("cpu")


def run_xray_loop(
    model: Any,
    tokenizer: Any,
    device: Any,
    prompt: str,
    max_new_tokens: int,
    emit: EmitFn,
    model_label: str = "model",
) -> dict[str, Any]:
    """Generate token-by-token, emitting real logit-lens / activation events.

    For every generated token this emits an ``xray_step`` with:

    * ``logit_lens`` — each layer's last-position residual stream projected
      through the model's real final-norm + unembedding (top-3 tokens). Reading
      this column bottom→top shows the prediction forming inside the network.
    * ``grid`` — an ``(n_layers x n_tokens)`` residual-stream magnitude map.
    * ``attention`` — the last layer's head-averaged attention (short prompts).

    Returns ``{"completion", "n_steps"}``.  All numbers come from a real forward
    pass; there is no synthetic path.
    """
    import torch

    unembed = model.get_output_embeddings()
    final_norm = resolve_final_norm(model)

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id

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
                    if final_norm is not None:
                        h = final_norm(h)
                    probs_l = torch.softmax(unembed(h).to(torch.float32), dim=-1)
                    p, idx = torch.topk(probs_l, k=3)
                    logit_lens.append(
                        {
                            "layer": li,
                            "top": [
                                [tokenizer.decode([int(i)]), float(pp)]
                                for i, pp in zip(idx.tolist(), p.tolist(), strict=False)
                            ],
                        }
                    )

            grid = [
                [float(hidden[li][0, t].to(torch.float32).norm().cpu()) for t in range(seq)]
                for li in range(n_layers)
            ]

            # ``out.attentions`` may be None OR an empty tuple when the model's
            # attention backend (e.g. SDPA) doesn't return weights even when
            # asked — guard against both so attach() works on any loaded model.
            attention: list[list[float]] = []
            if out.attentions and seq <= 40:
                attention = out.attentions[-1][0].to(torch.float32).mean(0).cpu().tolist()

            next_id = int(torch.argmax(out.logits[0, -1].to(torch.float32)))
            next_tok = tokenizer.decode([next_id])

            emit(
                "xray_step",
                {
                    "step": step,
                    "token": next_tok,
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
