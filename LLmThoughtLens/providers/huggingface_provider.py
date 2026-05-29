"""HuggingFace provider — real white-box backend.

Loads any HuggingFace causal LM, runs a forward pass with
``output_hidden_states=True, output_attentions=True``, and packs the
resulting tensors into :class:`ProviderOutput` so downstream feature
extraction, SAE training, and the attribution-graph tracer have real
internals to work on.

Also exposes :meth:`run_with_intervention` that registers
``forward_pre_hook`` callbacks on the target transformer block so
:class:`FeatureIntervention` objects can amplify / inhibit / clamp
activations mid-forward-pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.utils.math_utils import softmax

if TYPE_CHECKING:
    import torch  # noqa: F401


class HuggingFaceProvider(BaseProvider):
    """White-box provider that loads any HuggingFace causal LM locally.

    Parameters
    ----------
    model_name:
        HuggingFace model id, e.g. ``"gpt2"`` or ``"meta-llama/Llama-3.2-1B"``.
    device:
        Target device.  ``"auto"`` picks cuda → mps → cpu in that order.
    torch_dtype:
        Optional dtype string, e.g. ``"float16"`` / ``"bfloat16"``.
    capture_internals:
        When ``True`` (default), hidden states and attentions are populated
        in the output envelope.
    """

    evidence_kind = "white_box"

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "auto",
        torch_dtype: str | None = None,
        capture_internals: bool = True,
    ) -> None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:  # pragma: no cover — gated by extras
            raise ImportError(
                "HuggingFaceProvider needs the `huggingface` extra. "
                "Install with: pip install 'LLmThoughtLens[huggingface]'"
            ) from exc

        self.model_name = model_name
        self._device_request = device
        self.torch_dtype = torch_dtype
        self.capture_internals = capture_internals
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def model_id(self) -> str:
        return f"hf/{self.model_name}"

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        import torch

        req = self._device_request
        if req != "auto":
            return req
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self) -> Any:
        import torch

        if self.torch_dtype is None:
            return None
        return {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(self.torch_dtype)

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = self._resolve_device()
        kwargs: dict[str, Any] = {
            "output_hidden_states": self.capture_internals,
            "output_attentions": self.capture_internals,
        }
        dtype = self._resolve_dtype()
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self._model.to(self._device)
        self._model.eval()

    # ------------------------------------------------------------------
    # BaseProvider API
    # ------------------------------------------------------------------

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        return self._forward(prompt, intervention_hooks=None, **kwargs)

    def run_with_intervention(
        self,
        prompt: str,
        interventions: list[Any] | None = None,
        **kwargs: Any,
    ) -> ProviderOutput:
        if not interventions:
            return self._forward(prompt, intervention_hooks=None, **kwargs)
        return self._forward(prompt, intervention_hooks=interventions, **kwargs)

    # ------------------------------------------------------------------
    # Forward pass with optional intervention hooks
    # ------------------------------------------------------------------

    def _forward(
        self,
        prompt: str,
        intervention_hooks: list[Any] | None,
        **kwargs: Any,
    ) -> ProviderOutput:
        import torch

        self._load()
        assert self._tokenizer is not None and self._model is not None

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        token_ids: list[int] = inputs["input_ids"][0].tolist()
        tokens: list[str] = [self._tokenizer.decode([tid]) for tid in token_ids]

        blocks = _resolve_transformer_blocks(self._model)
        handles: list[Any] = []
        if intervention_hooks:
            for spec in intervention_hooks:
                handle = _register_intervention_hook(blocks, spec)
                if handle is not None:
                    handles.append(handle)

        try:
            with torch.no_grad():
                outputs = self._model(
                    **inputs,
                    output_hidden_states=self.capture_internals,
                    output_attentions=self.capture_internals,
                    use_cache=False,
                    **kwargs,
                )
        finally:
            for h in handles:
                h.remove()

        activations: np.ndarray | None = None
        attentions: np.ndarray | None = None
        if self.capture_internals and outputs.hidden_states is not None:
            hs = torch.stack(outputs.hidden_states[1:], dim=0)  # (L, B, T, D)
            activations = hs.squeeze(1).to(torch.float32).cpu().numpy()
        if self.capture_internals and outputs.attentions is not None:
            attn = torch.stack(outputs.attentions, dim=0)  # (L, B, H, T, T)
            attentions = attn.squeeze(1).to(torch.float32).cpu().numpy()

        last_logits = outputs.logits[0, -1].to(torch.float32).cpu().numpy()
        probs = softmax(last_logits)
        top_k = min(5, probs.shape[0])
        top_idx = np.argpartition(-probs, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-probs[top_idx])]
        top_tokens: list[tuple[str, float]] = [
            (self._tokenizer.decode([int(i)]), float(probs[i])) for i in top_idx.tolist()
        ]

        meta: dict[str, Any] = {
            "provider": "HuggingFaceProvider",
            "model": self.model_name,
            "device": str(self._device),
            "n_intervention_hooks": len(handles),
            "evidence_note": (
                "Hidden states and attentions captured via "
                "output_hidden_states / output_attentions on the underlying "
                "transformer — direct internal observation."
            ),
        }
        return ProviderOutput(
            prompt=prompt,
            tokens=tokens,
            token_ids=token_ids,
            activations=activations,
            attentions=attentions,
            logits=last_logits.astype(np.float32),
            top_tokens=top_tokens,
            evidence_kind="white_box",
            meta=meta,
        )


# ---------------------------------------------------------------------------
# Module resolution helpers
# ---------------------------------------------------------------------------


def _resolve_transformer_blocks(model: Any) -> list[Any]:
    """Return the ordered list of transformer-block modules.

    Walks common HF attribute paths (GPT-2, Llama, Mistral, Phi, Qwen, OPT,
    GPT-NeoX).  Falls back to scanning ``named_modules`` for the first
    ``ModuleList`` whose children look like transformer blocks.
    """
    candidates: list[tuple[str, ...]] = [
        ("transformer", "h"),
        ("model", "layers"),
        ("gpt_neox", "layers"),
        ("transformer", "blocks"),
        ("model", "decoder", "layers"),
    ]
    for path in candidates:
        obj: Any = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue

    import torch.nn as nn

    for _name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            head = mod[0]
            attrs = dir(head)
            if any(a in attrs for a in ("self_attn", "attention", "attn")) and any(
                a in attrs for a in ("mlp", "feed_forward")
            ):
                return list(mod)
    return []


def _register_intervention_hook(blocks: list[Any], spec: Any) -> Any:
    """Register one forward-pre-hook on the target block; return its handle."""
    if not blocks:
        return None
    target = int(getattr(spec, "layer", 0)) % len(blocks)
    block = blocks[target]

    def _pre_hook(_module: Any, inputs: Any) -> Any:
        if not inputs:
            return inputs
        first = inputs[0]
        if first is None:
            return inputs
        modified = spec.apply_torch(first)
        return (modified,) + tuple(inputs[1:])

    return block.register_forward_pre_hook(_pre_hook)
