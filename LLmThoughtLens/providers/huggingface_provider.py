"""HuggingFace provider — real white-box backend.

Loads any HuggingFace causal LM locally and runs it through
:class:`~LLmThoughtLens.models.hooked.HookedModel`, the shared residual-stream
layer, so :class:`ProviderOutput` carries real internals:

* ``activations`` — the **true** residual stream ``resid_post`` ``(L, T, D)``
  captured with forward hooks on every transformer block.  Unlike HF's
  ``output_hidden_states`` the last layer is *not* passed through the final
  norm, so every layer is on the same footing and
  ``hooked.logit_lens(activations[-1])`` reproduces the model's logits.
* ``embeddings`` — ``resid_pre[0]`` ``(T, D)``, the embedding output entering
  block 0.
* ``attentions`` — eager-attention weights ``(L, H, T, T)``.  Models are loaded
  with ``attn_implementation="eager"`` when internals are captured (SDPA /
  flash attention return no weights); a model that rejects eager attention
  falls back to its default and reports ``attentions=None``.

:meth:`run_with_intervention` installs :class:`FeatureIntervention` hooks for
the duration of one forward pass; :attr:`hooked` exposes the
:class:`HookedModel` itself for gradient attribution, patching, steering and
generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.utils.math_utils import softmax

if TYPE_CHECKING:
    from LLmThoughtLens.models.hooked import HookedModel


class HuggingFaceProvider(BaseProvider):
    """White-box provider that loads any HuggingFace causal LM locally.

    Parameters
    ----------
    model_name:
        HuggingFace model id, e.g. ``"gpt2"`` or ``"meta-llama/Llama-3.2-1B"``,
        or a local weights folder.
    device:
        Target device.  ``"auto"`` picks cuda → mps → cpu in that order.
    torch_dtype:
        Optional dtype string, e.g. ``"float16"`` / ``"bfloat16"``.
    capture_internals:
        When ``True`` (default), the residual stream, embeddings and attention
        weights are populated in the output envelope.
    """

    evidence_kind = "white_box"
    supports_gradients = True

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
        self._hooked: HookedModel | None = None

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
        from LLmThoughtLens.models.hooked import resolve_device

        return str(resolve_device(self._device_request))

    def _resolve_dtype(self) -> Any:
        from LLmThoughtLens.models.hooked import resolve_dtype

        return resolve_dtype(self.torch_dtype, strict=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        from LLmThoughtLens.models.hooked import load_hf_model

        # Only construction kwargs reach from_pretrained (never output_* flags,
        # which transformers 5 rejects as invalid generation flags).  Eager
        # attention is requested only when attention weights will be read.
        self._model, self._tokenizer, self._device = load_hf_model(
            self.model_name,
            device=self._resolve_device(),
            dtype=self._resolve_dtype(),
            attn_implementation="eager" if self.capture_internals else None,
        )

    @property
    def hooked(self) -> HookedModel:
        """The :class:`HookedModel` over this provider's model (loads it on first use).

        Built lazily around whatever ``_model`` / ``_tokenizer`` / ``_device``
        currently hold, and rebuilt if they are replaced.
        """
        self._load()
        hm = self._hooked
        if hm is None or hm.model is not self._model or hm.tokenizer is not self._tokenizer:
            from LLmThoughtLens.models.hooked import HookedModel

            hm = HookedModel(self._model, self._tokenizer, device=self._device)
            self._hooked = hm
        return hm

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

        from LLmThoughtLens.features.intervention import intervention_context

        hm = self.hooked
        tp = hm.tokenize(prompt)

        # One shared, exception-safe code path installs and removes the hooks.
        with intervention_context(hm.blocks, intervention_hooks or []) as ictx:
            res = hm.forward(
                tp,
                capture=self.capture_internals,
                capture_attentions=self.capture_internals,
                **kwargs,
            )
            n_intervention_hooks = ictx.n_installed

        def _np(t: Any) -> np.ndarray:
            return np.asarray(t.detach().to(torch.float32).cpu().numpy(), dtype=np.float32)

        activations = _np(res.resid_post) if res.resid_post is not None else None
        embeddings = _np(res.resid_pre[0]) if res.resid_pre is not None else None
        attentions = _np(res.attentions) if res.attentions is not None else None

        last_logits = _np(res.logits[0, -1])
        probs = softmax(last_logits)
        top_k = min(5, probs.shape[0])
        top_idx = np.argpartition(-probs, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-probs[top_idx])]
        top_tokens: list[tuple[str, float]] = [
            (hm.token_str(int(i)), float(probs[i])) for i in top_idx.tolist()
        ]

        meta: dict[str, Any] = {
            "provider": "HuggingFaceProvider",
            "model": self.model_name,
            "device": str(self._device),
            "family": hm.family,
            "n_intervention_hooks": int(n_intervention_hooks),
            "activation_site": "resid_post",
            "bos_token_id": getattr(hm.tokenizer, "bos_token_id", None),
            "evidence_note": (
                "Residual stream (true resid_post, no final norm) captured with "
                "forward hooks on every transformer block; attention weights from "
                "eager attention — direct internal observation."
            ),
        }
        return ProviderOutput(
            prompt=prompt,
            tokens=list(tp.tokens),
            token_ids=list(tp.token_ids),
            activations=activations,
            attentions=attentions,
            logits=last_logits,
            top_tokens=top_tokens,
            evidence_kind="white_box",
            meta=meta,
            embeddings=embeddings,
        )


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------


def _resolve_transformer_blocks(model: Any) -> list[Any]:
    """Alias of :func:`LLmThoughtLens.models.families.resolve_transformer_blocks`."""
    from LLmThoughtLens.models.families import resolve_transformer_blocks

    return resolve_transformer_blocks(model)


def _register_intervention_hook(blocks: list[Any], spec: Any) -> Any:
    """Compatibility shim — delegates to :func:`intervention._install_mlp_hook`.

    Kept so any external caller that imported this name continues to work;
    new code should use :class:`intervention.intervention_context` (or
    :meth:`HookedModel.hooks`) instead.
    """
    from LLmThoughtLens.features.intervention import _install_mlp_hook

    return _install_mlp_hook(blocks, spec)
