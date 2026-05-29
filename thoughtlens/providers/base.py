"""BaseProvider ABC + ProviderOutput dataclass.

Every backend (OpenAI, Anthropic, HuggingFace, Ollama, Mock, custom) MUST
return a :class:`ProviderOutput` whose fields conform to this contract.
Downstream modules (FeatureExtractor, CircuitTracer, ReportBuilder, the TUI)
read these fields directly and must not branch on backend identity.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

EvidenceKind = Literal["white_box", "black_box"]


@dataclass
class ProviderOutput:
    """Normalised provider output envelope.

    Attributes
    ----------
    prompt:
        Original input string.
    tokens:
        Surface forms produced by the backend's tokenizer.  For black-box
        providers this falls back to a whitespace split of the prompt.
    token_ids:
        Integer ids parallel to ``tokens``.  Empty for backends that do
        not expose ids (OpenAI / Anthropic).
    activations:
        Real hidden-state tensor shaped ``(n_layers, n_tokens, d_model)``.
        ``None`` for black-box providers — the truthfulness guardrails
        require that ThoughtLens never fabricate this.
    attentions:
        Real attention tensor shaped ``(n_layers, n_heads, n_tokens, n_tokens)``.
        ``None`` for black-box providers.
    logits:
        Last-token logits ``(vocab_size,)`` for white-box providers.
        ``None`` for black-box providers.
    top_tokens:
        ``[(surface_form, probability), …]`` for the next token.  Computed
        from logits (white-box) or estimated from the provider's reported
        sampling probabilities (black-box).
    evidence_kind:
        ``"white_box"`` if ``activations`` is populated, ``"black_box"``
        otherwise.  Propagates to the report so the UI never claims direct
        internal observation it does not have.
    meta:
        Provider-specific metadata: model id, latency_ms, usage tokens,
        api_cost_usd, request_id, etc.
    """

    prompt: str
    tokens: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    activations: np.ndarray | None = None
    attentions: np.ndarray | None = None
    logits: np.ndarray | None = None
    top_tokens: list[tuple[str, float]] = field(default_factory=list)
    evidence_kind: EvidenceKind = "black_box"
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors used by the rest of the package
    # ------------------------------------------------------------------

    @property
    def n_tokens(self) -> int:
        """Number of input tokens."""
        return len(self.tokens)

    @property
    def n_layers(self) -> int:
        """Number of transformer layers for which we have activations."""
        return 0 if self.activations is None else int(self.activations.shape[0])

    @property
    def d_model(self) -> int:
        """Hidden dimension; 0 when no activations are available."""
        return 0 if self.activations is None else int(self.activations.shape[-1])

    @property
    def output_token(self) -> str:
        """Top-1 predicted next-token surface form (empty string if unknown)."""
        return self.top_tokens[0][0] if self.top_tokens else ""

    @property
    def output_prob(self) -> float:
        """Probability of the top-1 predicted token (0.0 if unknown)."""
        return self.top_tokens[0][1] if self.top_tokens else 0.0

    @property
    def has_internals(self) -> bool:
        """``True`` when real activations are present."""
        return self.activations is not None

    def to_summary(self) -> dict[str, Any]:
        """Lightweight dict suitable for JSON serialisation in the report."""
        return {
            "prompt": self.prompt,
            "tokens": self.tokens,
            "top_tokens": self.top_tokens,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "evidence_kind": self.evidence_kind,
            "meta": {k: v for k, v in self.meta.items() if _json_safe(v)},
        }


def _json_safe(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, list, dict)) or v is None


class BaseProvider(abc.ABC):
    """Abstract base for every ThoughtLens backend.

    Subclasses must implement :meth:`run` and declare :attr:`evidence_kind`
    (``"white_box"`` if they return real activations, otherwise ``"black_box"``).
    """

    #: Subclasses set this so the registry can pre-filter providers by capability.
    evidence_kind: EvidenceKind = "black_box"

    @abc.abstractmethod
    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        """Execute *prompt* and return a :class:`ProviderOutput`.

        Implementations must populate every field truthfully and must NOT
        synthesise ``activations`` / ``attentions`` / ``logits`` when the
        backend cannot supply them — leave them as ``None``.
        """

    @property
    def name(self) -> str:
        """Human-readable identifier displayed in the TUI/report."""
        return type(self).__name__

    @property
    def model_id(self) -> str:
        """Identifier of the underlying model.  Subclasses override."""
        return ""

    @property
    def supports_internals(self) -> bool:
        """``True`` when this provider can supply real activations."""
        return self.evidence_kind == "white_box"

    # ------------------------------------------------------------------
    # Optional capability hooks used by FeatureIntervention.
    # ------------------------------------------------------------------

    def run_with_intervention(
        self,
        prompt: str,
        interventions: list[Any] | None = None,
        **kwargs: Any,
    ) -> ProviderOutput:
        """Run *prompt* with optional mid-forward feature interventions.

        Default implementation delegates to :meth:`run` (no intervention)
        for providers that cannot intercept the forward pass.  HuggingFace
        provider overrides this with real PyTorch forward hooks.
        """
        return self.run(prompt, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model_id={self.model_id!r})"
