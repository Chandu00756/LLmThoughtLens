"""BaseProvider ABC + ProviderOutput dataclass.

Every backend (OpenAI, Anthropic, HuggingFace, Ollama, Mock, custom) returns
a :class:`ProviderOutput` with the same fields.  Downstream modules
(``FeatureExtractor``, ``CircuitTracer``, ``ReportBuilder``, the TUI) read
those fields directly and never branch on backend identity.

Truthfulness contract: ``activations`` / ``attentions`` / ``logits`` MUST
be ``None`` whenever the backend cannot legitimately observe them (closed
API).  They MUST NEVER be synthesised — that would lie about evidence kind.
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
        Original input string passed to :meth:`BaseProvider.run`.
    tokens:
        Surface forms produced by the backend's tokeniser.  For black-box
        backends this is a whitespace split of the prompt's text.
    token_ids:
        Integer ids parallel to ``tokens`` (empty for backends that don't
        expose ids).
    activations:
        Real hidden-state tensor ``(n_layers, n_tokens, d_model)``.
        ``None`` for black-box providers.
    attentions:
        Real attention tensor ``(n_layers, n_heads, n_tokens, n_tokens)``.
        ``None`` for black-box providers.
    logits:
        Last-token logits ``(vocab_size,)`` for white-box providers.
        ``None`` for black-box providers.
    top_tokens:
        ``[(surface_form, probability), …]`` for the next token.  Computed
        from logits (white-box) or from real OpenAI logprobs when available;
        a single sampled-token entry otherwise (Anthropic, Ollama).
    evidence_kind:
        ``"white_box"`` if ``activations`` is populated, else ``"black_box"``.
        Propagates to the UI so a closed-model trace never claims direct
        internal observation.
    meta:
        Provider-specific metadata: model id, latency_ms, usage tokens,
        api_cost_usd, evidence_note, etc.
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
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def n_layers(self) -> int:
        return 0 if self.activations is None else int(self.activations.shape[0])

    @property
    def n_heads(self) -> int:
        return 0 if self.attentions is None else int(self.attentions.shape[1])

    @property
    def d_model(self) -> int:
        return 0 if self.activations is None else int(self.activations.shape[-1])

    @property
    def output_token(self) -> str:
        return self.top_tokens[0][0] if self.top_tokens else ""

    @property
    def output_prob(self) -> float:
        return self.top_tokens[0][1] if self.top_tokens else 0.0

    @property
    def has_internals(self) -> bool:
        return self.activations is not None

    def to_summary(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "tokens": self.tokens,
            "top_tokens": self.top_tokens,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "evidence_kind": self.evidence_kind,
            "meta": {k: v for k, v in self.meta.items() if _json_safe(v)},
        }


def _json_safe(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, list, dict)) or v is None


class BaseProvider(abc.ABC):
    """Abstract base for every LLmThoughtLens backend.

    Subclasses MUST implement :meth:`run` and set :attr:`evidence_kind`.
    """

    evidence_kind: EvidenceKind = "black_box"

    @abc.abstractmethod
    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        """Execute *prompt* and return a :class:`ProviderOutput`.

        Implementations MUST populate fields truthfully and MUST NOT
        synthesise ``activations`` / ``attentions`` / ``logits`` when the
        backend does not expose them — leave them as ``None``.
        """

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def model_id(self) -> str:
        return ""

    @property
    def supports_internals(self) -> bool:
        return self.evidence_kind == "white_box"

    def run_with_intervention(
        self,
        prompt: str,
        interventions: list[Any] | None = None,
        **kwargs: Any,
    ) -> ProviderOutput:
        """Run *prompt* with optional mid-forward feature interventions.

        Default implementation ignores interventions; the HuggingFace
        provider overrides this with real torch forward-pre hooks.
        """
        return self.run(prompt, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model_id={self.model_id!r})"
