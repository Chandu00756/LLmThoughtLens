"""
BaseProvider ABC and ProviderOutput dataclass.

Every provider in llmscope must implement `BaseProvider.run()`.  The contract
is deliberately minimal so that providers for wildly-different backends (HTTP
APIs, local HuggingFace models, Ollama, …) can all return the same envelope.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class ProviderOutput:
    """Standardised envelope returned by every provider.

    Attributes
    ----------
    prompt:
        The original prompt string.
    tokens:
        Whitespace-split or tokenizer-split surface forms.
    token_ids:
        Integer token ids corresponding to ``tokens``.
    activations:
        Shape ``(n_layers, n_tokens, d_model)`` or ``None`` when the backend
        does not expose internal activations.
    attentions:
        Shape ``(n_layers, n_heads, n_tokens, n_tokens)`` or ``None``.
    logits:
        Shape ``(n_tokens, vocab_size)`` for the last generation step, or
        ``None`` when not available.
    top_tokens:
        ``[(surface_form, probability), …]`` for the *next* predicted token.
    meta:
        Freeform provider-specific metadata (model name, latency, …).
    """

    prompt: str
    tokens: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    activations: "np.ndarray | None" = None
    attentions: "np.ndarray | None" = None
    logits: "np.ndarray | None" = None
    top_tokens: list[tuple[str, float]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class BaseProvider(abc.ABC):
    """Abstract base class for all llmscope providers.

    Sub-classes **must** implement :meth:`run`.  All other helpers are
    optional overrides.
    """

    @abc.abstractmethod
    def run(self, prompt: str, **kwargs) -> ProviderOutput:
        """Execute the prompt and return a :class:`ProviderOutput`.

        Parameters
        ----------
        prompt:
            The text prompt to forward to the model.
        **kwargs:
            Provider-specific options (max_tokens, temperature, …).

        Returns
        -------
        ProviderOutput
        """

    # ------------------------------------------------------------------
    # Optional helpers — providers may override these for richer output.
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable identifier for this provider."""
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
