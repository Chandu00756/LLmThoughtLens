"""Scope — main entry point for LLM interpretability workflows in llmscope."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmscope.providers.base import BaseProvider, ProviderOutput

if TYPE_CHECKING:
    pass


class Scope:
    """High-level façade that wires together a provider, probes, and tracers.

    Create a :class:`Scope` using one of the class-method factories:

    .. code-block:: python

        scope = Scope.from_mock()
        out   = scope.trace("Hello world")

    Parameters
    ----------
    provider:
        The backend that executes prompts and returns activations.
    """

    def __init__(self, provider: BaseProvider) -> None:
        self._provider = provider

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_mock(cls, **kwargs: Any) -> "Scope":
        """Create a Scope backed by the deterministic :class:`~llmscope.providers.mock_provider.MockProvider`.

        Parameters
        ----------
        **kwargs:
            Passed to ``MockProvider.__init__`` (e.g. ``seed=0``,
            ``n_layers=4``).
        """
        from llmscope.providers.mock_provider import MockProvider

        return cls(MockProvider(**kwargs))

    @classmethod
    def from_openai(cls, api_key: str | None = None, model: str = "gpt-4o-mini") -> "Scope":
        """Create a Scope backed by the OpenAI Chat Completions API."""
        from llmscope.providers.openai_provider import OpenAIProvider

        return cls(OpenAIProvider(api_key=api_key, model=model))

    @classmethod
    def from_anthropic(
        cls,
        api_key: str | None = None,
        model: str = "claude-3-5-haiku-20241022",
    ) -> "Scope":
        """Create a Scope backed by the Anthropic Messages API."""
        from llmscope.providers.anthropic_provider import AnthropicProvider

        return cls(AnthropicProvider(api_key=api_key, model=model))

    @classmethod
    def from_huggingface(cls, model_name: str = "gpt2", device: str = "cpu") -> "Scope":
        """Create a Scope backed by a local HuggingFace model."""
        from llmscope.providers.huggingface_provider import HuggingFaceProvider

        return cls(HuggingFaceProvider(model_name=model_name, device=device))

    @classmethod
    def from_ollama(cls, model: str = "llama3.2", base_url: str = "http://localhost:11434") -> "Scope":
        """Create a Scope backed by a locally-running Ollama instance."""
        from llmscope.providers.ollama_provider import OllamaProvider

        return cls(OllamaProvider(model=model, base_url=base_url))

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def trace(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        """Run the prompt through the provider and return the raw output.

        Parameters
        ----------
        prompt:
            The input text to forward to the model.
        **kwargs:
            Provider-specific options forwarded to :meth:`~BaseProvider.run`.

        Returns
        -------
        ProviderOutput
            Structured envelope containing tokens, activations, attentions,
            logits, and metadata.
        """
        return self._provider.run(prompt, **kwargs)

    @property
    def provider(self) -> BaseProvider:
        """The underlying provider instance."""
        return self._provider

    def __repr__(self) -> str:
        return f"Scope(provider={self._provider!r})"
