"""Provider registry — factory helpers for all built-in providers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmscope.providers.base import BaseProvider

if TYPE_CHECKING:
    pass

_REGISTRY: dict[str, type[BaseProvider]] = {}


def register_provider(name: str, cls: type[BaseProvider]) -> None:
    """Register a provider class under *name* for use with :func:`get_provider`."""
    _REGISTRY[name] = cls


def get_provider(name: str, **kwargs) -> BaseProvider:
    """Instantiate a registered provider by name.

    Parameters
    ----------
    name:
        Registered name, e.g. ``"mock"``, ``"openai"``, ``"anthropic"``,
        ``"huggingface"``, ``"ollama"``.
    **kwargs:
        Passed to the provider's ``__init__``.

    Raises
    ------
    KeyError
        When *name* is not in the registry.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"Unknown provider {name!r}. Available: {available or '(none registered)'}"
        )
    return _REGISTRY[name](**kwargs)


def list_providers() -> list[str]:
    """Return sorted list of registered provider names."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Auto-register built-in providers
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    from llmscope.providers.mock_provider import MockProvider

    register_provider("mock", MockProvider)

    try:
        from llmscope.providers.openai_provider import OpenAIProvider

        register_provider("openai", OpenAIProvider)
    except ImportError:
        pass

    try:
        from llmscope.providers.anthropic_provider import AnthropicProvider

        register_provider("anthropic", AnthropicProvider)
    except ImportError:
        pass

    try:
        from llmscope.providers.huggingface_provider import HuggingFaceProvider

        register_provider("huggingface", HuggingFaceProvider)
    except ImportError:
        pass

    try:
        from llmscope.providers.ollama_provider import OllamaProvider

        register_provider("ollama", OllamaProvider)
    except ImportError:
        pass


_register_builtins()
