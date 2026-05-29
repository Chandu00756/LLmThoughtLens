"""Provider registry — lazy factory so optional backends never break import."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from LLmThoughtLens.providers.base import BaseProvider

_LOADERS: dict[str, Callable[[], type[BaseProvider]]] = {}


def _register_lazy(name: str, loader: Callable[[], type[BaseProvider]]) -> None:
    _LOADERS[name] = loader


def register_provider(name: str, cls: type[BaseProvider]) -> None:
    """Register a custom provider class under *name*."""

    def _loader() -> type[BaseProvider]:
        return cls

    _LOADERS[name] = _loader


def list_providers() -> list[str]:
    """All names known to the registry, sorted."""
    return sorted(_LOADERS)


def available_providers() -> list[str]:
    """Subset of :func:`list_providers` whose import succeeds right now."""
    available: list[str] = []
    for name, loader in _LOADERS.items():
        try:
            loader()
            available.append(name)
        except Exception:  # noqa: BLE001 — best-effort capability discovery
            continue
    return sorted(available)


def get_provider(name: str, **kwargs: Any) -> BaseProvider:
    """Instantiate a registered provider.

    Raises
    ------
    KeyError:
        If *name* is unknown.
    ImportError:
        If the provider's optional extras are not installed (propagated
        from the loader).
    """
    if name not in _LOADERS:
        raise KeyError(
            f"unknown provider {name!r}; known: {', '.join(list_providers()) or '(none)'}"
        )
    cls = _LOADERS[name]()
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Built-in loaders (one indirection so optional deps never fire at import).
# ---------------------------------------------------------------------------


def _load_mock() -> type[BaseProvider]:
    from LLmThoughtLens.providers.mock_provider import MockProvider

    return MockProvider


def _load_openai() -> type[BaseProvider]:
    from LLmThoughtLens.providers.openai_provider import OpenAIProvider

    return OpenAIProvider


def _load_anthropic() -> type[BaseProvider]:
    from LLmThoughtLens.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider


def _load_huggingface() -> type[BaseProvider]:
    from LLmThoughtLens.providers.huggingface_provider import HuggingFaceProvider

    return HuggingFaceProvider


def _load_ollama() -> type[BaseProvider]:
    from LLmThoughtLens.providers.ollama_provider import OllamaProvider

    return OllamaProvider


_register_lazy("mock", _load_mock)
_register_lazy("openai", _load_openai)
_register_lazy("anthropic", _load_anthropic)
_register_lazy("huggingface", _load_huggingface)
_register_lazy("ollama", _load_ollama)
