"""Provider registry — lazy factory so optional backends never ImportError on import."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from thoughtlens.providers.base import BaseProvider

# Map provider name → zero-arg loader returning the provider class.
# Lazy by design: importing thoughtlens must not blow up when transformers,
# openai, or anthropic are not installed.
_LOADERS: dict[str, Callable[[], type[BaseProvider]]] = {}


def _register_lazy(name: str, loader: Callable[[], type[BaseProvider]]) -> None:
    _LOADERS[name] = loader


def register_provider(name: str, cls: type[BaseProvider]) -> None:
    """Register a custom provider class under *name*."""
    _LOADERS[name] = lambda c=cls: c


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
        except Exception:  # noqa: BLE001 — best-effort discovery
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
# Built-in loaders
# ---------------------------------------------------------------------------


def _load_mock() -> type[BaseProvider]:
    from thoughtlens.providers.mock_provider import MockProvider

    return MockProvider


def _load_openai() -> type[BaseProvider]:
    from thoughtlens.providers.openai_provider import OpenAIProvider

    return OpenAIProvider


def _load_anthropic() -> type[BaseProvider]:
    from thoughtlens.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider


def _load_huggingface() -> type[BaseProvider]:
    from thoughtlens.providers.huggingface_provider import HuggingFaceProvider

    return HuggingFaceProvider


def _load_ollama() -> type[BaseProvider]:
    from thoughtlens.providers.ollama_provider import OllamaProvider

    return OllamaProvider


_register_lazy("mock", _load_mock)
_register_lazy("openai", _load_openai)
_register_lazy("anthropic", _load_anthropic)
_register_lazy("huggingface", _load_huggingface)
_register_lazy("ollama", _load_ollama)
