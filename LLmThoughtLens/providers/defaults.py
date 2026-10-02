"""Single source of truth for per-provider default model ids and endpoints.

Every entry point (``Scope.from_*``, the CLI, the SDK, the TUI, and the
dashboard server config) resolves defaults through this module so they can
never drift apart again.

Overrides
---------
Each provider's default model can be overridden without code changes via an
environment variable named ``LLMTHOUGHTLENS_<PROVIDER>_MODEL`` — for example
``LLMTHOUGHTLENS_ANTHROPIC_MODEL=claude-sonnet-5-5``.  The default Ollama URL
can likewise be overridden with ``LLMTHOUGHTLENS_OLLAMA_URL``.
"""

from __future__ import annotations

import os
import re
from types import MappingProxyType
from typing import Any

__all__ = [
    "DEFAULT_MODELS",
    "DEFAULT_OLLAMA_URL",
    "default_model",
    "default_ollama_url",
    "model_env_var",
    "provider_kwargs",
    "resolve_model",
    "upgrade_retired_model",
]

#: Built-in default model id per provider.  ``mock`` takes no model id.
DEFAULT_MODELS: MappingProxyType[str, str] = MappingProxyType(
    {
        "mock": "",
        "openai": "gpt-4o-mini",
        # Cheapest current Claude model; alias form (no date suffix) per the
        # Anthropic model table, so it tracks the latest Haiku 4.5 snapshot.
        "anthropic": "claude-haiku-4-5",
        "huggingface": "gpt2",
        "ollama": "llama3.2",
    }
)

#: Default base URL of a locally running Ollama server.
DEFAULT_OLLAMA_URL = "http://localhost:11434"

#: Model ids that earlier LLmThoughtLens releases shipped as defaults (and
#: persisted into ``~/.LLmThoughtLens/server.json``) but that the upstream
#: provider has since retired.  Loading a saved config upgrades these to the
#: current default so existing installs keep working.
_RETIRED_DEFAULTS: MappingProxyType[str, frozenset[str]] = MappingProxyType(
    {
        "anthropic": frozenset({"claude-3-5-haiku-20241022"}),
    }
)

_ENV_SANITIZE = re.compile(r"[^A-Za-z0-9]+")


def model_env_var(provider: str) -> str:
    """Return the env var that overrides *provider*'s default model.

    Examples
    --------
    >>> model_env_var("anthropic")
    'LLMTHOUGHTLENS_ANTHROPIC_MODEL'
    """
    slug = _ENV_SANITIZE.sub("_", provider).strip("_").upper()
    return f"LLMTHOUGHTLENS_{slug}_MODEL"


def default_model(provider: str) -> str:
    """Return the default model id for *provider*.

    Honours the ``LLMTHOUGHTLENS_<PROVIDER>_MODEL`` environment variable when
    it is set to a non-empty value.  Unknown providers default to ``""``.

    Parameters
    ----------
    provider:
        Registry name, e.g. ``"openai"`` or ``"anthropic"``.

    Returns
    -------
    str
        The model id to use when the caller did not supply one.
    """
    override = os.environ.get(model_env_var(provider), "").strip()
    if override:
        return override
    return DEFAULT_MODELS.get(provider, "")


def resolve_model(provider: str, model: str | None) -> str:
    """Return *model* if it is non-empty, else the provider default."""
    if model:
        return model
    return default_model(provider)


def default_ollama_url() -> str:
    """Return the Ollama base URL, honouring ``LLMTHOUGHTLENS_OLLAMA_URL``."""
    override = os.environ.get("LLMTHOUGHTLENS_OLLAMA_URL", "").strip()
    return override or DEFAULT_OLLAMA_URL


def upgrade_retired_model(provider: str, model: str) -> str:
    """Replace a retired former-default model id with the current default.

    Only ids that LLmThoughtLens itself used to ship as defaults are
    rewritten; any other user-chosen id is returned unchanged.
    """
    if model in _RETIRED_DEFAULTS.get(provider, frozenset()):
        return default_model(provider)
    return model


def provider_kwargs(
    provider: str,
    model: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Build constructor kwargs for a built-in provider with defaults applied.

    This is the one place that maps generic "provider / model / key / url"
    settings onto each provider class's constructor signature.  Empty strings
    are treated as "not set".  API keys left unset fall back to the
    provider's own environment variable inside the provider class.

    Parameters
    ----------
    provider:
        Registry name.  Unknown (custom-registered) providers get ``{}``.
    model:
        Model id; empty or ``None`` selects :func:`default_model`.
    api_key:
        Optional API key for black-box API providers.
    base_url:
        Optional endpoint override (OpenAI-compatible proxy, Anthropic
        gateway, or Ollama server).
    device:
        Optional torch device for the HuggingFace provider.

    Returns
    -------
    dict
        Keyword arguments suitable for ``get_provider(provider, **kwargs)``.
    """
    kwargs: dict[str, Any] = {}
    if provider in ("openai", "anthropic"):
        kwargs = {"model": resolve_model(provider, model), "api_key": api_key or None}
        if base_url:
            kwargs["base_url"] = base_url
    elif provider == "huggingface":
        kwargs = {"model_name": resolve_model(provider, model)}
        if device:
            kwargs["device"] = device
    elif provider == "ollama":
        kwargs = {
            "model": resolve_model(provider, model),
            "base_url": base_url or default_ollama_url(),
        }
    return kwargs
