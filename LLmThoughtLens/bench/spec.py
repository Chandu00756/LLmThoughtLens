"""Model specs — which (provider, model) cells a benchmark run covers.

A spec is written either as a string or as a :class:`ModelSpec`:

* ``"mock"`` — the synthetic mock provider (results are flagged synthetic);
* ``"hf:gpt2"`` / ``"huggingface:distilgpt2"`` — a local HuggingFace model;
* ``"ollama:qwen3:1.7b"`` — everything after the first ``:`` is the model tag;
* ``"openai:gpt-4o-mini"``, ``"anthropic:claude-haiku-4-5"``.

Constructor arguments go in :attr:`ModelSpec.kwargs`; a ``factory`` callable
can replace provider construction entirely (custom providers, tests).
Secrets in ``kwargs`` (``api_key`` and the like) are never written to results.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from LLmThoughtLens.providers.base import BaseProvider

__all__ = ["PROVIDER_ALIASES", "ModelSpec", "parse_spec", "parse_specs"]

#: Accepted provider spellings → registry name.
PROVIDER_ALIASES: dict[str, str] = {
    "mock": "mock",
    "hf": "huggingface",
    "huggingface": "huggingface",
    "transformers": "huggingface",
    "ollama": "ollama",
    "openai": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
}

_LABEL_PREFIX = {"huggingface": "hf"}
_SECRET_MARKERS = ("key", "token", "secret", "password", "auth")
#: Providers that run locally (warm-up calls cost nothing).
LOCAL_PROVIDERS = frozenset({"mock", "huggingface", "ollama"})


@dataclass
class ModelSpec:
    """One model in a benchmark matrix.

    Parameters
    ----------
    provider:
        Registry name or alias (``"hf"`` → ``"huggingface"``).
    model:
        Model id / tag; empty selects the provider default (unused by mock).
    kwargs:
        Extra provider constructor arguments.
    label:
        Unique display name; defaults to ``"<provider>/<model>"`` (``"hf/gpt2"``).
    factory:
        Optional zero-argument callable returning the provider; bypasses the
        registry.  Not serialised.
    """

    provider: str
    model: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    factory: Callable[[], BaseProvider] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.provider = PROVIDER_ALIASES.get(self.provider.strip().lower(), self.provider.strip())
        if not self.label:
            prefix = _LABEL_PREFIX.get(self.provider, self.provider)
            self.label = f"{prefix}/{self.model}" if self.model else prefix

    @classmethod
    def parse(cls, text: str, **kwargs: Any) -> ModelSpec:
        """Parse ``"provider[:model]"`` (see the module docstring)."""
        raw = text.strip()
        if not raw:
            raise ValueError("empty model spec")
        provider, _, model = raw.partition(":")
        return cls(provider=provider, model=model.strip(), kwargs=dict(kwargs))

    @property
    def is_local(self) -> bool:
        return self.provider in LOCAL_PROVIDERS

    def build(self) -> BaseProvider:
        """Construct the provider (lazy imports; optional extras may raise ImportError)."""
        if self.factory is not None:
            return self.factory()
        from LLmThoughtLens.providers.defaults import provider_kwargs
        from LLmThoughtLens.providers.registry import get_provider

        kw: dict[str, Any] = provider_kwargs(self.provider, self.model or None)
        if self.provider == "ollama":
            # Benchmarks gate logprobs on the server version and record it.
            kw.setdefault("request_logprobs", "auto")
        kw.update(self.kwargs)
        return get_provider(self.provider, **kw)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe description with secrets redacted."""
        return {
            "label": self.label,
            "provider": self.provider,
            "model": self.model,
            "kwargs": {k: _redact(k, v) for k, v in self.kwargs.items()},
            "custom_factory": self.factory is not None,
        }


def _redact(key: str, value: Any) -> Any:
    if any(m in key.lower() for m in _SECRET_MARKERS):
        return "***" if value else value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def parse_spec(spec: str | ModelSpec) -> ModelSpec:
    """Return *spec* as a :class:`ModelSpec`."""
    return spec if isinstance(spec, ModelSpec) else ModelSpec.parse(spec)


def parse_specs(specs: Sequence[str | ModelSpec] | str) -> list[ModelSpec]:
    """Parse a list of specs (or one comma-separated string) and check labels are unique."""
    items: Sequence[str | ModelSpec]
    items = [s for s in specs.split(",") if s.strip()] if isinstance(specs, str) else specs
    parsed = [parse_spec(s) for s in items]
    if not parsed:
        raise ValueError("at least one model spec is required")
    labels = [s.label for s in parsed]
    dupes = sorted({x for x in labels if labels.count(x) > 1})
    if dupes:
        raise ValueError(f"duplicate model labels {dupes}; give each spec a distinct label")
    return parsed
