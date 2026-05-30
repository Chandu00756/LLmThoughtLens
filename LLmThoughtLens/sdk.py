"""ThoughtLens SDK — observe your own LLM app in three lines.

For people building their own AI apps / agents / local models.  Two honest
integration styles:

1. **Trace what you run yourself** — point the SDK at any provider and trace a
   prompt, optionally streaming the result to a running dashboard::

       from LLmThoughtLens.sdk import trace
       r = trace("The capital of France is", provider="ollama", model="llama3.1:8b")
       print(r.output_token, r.top_features(3))

2. **Wrap a client you already use** — drop-in over an OpenAI-style client so
   every call you make is visualised, while the real response is returned
   untouched::

       from openai import OpenAI
       from LLmThoughtLens.sdk import wrap_openai
       client = wrap_openai(OpenAI(), dashboard="http://localhost:8000")
       client.chat.completions.create(model="gpt-4o-mini", messages=[...])

3. **Record an exchange you already have** — if your stack already produced a
   prompt+completion and you just want it on the live dashboard::

       from LLmThoughtLens.sdk import record_exchange
       record_exchange(prompt, completion, dashboard="http://localhost:8000")

The dashboard connection is optional; without it everything still works
in-process and returns real objects.  Nothing here calls home — the only
network calls are to the upstream model you configured and (optionally) to your
own local dashboard URL.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
    from LLmThoughtLens.scope import TraceResult


# ---------------------------------------------------------------------------
# Provider construction (standalone — no server config needed)
# ---------------------------------------------------------------------------


def _build_provider(
    provider: str,
    model: str = "",
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> BaseProvider:
    from LLmThoughtLens.providers.registry import get_provider

    pk: dict[str, Any] = dict(kwargs)
    if provider == "openai":
        pk.setdefault("model", model or "gpt-4o-mini")
        pk.setdefault("api_key", api_key or os.environ.get("OPENAI_API_KEY"))
    elif provider == "anthropic":
        pk.setdefault("model", model or "claude-3-5-haiku-20241022")
        pk.setdefault("api_key", api_key or os.environ.get("ANTHROPIC_API_KEY"))
    elif provider == "huggingface":
        pk.setdefault("model_name", model or "gpt2")
    elif provider == "ollama":
        pk.setdefault("model", model or "llama3.1:8b")
        pk.setdefault("base_url", base_url or "http://localhost:11434")
    return get_provider(provider, **pk)


def _push(dashboard: str | None, kind: str, data: dict[str, Any]) -> None:
    """Best-effort POST of an event to a running dashboard. Never raises."""
    if not dashboard:
        return
    try:
        import httpx

        httpx.post(
            f"{dashboard.rstrip('/')}/api/ingest",
            json={"kind": kind, "data": data},
            timeout=5.0,
        )
    except Exception:  # noqa: BLE001 — observability must never break the app
        pass


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class ObservedSession:
    """A provider + optional dashboard binding for repeated tracing."""

    def __init__(
        self,
        provider: str = "mock",
        model: str = "",
        api_key: str | None = None,
        base_url: str | None = None,
        dashboard: str | None = None,
        top_k_features: int = 20,
        attribution_threshold: float = 0.05,
        **provider_kwargs: Any,
    ) -> None:
        self.provider_name = provider
        self.dashboard = dashboard
        self._provider = _build_provider(provider, model, api_key, base_url, **provider_kwargs)
        self._top_k = top_k_features
        self._threshold = attribution_threshold
        self.last_trace: TraceResult | None = None
        self.last_output: ProviderOutput | None = None

    @property
    def provider(self) -> BaseProvider:
        return self._provider

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        """Raw provider call (no trace)."""
        self.last_output = self._provider.run(prompt, **kwargs)
        return self.last_output

    def trace(self, prompt: str, run_probes: bool = False) -> TraceResult:
        """Full interpretability trace; streamed to the dashboard if configured."""
        from LLmThoughtLens.scope import Scope

        scope = Scope(
            self._provider,
            top_k_features=self._top_k,
            attribution_threshold=self._threshold,
        )
        result = scope.trace_full(prompt, run_probes=run_probes)
        self.last_trace = result
        _push(self.dashboard, "trace_complete", result.to_payload())
        return result

    def record(self, prompt: str, completion: str, **extra: Any) -> None:
        """Push a prompt→completion exchange you already have to the dashboard."""
        data = {
            "prompt": prompt,
            "completion": completion,
            "model": getattr(self._provider, "model_id", self.provider_name),
            "evidence_kind": getattr(self._provider, "evidence_kind", "black_box"),
            "next_token_distribution": [],
            **extra,
        }
        _push(self.dashboard, "proxy_exchange", data)


@contextmanager
def observe(
    provider: str = "mock",
    model: str = "",
    dashboard: str | None = None,
    **kwargs: Any,
) -> Iterator[ObservedSession]:
    """Context manager yielding an :class:`ObservedSession`.

    Example::

        with observe(provider="ollama", model="llama3.1:8b",
                     dashboard="http://localhost:8000") as obs:
            r = obs.trace("The capital of France is")
            print(r.output_token)
    """
    session = ObservedSession(provider=provider, model=model, dashboard=dashboard, **kwargs)
    yield session


def trace(
    prompt: str,
    provider: str = "mock",
    model: str = "",
    dashboard: str | None = None,
    run_probes: bool = False,
    **kwargs: Any,
) -> TraceResult:
    """One-shot: build a provider, trace *prompt*, return the :class:`TraceResult`."""
    session = ObservedSession(provider=provider, model=model, dashboard=dashboard, **kwargs)
    return session.trace(prompt, run_probes=run_probes)


def record_exchange(
    prompt: str,
    completion: str,
    dashboard: str,
    model: str = "",
    **extra: Any,
) -> None:
    """Push a single prompt→completion exchange to a running dashboard."""
    data = {
        "prompt": prompt,
        "completion": completion,
        "model": model,
        "next_token_distribution": [],
        **extra,
    }
    _push(dashboard, "proxy_exchange", data)


# ---------------------------------------------------------------------------
# Drop-in client wrapper (OpenAI-style)
# ---------------------------------------------------------------------------


class _ObservedCompletions:
    def __init__(self, real: Any, hook: Any) -> None:
        self._real = real
        self._hook = hook

    def create(self, **kwargs: Any) -> Any:
        resp = self._real.create(**kwargs)
        with contextlib.suppress(Exception):  # never break the caller's request
            self._hook(kwargs, resp)
        return resp


class _ObservedChat:
    def __init__(self, real: Any, hook: Any) -> None:
        self.completions = _ObservedCompletions(real.completions, hook)


class ObservedOpenAI:
    """Transparent wrapper over an OpenAI-style client.

    Intercepts ``chat.completions.create`` to emit a live exchange to the
    dashboard, then returns the real response unchanged.  All other attributes
    delegate to the wrapped client.
    """

    def __init__(self, client: Any, dashboard: str | None = None) -> None:
        self._client = client
        self._dashboard = dashboard
        self.chat = _ObservedChat(client.chat, self._on_exchange)

    def _on_exchange(self, kwargs: dict[str, Any], resp: Any) -> None:
        messages = kwargs.get("messages", [])
        prompt = "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}" for m in messages if isinstance(m, dict)
        )
        completion = ""
        try:
            completion = resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError):
            try:
                completion = resp["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError):
                completion = ""
        _push(
            self._dashboard,
            "proxy_exchange",
            {
                "prompt": prompt,
                "completion": completion,
                "model": kwargs.get("model", ""),
                "evidence_kind": "black_box",
                "next_token_distribution": [],
            },
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_openai(client: Any, dashboard: str | None = None) -> ObservedOpenAI:
    """Wrap an OpenAI-style client so every chat call streams to the dashboard."""
    return ObservedOpenAI(client, dashboard=dashboard)


# ---------------------------------------------------------------------------
# attach() — wire ThoughtLens into a model you already run (the "middle layer")
# ---------------------------------------------------------------------------


class ModelLens:
    """A live lens over a model **you already loaded** in your own process.

    This is the white-box version of the middle-layer idea: ThoughtLens reads
    your model's real hidden states and streams the X-ray (logit lens +
    activation grid + attention) to a running dashboard — like wiring Grafana
    to a service.  It works because your model is a real PyTorch object in your
    process, so its internals are directly observable (unlike an API model).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        dashboard: str | None = None,
        device: Any = None,
        model_label: str = "your-model",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dashboard = dashboard
        self.model_label = model_label
        if device is None:
            from LLmThoughtLens.xray_core import infer_device

            device = infer_device(model)
        self.device = device

    def xray(self, prompt: str, max_new_tokens: int = 12) -> dict[str, Any]:
        """Run the real logit-lens X-ray on your model and stream it live.

        Returns ``{"completion", "n_steps"}``.  Every event is pushed to the
        configured dashboard (if any) and is also returned for programmatic use.
        """
        from LLmThoughtLens.xray_core import run_xray_loop

        def emit(kind: str, data: dict[str, Any]) -> None:
            _push(self.dashboard, kind, data)

        return run_xray_loop(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            emit=emit,
            model_label=self.model_label,
        )


def attach(
    model: Any,
    tokenizer: Any = None,
    dashboard: str | None = None,
    device: Any = None,
    model_label: str = "your-model",
) -> ModelLens:
    """Wire ThoughtLens into a HuggingFace model you already have.

    Example — wire it once in your app, then watch it live in the dashboard::

        from LLmThoughtLens.sdk import attach
        lens = attach(my_model, my_tokenizer, dashboard="http://localhost:8000")
        lens.xray("The opposite of hot is")   # streams the logit lens live

    If *tokenizer* is omitted, ThoughtLens loads it from the model's
    ``name_or_path``.  Works on any local model — including your own trained
    weights — because it reads the model object's real activations.
    """
    if tokenizer is None:
        from transformers import AutoTokenizer

        name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None) or getattr(
            model, "name_or_path", None
        )
        if not name_or_path:
            raise ValueError(
                "could not infer a tokenizer; pass tokenizer=... explicitly to attach()"
            )
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    return ModelLens(model, tokenizer, dashboard=dashboard, device=device, model_label=model_label)


__all__ = [
    "ObservedSession",
    "ObservedOpenAI",
    "ModelLens",
    "observe",
    "trace",
    "record_exchange",
    "wrap_openai",
    "attach",
]
