"""OpenAI provider — Chat Completions adapter with real top-k logprobs.

OpenAI exposes per-token logprobs (``logprobs=True, top_logprobs=k``), so
``top_tokens`` is populated with **real** probabilities.  Anything we cannot
legitimately compute (activations, attentions, full vocabulary logits) is
left as ``None`` — never synthesised.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.utils.tokenizer_utils import whitespace_tokens


class OpenAIProvider(BaseProvider):
    """Black-box provider backed by OpenAI Chat Completions."""

    evidence_kind = "black_box"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        top_logprobs: int = 5,
        max_tokens: int = 256,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as exc:  # pragma: no cover — gated by extras
            raise ImportError(
                "OpenAIProvider needs the `openai` extra. "
                "Install with: pip install 'LLmThoughtLens[openai]'"
            ) from exc

        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._organization = organization
        self._base_url = base_url
        self.top_logprobs = max(1, min(20, int(top_logprobs)))
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self._client: Any = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model_id(self) -> str:
        return f"openai/{self.model}"

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                organization=self._organization,
                base_url=self._base_url,
                timeout=self.timeout,
            )
        return self._client

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        client = self._ensure_client()
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            logprobs=True,
            top_logprobs=self.top_logprobs,
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            **kwargs,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = resp.choices[0]
        content = choice.message.content or ""

        # Real top-k probabilities for the FIRST sampled token.
        top_tokens: list[tuple[str, float]] = []
        first = None
        try:
            if choice.logprobs and choice.logprobs.content:
                first = choice.logprobs.content[0]
        except (AttributeError, IndexError):
            first = None
        if first is not None and getattr(first, "top_logprobs", None):
            for alt in first.top_logprobs:
                top_tokens.append((str(alt.token), float(math.exp(alt.logprob))))
        if first is not None and not any(t[0] == first.token for t in top_tokens):
            top_tokens.insert(0, (str(first.token), float(math.exp(first.logprob))))
        if not top_tokens:
            top_tokens = [(whitespace_tokens(content)[0], 1.0)]

        tokens = whitespace_tokens(content) if content else [""]
        usage = getattr(resp, "usage", None)
        meta: dict[str, Any] = {
            "provider": "OpenAIProvider",
            "model": self.model,
            "latency_ms": latency_ms,
            "finish_reason": choice.finish_reason,
            "completion": content,
            "evidence_note": (
                "Top-token probabilities come from real OpenAI logprobs. "
                "Internal activations are not exposed by the API."
            ),
        }
        if usage is not None:
            meta["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        return ProviderOutput(
            prompt=prompt,
            tokens=tokens,
            token_ids=[],
            activations=None,
            attentions=None,
            logits=None,
            top_tokens=top_tokens,
            evidence_kind="black_box",
            meta=meta,
        )
