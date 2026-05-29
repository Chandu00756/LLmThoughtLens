"""Anthropic provider — Messages API adapter.

The Anthropic public API does not expose token-level logprobs.  We populate
``top_tokens`` with the sampled completion's first surface token at
probability 1.0 and annotate ``meta['evidence_note']`` accordingly so the UI
never claims real probability distributions it cannot produce.
"""

from __future__ import annotations

import os
import time
from typing import Any

from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.utils.tokenizer_utils import whitespace_tokens


class AnthropicProvider(BaseProvider):
    """Black-box provider backed by the Anthropic Messages API."""

    evidence_kind = "black_box"

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        api_key: str | None = None,
        max_tokens: int = 256,
        timeout: float = 60.0,
    ) -> None:
        try:
            from anthropic import Anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover — gated by extras
            raise ImportError(
                "AnthropicProvider needs the `anthropic` extra. "
                "Install with: pip install 'LLmThoughtLens[anthropic]'"
            ) from exc

        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self._client: Any = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model_id(self) -> str:
        return f"anthropic/{self.model}"

    def _ensure_client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self._api_key, timeout=self.timeout)
        return self._client

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        client = self._ensure_client()
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=self.model,
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        tokens = whitespace_tokens(text) if text else [""]
        top_tokens: list[tuple[str, float]] = [(tokens[0], 1.0)]

        usage = getattr(resp, "usage", None)
        meta: dict[str, Any] = {
            "provider": "AnthropicProvider",
            "model": self.model,
            "latency_ms": latency_ms,
            "stop_reason": getattr(resp, "stop_reason", None),
            "completion": text,
            "evidence_note": (
                "Anthropic Messages API does not expose per-token logprobs; "
                "the top-token probability shown is the sampled completion only."
            ),
        }
        if usage is not None:
            meta["usage"] = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
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
