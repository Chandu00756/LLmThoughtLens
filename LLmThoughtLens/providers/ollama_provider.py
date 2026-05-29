"""Ollama provider — black-box adapter that calls a local Ollama HTTP server.

Ollama exposes ``/api/generate`` for completions but does not expose
activations, attentions, or token-level logprobs.  We populate ``top_tokens``
with the sampled completion's first surface token at probability 1.0 and
record ``meta['evidence_note']`` so the UI never claims internal observation.
"""

from __future__ import annotations

import time
from typing import Any

from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.utils.tokenizer_utils import whitespace_tokens


class OllamaProvider(BaseProvider):
    """Provider that calls a locally-running Ollama instance."""

    evidence_kind = "black_box"

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OllamaProvider needs the `ollama` extra. "
                "Install with: pip install 'LLmThoughtLens[ollama]'"
            ) from exc
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_id(self) -> str:
        return f"ollama/{self.model}"

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        import httpx

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text: str = data.get("response", "") or ""
        tokens = whitespace_tokens(text) if text else [""]
        top_tokens = [(tokens[0], 1.0)]

        meta: dict[str, Any] = {
            "provider": "OllamaProvider",
            "model": self.model,
            "latency_ms": latency_ms,
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "completion": text,
            "evidence_note": (
                "Ollama HTTP API does not expose activations or per-token logprobs; "
                "top-token probability shown is the sampled completion only."
            ),
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

    def ping(self) -> bool:
        """Lightweight health check used by the TUI provider-connect screen."""
        import httpx

        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False
