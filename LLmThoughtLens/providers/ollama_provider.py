"""Ollama provider — black-box adapter that calls a local Ollama HTTP server.

Ollama does not expose activations or attentions, so this is a black-box
backend.  However, Ollama >= 0.12 *does* return per-token ``logprobs`` from
``/api/generate`` when ``logprobs`` is requested, which lets us populate
``top_tokens`` with **real graded probabilities** rather than a single
sampled token at probability 1.0.  Graded probabilities are what make the
token-masking importance engine produce meaningful (non-saturated) causal
scores, so we request them by default and fall back gracefully to the
sampled-token behaviour on older servers that ignore the field.
"""

from __future__ import annotations

import math
import time
from typing import Any

from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.utils.tokenizer_utils import whitespace_tokens


class OllamaProvider(BaseProvider):
    """Provider that calls a locally-running Ollama instance.

    Parameters
    ----------
    model:
        Ollama model tag, e.g. ``"llama3.1:8b"``.
    base_url:
        Base URL of the Ollama server.
    timeout:
        Per-request timeout in seconds.
    top_logprobs:
        Number of alternative tokens to request per position (when the
        server supports logprobs).
    request_logprobs:
        When ``True`` (default) the provider asks Ollama for token
        logprobs and exposes them as real probabilities in ``top_tokens``.
    """

    evidence_kind = "black_box"

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        top_logprobs: int = 5,
        request_logprobs: bool = True,
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
        self.top_logprobs = max(1, min(20, int(top_logprobs)))
        self.request_logprobs = bool(request_logprobs)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_id(self) -> str:
        return f"ollama/{self.model}"

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }
        if self.request_logprobs:
            payload.setdefault("logprobs", True)
            payload.setdefault("top_logprobs", self.top_logprobs)

        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text: str = data.get("response", "") or ""
        logprobs = data.get("logprobs")

        top_tokens, used_logprobs = self._first_token_distribution(logprobs, text)
        tokens = whitespace_tokens(text) if text else [""]

        if used_logprobs:
            note = (
                "Ollama exposed real per-token logprobs; top-token "
                "probabilities are genuine model probabilities (black-box: "
                "no activations or attentions are available from this API)."
            )
        else:
            note = (
                "This Ollama server did not return logprobs; the top-token "
                "probability is the sampled completion only. Upgrade Ollama "
                "(>= 0.12) for graded probabilities."
            )

        meta: dict[str, Any] = {
            "provider": "OllamaProvider",
            "model": self.model,
            "latency_ms": latency_ms,
            "eval_count": data.get("eval_count"),
            "eval_duration": data.get("eval_duration"),
            "completion": text,
            "has_logprobs": used_logprobs,
            "evidence_note": note,
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

    # ------------------------------------------------------------------
    # logprobs parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _first_token_distribution(
        logprobs: Any,
        text: str,
    ) -> tuple[list[tuple[str, float]], bool]:
        """Return ``([(token, prob), …], used_logprobs)`` for the FIRST token.

        Ollama's ``logprobs`` is a list (one entry per generated token); each
        entry has ``token``, ``logprob`` and an optional ``top_logprobs``
        list of alternatives.  We expose the first position's distribution —
        that is the next-token prediction the masking engine compares against.
        """
        if isinstance(logprobs, list) and logprobs:
            first = logprobs[0]
            out: list[tuple[str, float]] = []
            alts = first.get("top_logprobs") if isinstance(first, dict) else None
            if isinstance(alts, list) and alts:
                for alt in alts:
                    tok = str(alt.get("token", ""))
                    lp = alt.get("logprob")
                    if lp is not None:
                        out.append((tok, float(math.exp(lp))))
            # Always include the sampled token itself if not already present.
            sampled_tok = str(first.get("token", ""))
            sampled_lp = first.get("logprob")
            if sampled_lp is not None and not any(t == sampled_tok for t, _ in out):
                out.insert(0, (sampled_tok, float(math.exp(sampled_lp))))
            if out:
                out.sort(key=lambda kv: kv[1], reverse=True)
                return out, True

        # Fallback: no logprobs — sampled completion at probability 1.0.
        fallback_tok = whitespace_tokens(text)[0] if text else ""
        return [(fallback_tok, 1.0)], False

    def ping(self) -> bool:
        """Lightweight health check used by the TUI provider-connect screen."""
        import httpx

        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:  # noqa: BLE001
            return False
