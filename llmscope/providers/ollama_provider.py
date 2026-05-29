"""Ollama provider — wraps the local Ollama HTTP API via httpx."""

from __future__ import annotations

from llmscope.providers.base import BaseProvider, ProviderOutput


class OllamaProvider(BaseProvider):
    """Provider that calls a locally-running Ollama instance.

    Requires the ``ollama`` extra: ``pip install 'llmscope[ollama]'``.

    Parameters
    ----------
    model:
        Ollama model tag, e.g. ``"llama3.2"`` or ``"phi3"``.
    base_url:
        Base URL of the Ollama server (default: ``http://localhost:11434``).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Ollama provider requires the 'ollama' extra. "
                "Install with: pip install 'llmscope[ollama]'"
            ) from exc
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def run(self, prompt: str, **kwargs) -> ProviderOutput:
        """Forward prompt to Ollama and return a :class:`ProviderOutput`.

        Note: activation, attention, and logit tensors are not exposed by the
        Ollama API and will be ``None``.
        """
        import httpx

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()

        response_text: str = data.get("response", "")
        tokens = response_text.split()
        return ProviderOutput(
            prompt=prompt,
            tokens=tokens,
            token_ids=[],
            meta={
                "provider": "OllamaProvider",
                "model": self.model,
                "eval_count": data.get("eval_count"),
                "eval_duration": data.get("eval_duration"),
            },
        )

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"
