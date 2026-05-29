"""OpenAI provider — wraps openai>=1.0 Chat Completions API."""

from __future__ import annotations

from llmscope.providers.base import BaseProvider, ProviderOutput


class OpenAIProvider(BaseProvider):
    """Provider backed by the OpenAI Chat Completions API.

    Requires the ``openai`` extra: ``pip install 'llmscope[openai]'``.

    Parameters
    ----------
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when ``None``.
    model:
        Model identifier, e.g. ``"gpt-4o"`` or ``"gpt-4o-mini"``.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OpenAI provider requires the 'openai' extra. "
                "Install with: pip install 'llmscope[openai]'"
            ) from exc
        self.model = model
        self._api_key = api_key

    def run(self, prompt: str, **kwargs) -> ProviderOutput:
        """Forward prompt to OpenAI and return a :class:`ProviderOutput`.

        Note: activation, attention, and logit tensors are not available via
        the Chat Completions API and will be ``None``.
        """
        import openai

        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = response.choices[0].message.content or ""
        tokens = content.split()
        return ProviderOutput(
            prompt=prompt,
            tokens=tokens,
            token_ids=[],
            meta={
                "provider": "OpenAIProvider",
                "model": self.model,
                "usage": response.usage.model_dump() if response.usage else {},
            },
        )

    @property
    def name(self) -> str:
        return f"openai/{self.model}"
