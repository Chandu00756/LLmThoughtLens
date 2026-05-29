"""Anthropic provider — wraps anthropic>=0.20 Messages API."""

from __future__ import annotations

from llmscope.providers.base import BaseProvider, ProviderOutput


class AnthropicProvider(BaseProvider):
    """Provider backed by the Anthropic Messages API.

    Requires the ``anthropic`` extra: ``pip install 'llmscope[anthropic]'``.

    Parameters
    ----------
    api_key:
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY`` environment
        variable when ``None``.
    model:
        Model identifier, e.g. ``"claude-3-5-sonnet-20241022"``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-haiku-20241022",
    ) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' extra. "
                "Install with: pip install 'llmscope[anthropic]'"
            ) from exc
        self.model = model
        self._api_key = api_key

    def run(self, prompt: str, **kwargs) -> ProviderOutput:
        """Forward prompt to Anthropic and return a :class:`ProviderOutput`.

        Note: activation, attention, and logit tensors are not available via
        the Messages API and will be ``None``.
        """
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=kwargs.pop("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        content = response.content[0].text if response.content else ""
        tokens = content.split()
        return ProviderOutput(
            prompt=prompt,
            tokens=tokens,
            token_ids=[],
            meta={
                "provider": "AnthropicProvider",
                "model": self.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            },
        )

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"
