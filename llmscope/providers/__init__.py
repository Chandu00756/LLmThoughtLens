"""llmscope.providers — model backend adapters."""

from llmscope.providers.base import BaseProvider, ProviderOutput
from llmscope.providers.mock_provider import MockProvider

__all__ = ["BaseProvider", "ProviderOutput", "MockProvider"]
