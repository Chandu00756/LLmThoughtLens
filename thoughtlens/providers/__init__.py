"""Provider layer — adapters that expose a uniform `ProviderOutput` envelope.

Each provider speaks one upstream system (OpenAI, Anthropic, HuggingFace,
Ollama, or a deterministic mock) and returns the same shape so the rest of
ThoughtLens never branches on backend identity.
"""

from thoughtlens.providers.base import BaseProvider, EvidenceKind, ProviderOutput
from thoughtlens.providers.mock_provider import MockProvider
from thoughtlens.providers.registry import (
    available_providers,
    get_provider,
    list_providers,
    register_provider,
)

__all__ = [
    "BaseProvider",
    "ProviderOutput",
    "EvidenceKind",
    "MockProvider",
    "available_providers",
    "list_providers",
    "register_provider",
    "get_provider",
]
