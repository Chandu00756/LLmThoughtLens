"""Provider layer — adapters that expose a uniform :class:`ProviderOutput` envelope."""

from LLmThoughtLens.providers.base import BaseProvider, EvidenceKind, ProviderOutput
from LLmThoughtLens.providers.mock_provider import MockProvider
from LLmThoughtLens.providers.registry import (
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
