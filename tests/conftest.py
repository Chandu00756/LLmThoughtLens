"""Shared pytest fixtures for the LLmThoughtLens test suite."""

from __future__ import annotations

import pytest
from LLmThoughtLens.providers.base import ProviderOutput
from LLmThoughtLens.providers.mock_provider import MockProvider


@pytest.fixture(scope="session")
def mock_provider() -> MockProvider:
    """A session-scoped MockProvider with a fixed seed."""
    return MockProvider(n_layers=4, n_heads=2, d_model=16, seed=0)


@pytest.fixture(scope="session")
def simple_output(mock_provider: MockProvider) -> ProviderOutput:
    """A ProviderOutput produced by running 'Hello world' through the mock."""
    return mock_provider.run("Hello world")


@pytest.fixture
def fresh_provider() -> MockProvider:
    """A function-scoped MockProvider — new instance per test."""
    return MockProvider(n_layers=2, n_heads=1, d_model=8, seed=99)
