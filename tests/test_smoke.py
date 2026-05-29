"""Smoke tests — verify the top-level public API imports and wires together."""

from __future__ import annotations

import llmscope
from llmscope import Scope, Feature, FeatureSet, AttributionGraph, BaseProbe, ProbeResult
from llmscope.providers import BaseProvider, ProviderOutput, MockProvider
from llmscope.providers.registry import get_provider, list_providers


class TestPublicAPI:
    def test_version_is_string(self):
        assert isinstance(llmscope.__version__, str)

    def test_all_exports_importable(self):
        for sym in [Scope, Feature, FeatureSet, AttributionGraph, BaseProbe, ProbeResult]:
            assert sym is not None

    def test_scope_from_mock(self):
        scope = Scope.from_mock(n_layers=2)
        assert repr(scope).startswith("Scope(")

    def test_scope_trace_returns_provider_output(self):
        scope = Scope.from_mock()
        out = scope.trace("hello")
        assert isinstance(out, ProviderOutput)
        assert out.activations is not None

    def test_scope_provider_property(self):
        scope = Scope.from_mock()
        assert isinstance(scope.provider, BaseProvider)

    def test_feature_dataclass(self):
        f = Feature(id=42, label="capital city", layer=3, score=0.95)
        assert f.id == 42
        assert f.score == 0.95

    def test_feature_set_top_k(self):
        fs = FeatureSet(name="test")
        for i, score in enumerate([0.1, 0.9, 0.5, 0.3, 0.7]):
            fs.add(Feature(id=i, score=score))
        top2 = fs.top(2)
        assert top2[0].score >= top2[1].score

    def test_attribution_graph_add_edge(self):
        g = AttributionGraph(name="test")
        g.add_edge(0, 1, weight=0.8)
        g.add_edge(1, 2, weight=0.3)
        assert g.num_nodes == 3
        assert g.num_edges == 2

    def test_attribution_graph_successors(self):
        g = AttributionGraph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        succs = g.successors(0)
        assert set(succs) == {1, 2}

    def test_registry_mock_available(self):
        assert "mock" in list_providers()

    def test_registry_get_mock(self):
        provider = get_provider("mock", seed=123)
        assert isinstance(provider, MockProvider)

    def test_registry_unknown_raises(self):
        import pytest

        with pytest.raises(KeyError):
            get_provider("does_not_exist")
