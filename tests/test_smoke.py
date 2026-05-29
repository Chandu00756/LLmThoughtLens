"""Smoke tests — verify the top-level public API imports and wires together."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import LLmThoughtLens
import pytest
from LLmThoughtLens import (
    AttributionGraph,
    BaseProbe,
    CircuitTracer,
    Feature,
    FeatureExtractor,
    FeatureIntervention,
    FeatureSet,
    GraphDiff,
    ProbeResult,
    ProbeRunner,
    ReportBuilder,
    Scope,
    SparseAutoencoder,
    SupernodeGrouper,
    TraceResult,
)
from LLmThoughtLens.providers import BaseProvider, MockProvider, ProviderOutput
from LLmThoughtLens.providers.registry import (
    available_providers,
    get_provider,
    list_providers,
)


class TestPublicAPI:
    def test_version_is_string(self):
        assert isinstance(LLmThoughtLens.__version__, str)
        assert LLmThoughtLens.__version__.count(".") >= 1

    def test_all_exports_importable(self):
        for sym in [
            Scope,
            TraceResult,
            Feature,
            FeatureSet,
            FeatureExtractor,
            FeatureIntervention,
            SparseAutoencoder,
            AttributionGraph,
            CircuitTracer,
            SupernodeGrouper,
            GraphDiff,
            BaseProbe,
            ProbeResult,
            ProbeRunner,
            ReportBuilder,
        ]:
            assert sym is not None

    def test_scope_from_mock(self):
        scope = Scope.from_mock(n_layers=2)
        assert repr(scope).startswith("Scope(")

    def test_scope_trace_returns_provider_output(self):
        scope = Scope.from_mock()
        out = scope.trace("hello")
        assert isinstance(out, ProviderOutput)
        assert out.activations is not None
        assert out.evidence_kind == "white_box"

    def test_scope_provider_property(self):
        scope = Scope.from_mock()
        assert isinstance(scope.provider, BaseProvider)

    def test_feature_dataclass(self):
        f = Feature(id=42, label="capital city", layer=3, score=0.95)
        assert f.id == 42
        assert f.score == 0.95
        assert f.node_type == "feature"
        assert f.evidence_kind == "white_box"

    def test_feature_set_top_k(self):
        fs = FeatureSet(name="test")
        for i, score in enumerate([0.1, 0.9, 0.5, 0.3, 0.7]):
            fs.add(Feature(id=i, score=score))
        top2 = fs.top(2)
        assert top2[0].score >= top2[1].score
        assert fs.total_score() == pytest.approx(2.5)

    def test_registry_mock_available(self):
        assert "mock" in list_providers()
        assert "mock" in available_providers()

    def test_registry_get_mock(self):
        provider = get_provider("mock", seed=123)
        assert isinstance(provider, MockProvider)

    def test_registry_unknown_raises(self):
        with pytest.raises(KeyError):
            get_provider("does_not_exist")


class TestEndToEnd:
    """Mock-only end-to-end pipeline — no API keys, no GPU."""

    def test_trace_full_pipeline(self):
        scope = Scope.from_mock(n_layers=3, n_heads=2, d_model=16, seed=4)
        result = scope.trace_full("The capital of France is", run_probes=False)
        assert isinstance(result, TraceResult)
        assert result.output_token  # not empty
        assert len(result.features) > 0
        assert result.graph.num_nodes > 0
        assert result.evidence_kind == "white_box"

    def test_save_html_report(self):
        scope = Scope.from_mock(seed=2)
        result = scope.trace_full("hi there")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "report.html"
            result.save(path)
            text = path.read_text()
            for needle in (
                "cdn.plot.ly",
                "tl-tab-btn",
                "tlp-heatmap",
                "tlp-graph",
                "tlp-stream",
                "tlp-features",
                "tlp-probes",
                "fb-table",
            ):
                assert needle in text, f"missing {needle!r} in report"
            assert text.strip().endswith("</html>")

    def test_graph_json_csv_export(self):
        scope = Scope.from_mock()
        result = scope.trace_full("hello world")
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "g.json"
            cp = Path(td) / "g.csv"
            fp = Path(td) / "f.csv"
            result.save_graph_json(jp)
            result.save_graph_csv(cp)
            result.save_features_csv(fp)
            assert jp.exists() and cp.exists() and fp.exists()
            graph = json.loads(jp.read_text())
            assert graph["nodes"]
            assert graph["edges"]

    def test_intervention_runs(self):
        scope = Scope.from_mock(seed=3)
        intervention = FeatureIntervention.inhibit(feature_id=0, scale=1.0, layer=1)
        result = scope.trace_full("the cat sat", interventions=[intervention])
        assert isinstance(result, TraceResult)
        assert result.output_token  # still produces a token
