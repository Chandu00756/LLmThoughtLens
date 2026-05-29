"""Tests for the five-tab HTML report + visualisation components."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from LLmThoughtLens.circuits.tracer import CircuitTracer
from LLmThoughtLens.features.extractor import FeatureExtractor
from LLmThoughtLens.probes.builtin import all_probes
from LLmThoughtLens.probes.runner import ProbeRunner
from LLmThoughtLens.providers.mock_provider import MockProvider
from LLmThoughtLens.scope import TraceResult
from LLmThoughtLens.visualization.feature_browser import FeatureBrowser
from LLmThoughtLens.visualization.graph_viz import GraphVisualizer
from LLmThoughtLens.visualization.layer_stream import ResidualStreamView
from LLmThoughtLens.visualization.probe_dashboard import ProbeDashboard
from LLmThoughtLens.visualization.report import ReportBuilder
from LLmThoughtLens.visualization.token_heatmap import TokenHeatmap


@pytest.fixture(scope="module")
def trace_artefacts():
    mp = MockProvider(n_layers=3, n_heads=2, d_model=16, seed=8)
    out = mp.run("the capital of France is Paris")
    feats = FeatureExtractor(top_k=10).extract(out, provider=mp)
    g = CircuitTracer(min_weight=0.02).trace(out, feats, provider=mp)
    probe_report = ProbeRunner(all_probes()).run_all(mp)
    return TraceResult(
        prompt=out.prompt,
        output=out,
        features=feats,
        graph=g,
        probe_results=probe_report.results,
        meta={"provider": mp.name, "model": mp.model_id},
    )


class TestComponentRenderers:
    def test_token_heatmap_html(self, trace_artefacts):
        html = TokenHeatmap(trace_artefacts.output, trace_artefacts.features).to_html()
        assert "plotly" in html.lower()
        assert "heatmap" in html.lower()

    def test_graph_viz_html(self, trace_artefacts):
        html = GraphVisualizer(trace_artefacts.graph).to_html()
        assert "plotly" in html.lower()

    def test_residual_stream_html(self, trace_artefacts):
        html = ResidualStreamView(trace_artefacts.output).to_html()
        assert "plotly" in html.lower()

    def test_feature_browser_table(self, trace_artefacts):
        html = FeatureBrowser(trace_artefacts.features).to_html()
        assert "fb-table" in html
        assert "fb-controls" in html

    def test_probe_dashboard_radar(self, trace_artefacts):
        html = ProbeDashboard(trace_artefacts.probe_results).to_html()
        assert "scatterpolar" in html.lower() or "polar" in html.lower()
        assert "probe-row" in html


class TestReport:
    def test_report_has_all_five_tabs(self, trace_artefacts):
        builder = ReportBuilder.from_trace_result(trace_artefacts)
        html = builder.render()
        for tab in ("heatmap", "graph", "stream", "features", "probes"):
            assert f"tlp-{tab}" in html, f"missing tab tlp-{tab}"

    def test_report_is_self_contained_with_cdn_plotly(self, trace_artefacts):
        html = ReportBuilder.from_trace_result(trace_artefacts).render()
        assert "cdn.plot.ly" in html
        assert html.strip().endswith("</html>")

    def test_report_saves_to_disk(self, trace_artefacts):
        builder = ReportBuilder.from_trace_result(trace_artefacts)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "report.html"
            builder.save(path)
            assert path.read_text().startswith("<!DOCTYPE html>")
