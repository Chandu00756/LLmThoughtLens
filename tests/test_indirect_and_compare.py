"""Tests for indirect (transitive) influence + before/after intervention reports."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from LLmThoughtLens.circuits.graph import AttributionGraph
from LLmThoughtLens.scope import Scope, compare_traces, save_intervention_report


class TestIndirectInfluence:
    def test_two_hop_product(self):
        g = AttributionGraph()
        g.add_edge(0, 1, weight=0.5)
        g.add_edge(1, 2, weight=0.4)
        # No direct 0->2 edge; indirect influence = 0.5 * 0.4 = 0.2.
        assert g.indirect_influence(0, 2) == pytest.approx(0.2)
        # Direct neighbour has no *indirect* path here.
        assert g.indirect_influence(0, 1) == pytest.approx(0.0)

    def test_sign_preserved(self):
        g = AttributionGraph()
        g.add_edge(0, 1, weight=0.5)
        g.add_edge(1, 2, weight=-0.6)  # suppressing second hop
        assert g.indirect_influence(0, 2) == pytest.approx(-0.3)

    def test_strongest_path_chosen(self):
        g = AttributionGraph()
        g.add_edge(0, 1, weight=0.9)
        g.add_edge(1, 3, weight=0.9)  # path A product 0.81
        g.add_edge(0, 2, weight=0.2)
        g.add_edge(2, 3, weight=0.2)  # path B product 0.04
        assert g.indirect_influence(0, 3) == pytest.approx(0.81)

    def test_indirect_edges_excludes_direct(self):
        g = AttributionGraph()
        g.add_edge(0, 1, weight=0.5)
        g.add_edge(1, 2, weight=0.5)
        g.add_edge(0, 2, weight=0.9)  # direct edge — must be excluded
        edges = g.indirect_edges(min_weight=0.1)
        pairs = {(s, d) for s, d, _ in edges}
        assert (0, 2) not in pairs  # has a direct edge


class TestBeforeAfterComparison:
    def test_compare_traces_detects_intervention_effect(self):
        scope = Scope.from_mock(n_layers=4, n_heads=2, d_model=16, seed=11)
        baseline = scope.trace_full("the capital of France is Paris", run_probes=False)

        from LLmThoughtLens.features.intervention import FeatureIntervention

        # Zero out token 0 across all dims/layers — a real causal change.
        interventions = [
            FeatureIntervention.clamp(feature_id=d, value=0.0, layer=-1, token_idx=0)
            for d in range(scope.provider.d_model)
        ]
        intervened = scope.trace_full(
            "the capital of France is Paris", interventions=interventions, run_probes=False
        )
        diff = compare_traces(baseline, intervened, threshold=0.01)
        # The intervention must change *something* in the graph.
        total_changes = (
            len(diff.added_nodes)
            + len(diff.removed_nodes)
            + len(diff.added_edges)
            + len(diff.removed_edges)
            + len(diff.changed_edges)
        )
        assert total_changes > 0, "intervention produced no measurable graph change"

    def test_save_intervention_report_writes_three_tabs(self):
        scope = Scope.from_mock(seed=3)
        baseline = scope.trace_full("hello world")
        intervened = scope.trace_full("hello world")  # same here; report still renders
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "intervention.html"
            diff = save_intervention_report(baseline, intervened, path, threshold=0.01)
            html = path.read_text()
            assert "tlp-baseline" in html
            assert "tlp-intervention" in html
            assert "tlp-diff" in html
            assert diff is not None
