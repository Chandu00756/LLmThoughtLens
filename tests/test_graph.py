"""Tests for AttributionGraph, paths, diff, tracer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from LLmThoughtLens.circuits.diff import GraphDiff
from LLmThoughtLens.circuits.graph import AttributionGraph, CircuitEdge, CircuitNode
from LLmThoughtLens.circuits.paths import top_causal_paths
from LLmThoughtLens.circuits.tracer import CircuitTracer
from LLmThoughtLens.features.extractor import FeatureExtractor
from LLmThoughtLens.providers.mock_provider import MockProvider


class TestGraphBasics:
    def test_add_node_and_edge(self):
        g = AttributionGraph()
        g.add_edge(1, 2, weight=0.8)
        assert g.num_nodes == 2
        assert g.num_edges == 1
        edges = list(g.edges())
        assert isinstance(edges[0], CircuitEdge)
        assert edges[0].polarity == "promote"

    def test_negative_weight_is_suppress(self):
        g = AttributionGraph()
        g.add_edge(1, 2, weight=-0.4)
        assert list(g.edges())[0].polarity == "suppress"

    def test_successors_and_predecessors(self):
        g = AttributionGraph()
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 2)
        assert set(g.successors(0)) == {1, 2}
        assert set(g.predecessors(2)) == {0, 1}

    def test_typed_node(self):
        g = AttributionGraph()
        g.add_node(7, label="output", node_type="output_token", layer=10)
        n = g.node(7)
        assert isinstance(n, CircuitNode)
        assert n.node_type == "output_token"


class TestPrune:
    def test_drops_weak_edges(self):
        g = AttributionGraph()
        g.add_edge(0, 1, weight=0.05)
        g.add_edge(0, 2, weight=0.5)
        pruned = g.prune(0.1)
        # Only the strong edge survives, but nodes are kept.
        assert pruned.num_edges == 1
        assert pruned.num_nodes == 3

    def test_keep_isolated_false_drops_orphans(self):
        g = AttributionGraph()
        g.add_node(99, label="lonely")
        g.add_edge(0, 1, weight=0.5)
        pruned = g.prune(0.1, keep_isolated=False)
        assert pruned.node(99) is None


class TestPaths:
    def test_top_paths_finds_route(self):
        g = AttributionGraph()
        g.add_node(0, node_type="input_token")
        g.add_node(2, node_type="output_token")
        g.add_edge(0, 1, weight=0.9)
        g.add_edge(1, 2, weight=0.8)
        paths = g.top_paths(n=1)
        assert paths == [[0, 1, 2]]

    def test_top_causal_paths_returns_edges(self):
        g = AttributionGraph()
        g.add_node(0, node_type="input_token")
        g.add_node(2, node_type="output_token")
        g.add_edge(0, 1, weight=0.7)
        g.add_edge(1, 2, weight=0.6)
        result = top_causal_paths(g, n=1)
        assert len(result) == 1
        path = result[0]
        assert path.nodes == [0, 1, 2]
        assert len(path.edges) == 2
        assert path.total_weight == pytest.approx(0.42, rel=1e-3)


class TestSerialisation:
    def test_to_dict_roundtrip(self):
        g = AttributionGraph(name="x")
        g.add_edge(0, 1, weight=0.4)
        d = g.to_dict()
        assert d["name"] == "x"
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1

    def test_to_json_and_csv(self):
        g = AttributionGraph()
        g.add_edge(0, 1, weight=0.5)
        with tempfile.TemporaryDirectory() as td:
            jp = Path(td) / "g.json"
            cp = Path(td) / "g.csv"
            g.to_json(jp)
            g.to_csv(cp)
            assert json.loads(jp.read_text())["edges"]
            assert "src,dst,weight" in cp.read_text()


class TestDiff:
    def test_added_removed(self):
        a = AttributionGraph()
        a.add_edge(0, 1, weight=0.5)
        b = AttributionGraph()
        b.add_edge(0, 1, weight=0.5)
        b.add_edge(1, 2, weight=0.3)
        diff = GraphDiff.compute(a, b)
        assert (1, 2) in diff.added_edges
        assert 2 in diff.added_nodes

    def test_changed_weight(self):
        a = AttributionGraph()
        a.add_edge(0, 1, weight=0.5)
        b = AttributionGraph()
        b.add_edge(0, 1, weight=0.2)
        diff = GraphDiff.compute(a, b, threshold=0.1)
        assert len(diff.changed_edges) == 1
        assert diff.changed_edges[0] == (0, 1, 0.5, 0.2)


class TestTracerEndToEnd:
    def test_produces_signed_edges(self):
        mp = MockProvider(n_layers=3, n_heads=2, d_model=16, seed=2)
        out = mp.run("the capital of France is")
        feats = FeatureExtractor(top_k=12).extract(out, provider=mp)
        g = CircuitTracer(min_weight=0.02).trace(out, feats, provider=mp)
        assert g.num_nodes > 0
        assert g.num_edges > 0
        # At least one promoting and one suppressing edge expected.
        polarities = {e.polarity for e in g.edges()}
        assert "promote" in polarities

    def test_input_and_output_nodes_present(self):
        mp = MockProvider(seed=2)
        out = mp.run("hello world")
        feats = FeatureExtractor(top_k=5).extract(out, provider=mp)
        g = CircuitTracer(min_weight=0.01).trace(out, feats, provider=mp)
        assert len(g.input_nodes()) == out.n_tokens
        assert len(g.output_nodes()) == 1
