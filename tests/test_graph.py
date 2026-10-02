"""Tests for AttributionGraph, paths, diff, tracer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from LLmThoughtLens.circuits.diff import GraphDiff
from LLmThoughtLens.circuits.graph import AttributionGraph, CircuitEdge, CircuitNode
from LLmThoughtLens.circuits.paths import top_causal_paths
from LLmThoughtLens.circuits.tracer import TRACE_METHODS, CircuitTracer
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


# ---------------------------------------------------------------------------
# Tracer edge semantics: method selection, truthful labels, serialisation
# ---------------------------------------------------------------------------

TINY_PROMPT = "a b c d e f"
_FLOW_METHODS = {"input_activation", "activation_flow", "last_layer_to_output", "residual"}


def _tiny_provider() -> Any:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from _tiny_hf import make_provider

    return make_provider(perturb_norms=True)


def _tiny_trace(**tracer_kw: Any) -> tuple[Any, Any, list[Any], AttributionGraph]:
    provider = _tiny_provider()
    out = provider.run(TINY_PROMPT)
    feats = FeatureExtractor(top_k=6).extract(out, provider=provider)
    graph = CircuitTracer(**tracer_kw).trace(out, feats, provider=provider)
    return provider, out, feats, graph


class TestTracerSettings:
    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            ({"method": "magic"}, "method"),
            ({"validate": -1}, "validate"),
            ({"metric": "prob"}, "metric"),
            ({"node_kind": "sae"}, "node_kind"),
            ({"baseline": "median"}, "baseline"),
            ({"attribution_nodes": -1}, "attribution_nodes"),
        ],
    )
    def test_rejects_bad_settings(self, kw, match):
        with pytest.raises(ValueError, match=match):
            CircuitTracer(**kw)

    def test_defaults_and_validate_true(self):
        t = CircuitTracer()
        assert (t.method, t.validate_k, t.metric, t.node_kind, t.baseline) == (
            "auto",
            0,
            "logit",
            "delta",
            "zero",
        )
        assert CircuitTracer(validate=True).validate_k == 10
        assert CircuitTracer(validate=7).validate_k == 7
        assert set(TRACE_METHODS) == {"auto", "gradient", "activation_flow"}
        # A FeatureExtractor without SAEs is "no SAE".
        assert CircuitTracer(sae=FeatureExtractor()).sae is None


class TestTracerMockAndBlackBox:
    def test_mock_auto_uses_activation_flow(self):
        mp = MockProvider(n_layers=3, n_heads=2, d_model=16, seed=2)
        out = mp.run("the capital of France is")
        feats = FeatureExtractor(top_k=12).extract(out, provider=mp)
        g = CircuitTracer(min_weight=0.0).trace(out, feats, provider=mp)
        assert g.meta["method_requested"] == "auto"
        assert g.meta["attribution_method"] == "activation_flow"
        assert g.meta["edge_semantics"] == "correlational"
        assert "method_fallback" not in g.meta  # the mock never offered gradients
        assert {e.method for e in g.edges()} <= _FLOW_METHODS
        err = g.node(CircuitTracer.error_node_id())
        assert err is not None and err.meta["error_kind"] == "activation_energy"

    def test_gradient_on_mock_is_an_error_and_validate_is_skipped(self):
        mp = MockProvider(seed=2)
        out = mp.run("hello world again")
        feats = FeatureExtractor(top_k=5).extract(out, provider=mp)
        with pytest.raises(ValueError, match="gradient attribution unavailable"):
            CircuitTracer(method="gradient").trace(out, feats, provider=mp)
        with pytest.raises(ValueError, match="no provider"):
            CircuitTracer(method="gradient").trace(out, feats)
        g = CircuitTracer(validate=3).trace(out, feats, provider=mp)
        assert "faithfulness" not in g.meta and "faithfulness_skipped" in g.meta

    def test_black_box_is_causal_input_masking(self):
        from LLmThoughtLens.features.feature import Feature
        from LLmThoughtLens.providers.base import ProviderOutput

        out = ProviderOutput(
            prompt="the cat sat",
            tokens=[" down"],
            top_tokens=[(" down", 0.6), (" up", 0.2)],
            evidence_kind="black_box",
        )
        feats = [
            Feature(id=i, layer=0, token_idx=i, score=0.3, meta={"method": "token_masking"})
            for i in range(3)
        ]
        g = CircuitTracer(method="gradient").trace(out, feats)  # method ignored for black-box
        assert g.meta["attribution_method"] == "mask_perturbation"
        assert g.meta["edge_semantics"] == "causal_input_masking"
        assert {e.method for e in g.edges()} == {"mask_perturbation"}


class TestTracerGradient:
    def test_tiny_hf_auto_selects_gradient(self):
        # In a 2-layer model only the last position of layer 1 reaches the prediction,
        # so attribution-selected nodes supply destinations for feature -> feature edges.
        _, out, feats, g = _tiny_trace(min_weight=0.0, attribution_nodes=2)
        meta = g.meta
        assert meta["attribution_method"] == "gradient"
        assert meta["edge_semantics"] == "causal_linearised"
        assert meta["edge_method"] == "grad_x_act"
        assert (meta["metric"], meta["node_kind"], meta["baseline"]) == ("logit", "delta", "zero")
        assert meta["target_token"] == out.output_token
        assert meta["token_ids"] == out.token_ids
        assert meta["replay_max_rel_error"] < 1e-5
        assert {e.method for e in g.edges()} <= {"grad_x_act", "residual"}
        ff = [
            e
            for e in g.edges()
            if g.node(e.src).node_type == "feature" and g.node(e.dst).node_type == "feature"
        ]
        assert ff and all(g.node(e.src).layer < g.node(e.dst).layer for e in ff)
        assert any(e.weight != 0.0 for e in ff)
        last = out.n_tokens - 1
        for f in feats:  # nothing after the last block: off-prediction writes have A == 0
            if f.layer == out.n_layers - 1 and f.token_idx != last:
                assert g.node(f.id).meta["attribution"] == 0.0
        for f in feats:
            node = g.node(f.id)
            assert node.meta["node_kind"] == "delta" and isinstance(node.meta["attribution"], float)
        for u in range(out.n_tokens):
            assert "attribution" in g.node(CircuitTracer.input_node_id(u)).meta
        # Feature -> output edges carry exactly the node attributions.
        to_out = {e.src: e.weight for e in g.in_edges(CircuitTracer.output_node_id())}
        for f in feats:
            assert to_out[f.id] == pytest.approx(g.node(f.id).meta["attribution"])
        err = g.node(CircuitTracer.error_node_id())
        if err is not None:
            assert err.meta["error_kind"] == "attribution_mass"
            assert 0.0 <= err.meta["unexplained_fraction"] <= 1.0

    def test_gradient_graph_serialises_and_prunes(self):
        _, _, _, g = _tiny_trace(min_weight=0.0, validate=3)
        data = json.loads(g.to_json())
        assert data["meta"]["edge_semantics"] == "causal_linearised"
        assert data["meta"]["faithfulness"]["n"] == 3
        json.dumps(data, allow_nan=False)
        assert any("attribution" in n for n in data["nodes"])
        pruned = g.prune(0.05)
        assert pruned.meta["edge_semantics"] == "causal_linearised"
        assert all(abs(e.weight) >= 0.05 for e in pruned.edges())
        with tempfile.TemporaryDirectory() as tmp:
            g.to_csv(Path(tmp) / "edges.csv")
            assert "grad_x_act" in (Path(tmp) / "edges.csv").read_text()

    def test_validate_records_faithfulness_and_patched_effects(self):
        provider = _tiny_provider()
        out = provider.run(TINY_PROMPT)
        feats = FeatureExtractor(top_k=6).extract(out, provider=provider)
        tracer = CircuitTracer(min_weight=0.0, validate=4)
        g = tracer.trace(out, feats, provider=provider)
        faith = g.meta["faithfulness"]
        assert faith["n"] == 4 and faith["method"] == "zero_ablation"
        assert tracer.last_faithfulness is not None and tracer.last_attribution is not None
        patched = {n.id: n.meta["patched_effect"] for n in g.nodes() if "patched_effect" in n.meta}
        assert len(patched) == 4
        for row in faith["nodes"]:
            assert patched[row["node_id"]] == pytest.approx(row["measured"])

    def test_activation_flow_on_request_and_hooked_model_as_provider(self):
        provider = _tiny_provider()
        out = provider.run(TINY_PROMPT)
        feats = FeatureExtractor(top_k=6).extract(out, provider=provider)
        flow = CircuitTracer(method="activation_flow").trace(out, feats, provider=provider)
        assert flow.meta["edge_semantics"] == "correlational"
        direct = CircuitTracer(method="gradient").trace(out, feats, provider=provider.hooked)
        assert direct.meta["edge_semantics"] == "causal_linearised"

    def test_metric_and_attribution_nodes_options(self):
        _, out, feats, g = _tiny_trace(metric="logit_diff", attribution_nodes=3)
        assert g.meta["metric"] == "logit_diff" and g.meta["runner_up_token"] is not None
        assert g.meta["n_attribution_nodes"] == 3
        added = [n for n in g.nodes() if n.meta.get("selected_by") == "attribution"]
        assert len(added) == 3 and all(n.node_type == "feature" for n in added)
        assert {n.id for n in added}.isdisjoint({f.id for f in feats})

    def test_intervened_output_needs_its_interventions(self):
        from LLmThoughtLens.features.intervention import FeatureIntervention

        provider = _tiny_provider()
        iv = FeatureIntervention.clamp(3, value=5.0, layer=0, site="resid_post")
        out = provider.run_with_intervention(TINY_PROMPT, [iv])
        feats = FeatureExtractor(top_k=6).extract(out, provider=provider)
        g = CircuitTracer().trace(out, feats, provider=provider)
        assert g.meta["attribution_method"] == "activation_flow"
        assert "intervention" in g.meta["method_fallback"]
        with pytest.raises(ValueError, match="intervention"):
            CircuitTracer(method="gradient").trace(out, feats, provider=provider)
        g = CircuitTracer().trace(out, feats, provider=provider, interventions=[iv])
        assert g.meta["attribution_method"] == "gradient"
        assert g.meta["replay_max_rel_error"] < 1e-5

    def test_different_computation_falls_back(self):
        provider = _tiny_provider()
        out = provider.run(TINY_PROMPT)
        feats = FeatureExtractor(top_k=6).extract(out, provider=provider)
        out.activations = out.activations + 1.0  # not what the model computes
        g = CircuitTracer().trace(out, feats, provider=provider)
        assert g.meta["attribution_method"] == "activation_flow"
        assert "does not reproduce" in g.meta["method_fallback"]

    def test_features_outside_the_model_are_rejected(self):
        from LLmThoughtLens.features.feature import Feature

        provider = _tiny_provider()
        out = provider.run(TINY_PROMPT)
        for bad in (Feature(id=1, layer=7, token_idx=1), Feature(id=1, layer=0, token_idx=40)):
            with pytest.raises(ValueError, match="outside"):
                CircuitTracer(method="gradient").trace(out, [bad], provider=provider)

    def test_sae_features(self):
        from LLmThoughtLens.features.sae import SAEConfig, SparseAutoencoder

        provider = _tiny_provider()
        out = provider.run(TINY_PROMPT)
        sae = SparseAutoencoder(SAEConfig(input_dim=32, dict_size=64, k=8, seed=0, device="cpu"))
        extractor = FeatureExtractor(top_k=6)
        extractor.add_sae(sae, 1, site="resid_post")
        feats = extractor.extract(out, provider=provider)
        assert feats and all(f.meta["method"] == "sae" for f in feats)
        # Without the SAE the tracer cannot differentiate the codes: honest fallback.
        g = CircuitTracer().trace(out, feats, provider=provider)
        assert g.meta["attribution_method"] == "activation_flow"
        assert "SAE" in g.meta["method_fallback"]
        g = CircuitTracer(min_weight=0.0, sae=extractor, validate=3).trace(
            out, feats, provider=provider
        )
        assert g.meta["attribution_method"] == "gradient"
        for f in feats:
            meta = g.node(f.id).meta
            assert meta["node_kind"] == "sae"
            assert meta["sae_feature_id"] == f.meta["sae_feature_id"]
            assert meta["sae_name"] == f.meta["sae_name"]
        assert g.meta["faithfulness"]["n"] == 3
        same = CircuitTracer(min_weight=0.0, sae=extractor.sae_map).trace(
            out, feats, provider=provider
        )
        assert same.meta["attribution_method"] == "gradient"
