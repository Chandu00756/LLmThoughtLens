"""Score-scale regressions in the visualisation layer + Scope-level extractor options.

The extractor's default white-box score is unitless (``scoring="centered"``)
and the massive-activation "attention sink" position is excluded from the
ranking.  These tests pin that every renderer respects that: no raw-norm /
unitless mixing, no cross-node-type score ranking, and an honest, visible
record of excluded positions.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pytest
from LLmThoughtLens.circuits.graph import AttributionGraph
from LLmThoughtLens.circuits.tracer import CircuitTracer
from LLmThoughtLens.features.extractor import (
    EXCLUSION_REASON_OUTLIER,
    EXCLUSION_REASON_REQUESTED,
    EXCLUSION_REASON_UNKNOWN,
    FeatureExtractor,
    exclusion_reasons,
)
from LLmThoughtLens.features.feature import Feature
from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.providers.mock_provider import MockProvider
from LLmThoughtLens.scope import Scope, TraceResult
from LLmThoughtLens.visualization.feature_browser import FeatureBrowser
from LLmThoughtLens.visualization.graph_viz import GraphVisualizer
from LLmThoughtLens.visualization.layer_stream import ResidualStreamView
from LLmThoughtLens.visualization.report import ReportBuilder
from LLmThoughtLens.visualization.token_heatmap import TokenHeatmap

_SINK = 0
_CONTENT = 4


# ---------------------------------------------------------------------------
# Synthetic white-box data with a GPT-2-style attention sink
# ---------------------------------------------------------------------------


def _sink_activations(n_layers: int = 6, n_tokens: int = 7, d_model: int = 32) -> np.ndarray:
    """Position 0 is ~40x every other token at layers 1..L-1; position 4 carries content."""
    rng = np.random.default_rng(0)
    acts = rng.standard_normal((n_layers, n_tokens, d_model))
    acts += 2.0 * rng.standard_normal(d_model)[None, None, :]
    direction = rng.standard_normal(d_model)
    direction /= np.linalg.norm(direction)
    acts[:, _CONTENT] += 8.0 * direction
    acts[1:, _SINK] *= 40.0
    return acts.astype(np.float32)


def _sink_output(tokens: list[str] | None = None) -> ProviderOutput:
    acts = _sink_activations(n_tokens=len(tokens) if tokens else 7)
    toks = tokens or [f"t{i}" for i in range(acts.shape[1])]
    return ProviderOutput(
        prompt=" ".join(toks),
        tokens=toks,
        token_ids=list(range(len(toks))),
        activations=acts,
        top_tokens=[("next", 0.5), ("other", 0.2)],
        evidence_kind="white_box",
    )


class _SinkProvider(BaseProvider):
    """White-box provider that always returns the sink activations."""

    evidence_kind = "white_box"

    @property
    def name(self) -> str:
        return "sink_test"

    def run(self, prompt: str, **_: Any) -> ProviderOutput:
        out = _sink_output()
        out.prompt = prompt
        return out


class _FakeSAE:
    """Duck-typed SAE (encode + labels) so the SAE path runs without torch."""

    labels: dict[int, str] = {}

    def encode(self, x: np.ndarray) -> np.ndarray:
        codes = np.zeros((x.shape[0], 4), dtype=np.float32)
        codes[:, 1] = np.abs(x[:, 0]) + 0.1
        return codes

    def feature_direction(self, fid: int) -> np.ndarray:
        direction = np.zeros(32, dtype=np.float32)
        direction[fid % 32] = 1.0
        return direction


def _feature_hover(fig, trace_idx: int, pos: int) -> str:
    return str(fig.data[trace_idx].hovertext[0][pos])


# ---------------------------------------------------------------------------
# exclusion_reasons
# ---------------------------------------------------------------------------


class TestExclusionReasons:
    def test_outlier_vs_requested(self):
        out = _sink_output()
        feats = FeatureExtractor(top_k=8, exclude_positions=[2]).extract(out)
        assert exclusion_reasons(feats) == {
            _SINK: EXCLUSION_REASON_OUTLIER,
            2: EXCLUSION_REASON_REQUESTED,
        }

    def test_graph_meta_only_positions_get_generic_reason(self):
        assert exclusion_reasons([], [3, 1]) == {
            1: EXCLUSION_REASON_UNKNOWN,
            3: EXCLUSION_REASON_UNKNOWN,
        }

    def test_black_box_and_mock_traces_exclude_nothing(self):
        mp = MockProvider(n_layers=3, n_heads=2, d_model=16, seed=1)
        out = mp.run("hello world test")
        assert exclusion_reasons(FeatureExtractor(top_k=5).extract(out)) == {}


# ---------------------------------------------------------------------------
# TokenHeatmap
# ---------------------------------------------------------------------------


class TestTokenHeatmap:
    def test_sink_is_not_hottest_and_is_flagged(self):
        out = _sink_output()
        feats = FeatureExtractor(top_k=10).extract(out)
        hm = TokenHeatmap(out, feats)
        scores = hm.token_scores()
        assert len(scores) == len(out.tokens)
        assert scores[_SINK] == 0.0
        hottest = max(range(len(scores)), key=scores.__getitem__)
        assert hottest == _CONTENT
        assert hm.excluded_positions() == {_SINK: EXCLUSION_REASON_OUTLIER}

    def test_scores_are_feature_sums_not_raw_norms(self):
        out = _sink_output()
        feats = FeatureExtractor(top_k=10).extract(out)
        scores = TokenHeatmap(out, feats).token_scores()
        expected = [0.0] * len(out.tokens)
        for f in feats:
            expected[f.token_idx] += max(0.0, f.score)
        assert scores == pytest.approx(expected)
        # Centred scores are O(1); raw norms of this data are O(10-1000).
        assert max(scores) < 10 * len(feats)

    def test_excluded_cell_is_grey_with_hover_note(self):
        out = _sink_output()
        feats = FeatureExtractor(top_k=10).extract(out)
        fig = TokenHeatmap(out, feats).to_figure()
        assert len(fig.data) == 2
        main, excl = fig.data
        assert main.z[0][_SINK] is None  # not silently hot, not silently zero
        assert excl.z[0][_SINK] == 1.0
        assert all(v is None for i, v in enumerate(excl.z[0]) if i != _SINK)
        hover = _feature_hover(fig, 1, _SINK)
        assert f"excluded: {EXCLUSION_REASON_OUTLIER}" in hover
        assert "mean raw ‖h‖" in hover
        assert "grey" in fig.layout.title.text
        # The colour of every kept cell is normalised to [0, 1].
        kept = [v for v in main.z[0] if v is not None]
        assert max(kept) == pytest.approx(1.0)

    def test_l2_scoring_still_renders_in_one_unit(self):
        out = _sink_output()
        feats = FeatureExtractor(top_k=10, scoring="l2").extract(out)
        hm = TokenHeatmap(out, feats)
        assert hm.excluded_positions() == {}
        assert hm.score_unit() == "Σ raw ‖h‖"
        scores = hm.token_scores()
        # Legacy l2 ranks by raw norm, so the sink legitimately wins there.
        assert max(range(len(scores)), key=scores.__getitem__) == _SINK
        fig = hm.to_figure()
        assert len(fig.data) == 1
        assert "<" not in fig.layout.title.text

    def test_requested_exclusion_and_graph_meta_exclusion(self):
        out = _sink_output()
        feats = FeatureExtractor(top_k=10, exclude_positions=[-1]).extract(out)
        last = len(out.tokens) - 1
        assert TokenHeatmap(out, feats).excluded_positions() == {
            _SINK: EXCLUSION_REASON_OUTLIER,
            last: EXCLUSION_REASON_REQUESTED,
        }
        # Features without the meta record (e.g. hand-built) + graph meta.
        bare = [Feature(id=1, label="x", layer=1, score=0.5, token_idx=2)]
        hm = TokenHeatmap(out, bare, excluded_positions=[_SINK, 99])
        assert hm.excluded_positions() == {_SINK: EXCLUSION_REASON_UNKNOWN}

    def test_repeated_tokens_keep_separate_columns(self):
        out = _sink_output(tokens=["A", " the", " cat", " the", " dog", " the", " end"])
        feats = FeatureExtractor(top_k=10).extract(out)
        fig = TokenHeatmap(out, feats).to_figure()
        assert list(fig.data[0].x) == list(range(7))
        assert list(fig.layout.xaxis.ticktext) == out.tokens

    def test_black_box_heatmap_indexes_prompt_tokens(self):
        out = ProviderOutput(
            prompt="alpha beta gamma",
            tokens=["completion"],  # black-box tokens are the completion
            token_ids=[],
            top_tokens=[("x", 0.4)],
            evidence_kind="black_box",
        )
        feats = [
            Feature(
                id=i,
                label=f"token:{t}",
                layer=0,
                score=s,
                token_idx=i,
                node_type="input_token",
                evidence_kind="black_box",
                meta={"method": "token_masking"},
            )
            for i, (t, s) in enumerate([("alpha", 0.1), ("beta", 0.3), ("gamma", -0.2)])
        ]
        hm = TokenHeatmap(out, feats)
        assert hm.input_tokens() == ["alpha", "beta", "gamma"]
        assert hm.token_scores() == pytest.approx([0.1, 0.3, 0.0])
        assert hm.score_unit() == "Σ Δp (masking)"
        hm.to_html()

    def test_empty_inputs_render(self):
        out = ProviderOutput(prompt="", tokens=[], token_ids=[], evidence_kind="black_box")
        # Same placeholder input the tracer uses for an empty black-box prompt.
        assert TokenHeatmap(out, []).input_tokens() == ["<empty>"]
        assert TokenHeatmap(out, []).token_scores() == [0.0]
        assert "plotly" in TokenHeatmap(out, []).to_html().lower()
        wb = ProviderOutput(
            prompt="",
            tokens=[],
            token_ids=[],
            activations=np.zeros((2, 0, 4), dtype=np.float32),
            evidence_kind="white_box",
        )
        assert TokenHeatmap(wb, []).token_scores() == []
        assert "plotly" in TokenHeatmap(wb, []).to_html().lower()


# ---------------------------------------------------------------------------
# GraphVisualizer node selection
# ---------------------------------------------------------------------------


def _crowded_graph(n_inputs: int = 57, n_features: int = 20) -> AttributionGraph:
    """Input tokens at 1.0 outrank every centred feature (~0.8) on raw |score|."""
    g = AttributionGraph(name="crowded")
    for i in range(n_inputs):
        g.add_node(1_000_000_000 + i, label=f"in{i}", node_type="input_token", layer=-1,
                   token_idx=i, score=1.0)  # fmt: skip
    for j in range(n_features):
        tok = 1 + (j % 5)
        g.add_node(j, label=f"f{j}", node_type="feature", layer=j % 4, token_idx=tok,
                   score=0.9 - 0.01 * j)  # fmt: skip
    g.add_node(2_000_000_000, label="out", node_type="output_token", layer=5,
               token_idx=n_inputs - 1, score=0.4)  # fmt: skip
    g.add_node(3_000_000_000, label="error residual", node_type="error", layer=5,
               token_idx=0, score=1.2e6, unexplained_fraction=0.9)  # fmt: skip
    # Inputs 10 and 30 feed features directly; input 40 feeds nothing kept.
    g.add_edge(1_000_000_010, 0, weight=5.0, method="input_activation")
    g.add_edge(1_000_000_030, 1, weight=9.0, method="input_activation")
    g.add_edge(3_000_000_000, 2_000_000_000, weight=1000.0, method="residual")
    g.meta["excluded_positions"] = [0]
    return g


def _kept_types(gv: GraphVisualizer) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in gv.select_nodes():
        counts[n.node_type] = counts.get(n.node_type, 0) + 1
    return counts


class TestGraphVisualizerSelection:
    def test_features_output_error_kept_before_inputs(self):
        gv = GraphVisualizer(_crowded_graph(), max_nodes=30)
        assert _kept_types(gv) == {"output_token": 1, "error": 1, "feature": 20, "input_token": 8}
        assert set(gv._layout()) == {n.id for n in gv.select_nodes()}

    def test_inputs_with_edges_into_kept_nodes_come_first(self):
        gv = GraphVisualizer(_crowded_graph(), max_nodes=24)
        inputs = [n for n in gv.select_nodes() if n.node_type == "input_token"]
        # Two slots: the edge-connected inputs, strongest flow first.
        assert [n.token_idx for n in inputs] == [30, 10]
        gv = GraphVisualizer(_crowded_graph(), max_nodes=26)
        inputs = [n.token_idx for n in gv.select_nodes() if n.node_type == "input_token"]
        # Then inputs at a kept feature's token position (1..5), in prompt order.
        assert inputs == [30, 10, 1, 2]

    def test_tiny_budget_keeps_output_error_and_top_features(self):
        gv = GraphVisualizer(_crowded_graph(), max_nodes=5)
        kept = gv.select_nodes()
        assert _kept_types(gv) == {"output_token": 1, "error": 1, "feature": 3}
        assert sorted(n.label for n in kept if n.node_type == "feature") == ["f0", "f1", "f2"]
        # Output + error are kept even when the budget is smaller than them.
        assert _kept_types(GraphVisualizer(_crowded_graph(), max_nodes=1)) == {
            "output_token": 1,
            "error": 1,
        }

    def test_no_budget_or_small_graph_keeps_everything(self):
        g = _crowded_graph()
        n_total = sum(1 for _ in g.nodes())
        assert len(GraphVisualizer(g, max_nodes=None).select_nodes()) == n_total
        assert len(GraphVisualizer(g, max_nodes=n_total).select_nodes()) == n_total

    def test_error_node_never_dominates_scales(self):
        fig = GraphVisualizer(_crowded_graph(), max_nodes=30).to_figure()
        node_traces = [t for t in fig.data if t.mode == "markers+text"]
        sizes = {t.marker.size for t in node_traces}
        assert sizes == {14}  # fixed size: no score-proportional scaling across types
        err = next(t for t in node_traces if t.name == "error")
        assert "residual energy" in err.hovertext[0]
        assert "unexplained: 90.0%" in err.hovertext[0]
        assert "score: 1200000" not in err.hovertext[0]

    def test_excluded_input_token_is_greyed_with_reason(self):
        g = _crowded_graph()
        gv = GraphVisualizer(g, max_nodes=None)
        fig = gv.to_figure()
        inputs = next(t for t in fig.data if t.name == "input_token")
        colours = list(inputs.marker.color)
        assert len(set(colours)) == 2
        idx = [i for i, h in enumerate(inputs.hovertext) if "excluded:" in h]
        assert len(idx) == 1
        assert EXCLUSION_REASON_UNKNOWN in inputs.hovertext[idx[0]]
        gv2 = GraphVisualizer(g, max_nodes=None, excluded={0: EXCLUSION_REASON_OUTLIER})
        assert gv2.excluded == {0: EXCLUSION_REASON_OUTLIER}

    def test_real_mock_trace_keeps_every_feature(self):
        mp = MockProvider(n_layers=4, n_heads=2, d_model=16, seed=3)
        prompt = " ".join(f"w{i}" for i in range(45))
        out = mp.run(prompt)
        feats = FeatureExtractor(top_k=20).extract(out, provider=mp)
        graph = CircuitTracer(min_weight=0.0).trace(out, feats, provider=mp)
        gv = GraphVisualizer(graph, max_nodes=30)
        kept = _kept_types(gv)
        assert kept["feature"] == len(feats) == 20
        assert kept["output_token"] == 1
        assert sum(kept.values()) <= 30 + 2


# ---------------------------------------------------------------------------
# Residual stream, feature browser, report
# ---------------------------------------------------------------------------


class TestResidualStreamView:
    def test_sink_left_out_of_default_focus(self):
        view = ResidualStreamView(_sink_output())
        assert _SINK not in view.focus_tokens
        assert view.skipped_positions == [_SINK]
        assert "left out" in view.to_figure().layout.title.text

    def test_explicit_focus_and_exclusions(self):
        out = _sink_output()
        assert ResidualStreamView(out, focus_tokens=[0, 1]).focus_tokens == [0, 1]
        view = ResidualStreamView(out, exclude_positions=[1, -1])
        assert view.skipped_positions == [0, 1, 6]
        assert view.focus_tokens == [2, 3, 4, 5]

    def test_mock_output_unchanged(self, simple_output):
        view = ResidualStreamView(simple_output)
        assert view.focus_tokens == list(range(simple_output.n_tokens))[:6]
        assert view.skipped_positions == []


class TestFeatureBrowserUnits:
    def test_raw_norm_column_for_white_box(self):
        out = _sink_output()
        feats = FeatureExtractor(top_k=5).extract(out)
        html = FeatureBrowser(feats).to_html()
        assert "Raw ‖h‖" in html
        assert html.count("data-raw=") == 5
        assert "unitless" in html

    def test_no_raw_column_without_raw_norm(self):
        feats = [Feature(id=1, label="a", layer=0, score=0.3, token_idx=0)]
        html = FeatureBrowser(feats).to_html()
        assert "Raw ‖h‖" not in html
        assert "key === 'raw'" in FeatureBrowser.js()


def _sink_trace(**scope_kwargs: Any) -> TraceResult:
    return Scope(_SinkProvider(), top_k_features=10, **scope_kwargs).trace_full("t0 t1 t2")


class TestReportCaveats:
    def test_sink_exclusion_and_scale_are_surfaced(self):
        html = ReportBuilder.from_trace_result(_sink_trace()).render()
        caveats = re.findall(r'<div class="tl-caveat">(.*?)</div>', html)
        assert len(caveats) == 2
        assert "unitless" in caveats[0]
        assert "Excluded from feature ranking" in caveats[1]
        assert f"position {_SINK}" in caveats[1]
        assert EXCLUSION_REASON_OUTLIER in caveats[1]
        assert "x the other tokens' median norm" in caveats[1]
        assert "&quot;score_method&quot;: &quot;centered_norm&quot;" in html
        assert "&quot;raw_norm&quot;" in html  # feature meta reaches the JSON tab

    def test_l2_trace_caveat_and_no_exclusion(self):
        html = ReportBuilder.from_trace_result(_sink_trace(scoring="l2")).render()
        caveats = re.findall(r'<div class="tl-caveat">(.*?)</div>', html)
        assert len(caveats) == 1
        assert "legacy" in caveats[0]

    def test_mock_trace_has_no_exclusion_caveat(self):
        result = Scope.from_mock(n_layers=3, n_heads=2, d_model=16, seed=1).trace_full("a b c")
        html = ReportBuilder.from_trace_result(result).render()
        assert "Excluded from feature ranking" not in html


# ---------------------------------------------------------------------------
# Scope-level extractor options + payload
# ---------------------------------------------------------------------------


def _methods(result: TraceResult) -> set[str]:
    return {str(f.meta.get("method")) for f in result.features}


class TestScopeScoringOptions:
    def test_default_is_centered_with_sink_excluded(self):
        result = _sink_trace()
        assert _methods(result) == {"centered_norm"}
        assert result.meta["scoring"] == "centered"
        assert result.meta["excluded_positions"] == [_SINK]
        assert result.meta["outlier_positions"] == [_SINK]
        assert result.meta["outlier_stats"][_SINK]["max_ratio"] > 6.0
        assert all(f.token_idx != _SINK for f in result.features)

    def test_scope_level_l2(self):
        result = _sink_trace(scoring="l2")
        assert _methods(result) == {"l2_norm"}
        assert result.meta["excluded_positions"] == []

    def test_trace_full_override_does_not_stick(self):
        scope = Scope(_SinkProvider(), top_k_features=10)
        assert _methods(scope.trace_full("x", scoring="l2")) == {"l2_norm"}
        assert _methods(scope.trace_full("x")) == {"centered_norm"}

    def test_exclusion_options_pass_through(self):
        result = _sink_trace(exclude_positions=[2])
        assert result.meta["excluded_positions"] == [_SINK, 2]
        result = _sink_trace(exclude_outlier_positions=False)
        assert result.meta["excluded_positions"] == []
        assert result.meta["outlier_positions"] == [_SINK]
        scope = Scope(_SinkProvider(), top_k_features=10)
        r = scope.trace_full("x", exclude_positions=(3,), exclude_outlier_positions=False)
        assert r.meta["excluded_positions"] == [3]

    def test_invalid_scoring_rejected(self):
        with pytest.raises(ValueError, match="scoring"):
            Scope(_SinkProvider(), scoring="cosine")  # type: ignore[arg-type]

    def test_override_keeps_attached_sae(self):
        scope = Scope(_SinkProvider(), top_k_features=5)
        scope.attach_sae(_FakeSAE(), layer=2)  # type: ignore[arg-type]
        assert _methods(scope.trace_full("x")) == {"sae"}
        result = scope.trace_full("x", scoring="l2")
        assert _methods(result) == {"sae"}
        assert result.meta["excluded_positions"] == []

    def test_existing_signatures_still_work(self):
        mp = MockProvider(n_layers=2, n_heads=1, d_model=8, seed=4)
        result = Scope(mp, top_k_features=4, attribution_threshold=0.0).trace_full(
            "a b", top_k_features=3, use_supernodes=False
        )
        assert len(result.features) == 3
        assert "scoring='centered'" in repr(Scope(mp))


class TestPayload:
    def test_payload_carries_exclusions_and_score_method(self):
        payload = _sink_trace().to_payload()
        json.dumps(payload)  # JSON-safe
        assert payload["score_method"] == "centered_norm"
        assert payload["excluded_positions"] == [_SINK]
        assert payload["exclusion_reasons"] == {str(_SINK): EXCLUSION_REASON_OUTLIER}
        assert payload["input_tokens"] == payload["summary"]["tokens"]
        assert payload["graph"]["meta"]["excluded_positions"] == [_SINK]

    def test_black_box_payload_uses_prompt_tokens(self):
        out = ProviderOutput(
            prompt="alpha beta",
            tokens=["completion", "text"],
            token_ids=[],
            top_tokens=[("x", 0.4)],
            evidence_kind="black_box",
        )
        payload = TraceResult(prompt="alpha beta", output=out).to_payload()
        assert payload["input_tokens"] == ["alpha", "beta"]
        assert payload["excluded_positions"] == []
        assert payload["score_method"] == ""
