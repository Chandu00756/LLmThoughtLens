"""Tests for FeatureExtractor — white-box scoring, real masking + SAE feature paths."""

from __future__ import annotations

import numpy as np
import pytest
from LLmThoughtLens.circuits.tracer import CircuitTracer
from LLmThoughtLens.features.extractor import (
    FeatureExtractor,
    SAEAttachment,
    SAEHookUnavailableError,
    _median_excluding_each,
    detect_outlier_positions,
    sae_input_activations,
)
from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.providers.mock_provider import MockProvider
from LLmThoughtLens.utils.tokenizer_utils import MASK_TOKEN

# ---------------------------------------------------------------------------
# Synthetic white-box outputs
# ---------------------------------------------------------------------------

_SINK_POS = 0
_CONTENT_POS = 4


def _sink_output(
    n_layers: int = 6, n_tokens: int = 7, d_model: int = 32, seed: int = 0
) -> ProviderOutput:
    """White-box output with a GPT-2-style massive activation on position 0.

    Position 0's norm is ~40x every other token at layers 1..L-1 (the
    "attention sink"); position 4 carries a strong token-specific direction
    at every layer (the real content signal).
    """
    rng = np.random.default_rng(seed)
    acts = rng.standard_normal((n_layers, n_tokens, d_model))
    # Shared component every token carries (real residual streams have one).
    acts += 2.0 * rng.standard_normal(d_model)[None, None, :]
    direction = rng.standard_normal(d_model)
    direction /= np.linalg.norm(direction)
    if n_tokens > _CONTENT_POS:
        acts[:, _CONTENT_POS] += 8.0 * direction
    acts[1:, _SINK_POS] *= 40.0
    tokens = [f"t{i}" for i in range(n_tokens)]
    return ProviderOutput(
        prompt=" ".join(tokens),
        tokens=tokens,
        token_ids=list(range(n_tokens)),
        activations=acts.astype(np.float32),
        top_tokens=[("next", 0.5)],
        evidence_kind="white_box",
    )


def _legacy_l2_ranking(out: ProviderOutput, top_k: int) -> list[tuple[int, float]]:
    """Re-implementation of the pre-centring extractor: raw L2 norm, no exclusion."""
    acts = out.activations
    assert acts is not None
    n_layers, n_tokens, _ = acts.shape
    rows = [
        (layer * n_tokens + tok, float(np.linalg.norm(acts[layer, tok])))
        for layer in range(n_layers)
        for tok in range(n_tokens)
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:top_k]


class TestWhiteBox:
    def test_returns_white_box_features(self):
        mp = MockProvider(n_layers=3, n_heads=2, d_model=16, seed=1)
        out = mp.run("hello world test")
        feats = FeatureExtractor(top_k=8).extract(out, provider=mp)
        assert len(feats) == 8
        assert all(f.evidence_kind == "white_box" for f in feats)
        # Sorted by score descending
        scores = [f.score for f in feats]
        assert scores == sorted(scores, reverse=True)

    def test_raw_norm_is_kept_in_meta(self):
        mp = MockProvider(n_layers=2, n_heads=2, d_model=8, seed=5)
        out = mp.run("a b")
        feats = FeatureExtractor(top_k=4).extract(out, provider=mp)
        for f in feats:
            expected = float(np.linalg.norm(out.activations[f.layer, f.token_idx]))
            assert abs(f.meta["raw_norm"] - expected) < 1e-5
            assert f.meta["method"] == "centered_norm"

    def test_l2_scoring_score_is_l2_norm(self):
        mp = MockProvider(n_layers=2, n_heads=2, d_model=8, seed=5)
        out = mp.run("a b")
        feats = FeatureExtractor(top_k=4, scoring="l2").extract(out, provider=mp)
        for f in feats:
            expected = float(np.linalg.norm(out.activations[f.layer, f.token_idx]))
            assert abs(f.score - expected) < 1e-5
            assert f.meta["method"] == "l2_norm"

    def test_centered_score_definition(self):
        mp = MockProvider(n_layers=3, n_heads=2, d_model=16, seed=3)
        out = mp.run("one two three four five")
        n_tokens = out.n_tokens
        ex = FeatureExtractor(top_k=3 * n_tokens)
        feats = ex.extract(out, provider=mp)
        assert ex.last_excluded_positions == []
        acts = out.activations.astype(np.float64)
        for f in feats:
            layer = acts[f.layer]
            centre = np.median(layer, axis=0)
            scale = np.median(np.linalg.norm(layer, axis=-1))
            expected = np.linalg.norm(layer[f.token_idx] - centre) / scale
            assert f.score == pytest.approx(expected, rel=1e-6)
            assert f.meta["layer_scale"] == pytest.approx(scale, rel=1e-6)

    def test_feature_ids_match_legacy_numbering(self):
        mp = MockProvider(n_layers=3, n_heads=2, d_model=16, seed=4)
        out = mp.run("alpha beta gamma delta")
        feats = FeatureExtractor(top_k=100).extract(out, provider=mp)
        assert len(feats) == out.n_layers * out.n_tokens
        for f in feats:
            assert f.id == f.layer * out.n_tokens + f.token_idx

    @pytest.mark.parametrize("seed", range(6))
    def test_mock_provider_has_no_outlier_positions(self, seed):
        # Sensible defaults for the synthetic Gaussian mock: nothing is excluded.
        mp = MockProvider(n_layers=4, n_heads=2, d_model=16, seed=seed)
        out = mp.run("the capital of France is Paris and the population is large")
        ex = FeatureExtractor(top_k=100)
        feats = ex.extract(out, provider=mp)
        assert ex.last_outlier_positions == []
        assert ex.last_excluded_positions == []
        assert {f.token_idx for f in feats} == set(range(out.n_tokens))


class TestMassiveActivationOutliers:
    def test_sink_dominates_legacy_l2_scoring(self):
        out = _sink_output()
        feats = FeatureExtractor(top_k=5, scoring="l2").extract(out)
        # The legacy symptom: every top feature is the attention sink.
        assert all(f.token_idx == _SINK_POS for f in feats)

    def test_l2_scoring_reproduces_legacy_ranking(self):
        out = _sink_output()
        ex = FeatureExtractor(top_k=12, scoring="l2")
        feats = ex.extract(out)
        legacy = _legacy_l2_ranking(out, top_k=12)
        assert [f.id for f in feats] == [fid for fid, _ in legacy]
        for f, (_, score) in zip(feats, legacy, strict=True):
            assert f.score == pytest.approx(score, rel=1e-5)
            assert f.meta["method"] == "l2_norm"
            assert f.meta["excluded_positions"] == []
        # Detection still runs (for honest reporting) but nothing is excluded.
        assert ex.last_outlier_positions == [_SINK_POS]
        assert ex.last_excluded_positions == []

    def test_sink_does_not_dominate_by_default(self):
        out = _sink_output()
        ex = FeatureExtractor(top_k=10)
        feats = ex.extract(out)
        assert feats
        assert all(f.token_idx != _SINK_POS for f in feats)
        assert feats[0].token_idx == _CONTENT_POS
        assert ex.last_outlier_positions == [_SINK_POS]
        assert ex.last_excluded_positions == [_SINK_POS]
        stats = ex.last_outlier_stats[_SINK_POS]
        assert stats["max_ratio"] > 20.0
        assert stats["layer_frac"] == pytest.approx(5 / 6)
        for f in feats:
            assert f.meta["method"] == "centered_norm"
            assert f.meta["excluded_positions"] == [_SINK_POS]
            expected = float(np.linalg.norm(out.activations[f.layer, f.token_idx]))
            assert f.meta["raw_norm"] == pytest.approx(expected, rel=1e-5)

    def test_sink_excluded_from_layer_statistics(self):
        out = _sink_output()
        feats = FeatureExtractor(top_k=100).extract(out)
        acts = out.activations.astype(np.float64)
        keep = [t for t in range(out.n_tokens) if t != _SINK_POS]
        for f in feats:
            layer = acts[f.layer, keep]
            scale = np.median(np.linalg.norm(layer, axis=-1))
            centre = np.median(layer, axis=0)
            expected = np.linalg.norm(acts[f.layer, f.token_idx] - centre) / scale
            assert f.score == pytest.approx(expected, rel=1e-6)

    def test_outlier_exclusion_can_be_disabled(self):
        out = _sink_output()
        ex = FeatureExtractor(top_k=5, exclude_outlier_positions=False)
        feats = ex.extract(out)
        assert ex.last_excluded_positions == []
        assert feats[0].token_idx == _SINK_POS

    def test_l2_scoring_with_outlier_exclusion(self):
        out = _sink_output()
        feats = FeatureExtractor(top_k=5, scoring="l2", exclude_outlier_positions=True).extract(out)
        assert all(f.token_idx != _SINK_POS for f in feats)
        for f in feats:
            assert f.score == pytest.approx(f.meta["raw_norm"])

    def test_explicit_exclude_positions(self):
        mp = MockProvider(n_layers=3, n_heads=2, d_model=16, seed=2)
        out = mp.run("a b c d e")
        ex = FeatureExtractor(top_k=100, exclude_positions=[-1, 1, 99])
        feats = ex.extract(out, provider=mp)
        last = out.n_tokens - 1
        assert ex.last_excluded_positions == [1, last]
        assert {f.token_idx for f in feats} == {0, 2, 3}

    def test_excluding_every_position_returns_nothing(self):
        mp = MockProvider(n_layers=2, n_heads=2, d_model=8, seed=2)
        out = mp.run("a b")
        assert FeatureExtractor(exclude_positions=[0, 1]).extract(out, provider=mp) == []

    def test_single_rankable_position_falls_back_to_raw_norm(self):
        out = _sink_output(n_layers=4, n_tokens=2)
        ex = FeatureExtractor(top_k=10)
        feats = ex.extract(out)
        assert ex.last_excluded_positions == [_SINK_POS]
        assert len(feats) == 4
        for f in feats:
            assert f.token_idx == 1
            assert f.meta["method"] == "l2_norm"
            assert f.meta["fallback"] == "centered_needs_2_positions"
            assert f.score == pytest.approx(f.meta["raw_norm"])

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            FeatureExtractor(scoring="cosine")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            FeatureExtractor(outlier_ratio=1.0)
        with pytest.raises(ValueError):
            FeatureExtractor(outlier_min_layer_frac=0.0)


class TestOutlierDetection:
    def test_median_excluding_each_matches_brute_force(self):
        rng = np.random.default_rng(7)
        for n_cols in (2, 3, 4, 7, 10):
            x = rng.random((5, n_cols))
            x[0, 1] = x[0, 2] if n_cols > 2 else x[0, 1]  # include a tie
            got = _median_excluding_each(x)
            for r in range(x.shape[0]):
                for c in range(n_cols):
                    assert got[r, c] == pytest.approx(np.median(np.delete(x[r], c)))

    def test_detects_two_token_sink(self):
        norms = np.array([[5.0, 4.0], [200.0, 5.0], [210.0, 5.5]])
        positions, stats = detect_outlier_positions(norms)
        assert positions == [0]
        assert stats[0]["layer_frac"] == pytest.approx(2 / 3)

    def test_single_layer_spike_is_not_an_outlier(self):
        # A content token spiking in one of ten layers is not a massive activation.
        norms = np.ones((10, 6))
        norms[3, 2] = 50.0
        positions, _ = detect_outlier_positions(norms)
        assert positions == []

    def test_degenerate_inputs(self):
        assert detect_outlier_positions(np.ones((3, 1))) == ([], {})
        assert detect_outlier_positions(np.zeros((3, 4))) == ([], {})
        with pytest.raises(ValueError):
            detect_outlier_positions(np.ones(4))


class _DuckSAE:
    """Minimal SAE stand-in (``encode`` + ``labels``) so the SAE path runs without torch."""

    labels: dict[int, str] = {}

    def encode(self, x: np.ndarray) -> np.ndarray:
        codes = np.zeros((x.shape[0], 3), dtype=np.float32)
        codes[:, 0] = np.abs(x[:, 0]) + 0.5
        return codes


def _assert_record_empty(ex: FeatureExtractor) -> None:
    assert ex.last_excluded_positions == []
    assert ex.last_outlier_positions == []
    assert ex.last_outlier_stats == {}


class TestExclusionRecordReset:
    """``last_*`` exclusion attributes describe the *latest* extract() call only."""

    def _primed(self) -> FeatureExtractor:
        ex = FeatureExtractor(top_k=6, blackbox_budget=3)
        ex.extract(_sink_output())
        assert ex.last_excluded_positions == [_SINK_POS]
        assert ex.last_outlier_stats
        return ex

    def test_black_box_path_resets_record(self):
        ex = self._primed()
        bb = _BlackBoxWrapper()
        feats = ex.extract(bb.run("A Y B"), provider=bb)
        assert feats
        _assert_record_empty(ex)

    def test_black_box_without_provider_resets_record(self):
        ex = self._primed()
        ex.extract(_BlackBoxWrapper().run("A B"))
        _assert_record_empty(ex)

    def test_sae_path_resets_record(self):
        # The SAE path re-derives the record from the current output (it uses the
        # same sink exclusion as residual-site scoring), never a stale one.
        ex = self._primed()
        ex.attach_sae(_DuckSAE(), layer=2)  # type: ignore[arg-type]
        feats = ex.extract(MockProvider(n_layers=4, d_model=16, seed=0).run("one two three"))
        assert feats and all(f.meta["method"] == "sae" for f in feats)
        _assert_record_empty(ex)
        ex.extract(_sink_output())
        assert ex.last_excluded_positions == [_SINK_POS]
        assert ex.last_outlier_positions == [_SINK_POS]

    def test_outlier_positions_recorded_in_meta(self):
        feats = FeatureExtractor(top_k=4).extract(_sink_output())
        assert all(f.meta["outlier_positions"] == [_SINK_POS] for f in feats)
        feats = FeatureExtractor(top_k=4, exclude_outlier_positions=False).extract(_sink_output())
        assert all(f.meta["excluded_positions"] == [] for f in feats)
        assert all(f.meta["outlier_positions"] == [_SINK_POS] for f in feats)


class TestTracerConsumesRawNorm:
    """The tracer's energy accounting must stay in raw-activation units."""

    def test_error_residual_uses_raw_norm_and_skips_excluded_energy(self):
        out = _sink_output()
        feats = FeatureExtractor(top_k=8).extract(out)
        graph = CircuitTracer(min_weight=0.0).trace(out, feats)
        assert graph.meta["excluded_positions"] == [_SINK_POS]

        energy = np.sum(np.square(out.activations.astype(np.float64)), axis=-1)
        total = float(energy.sum() - energy[:, _SINK_POS].sum())
        explained = sum(f.meta["raw_norm"] ** 2 for f in feats)
        err = graph.node(CircuitTracer.error_node_id())
        assert err is not None
        assert err.meta["unexplained_fraction"] == pytest.approx(
            (total - explained) / total, rel=1e-6
        )
        assert err.meta["excluded_positions"] == [_SINK_POS]
        assert err.meta["excluded_energy_fraction"] == pytest.approx(
            energy[:, _SINK_POS].sum() / energy.sum(), rel=1e-6
        )

    def test_last_layer_to_output_edges_use_raw_norm(self):
        out = _sink_output()
        feats = FeatureExtractor(top_k=8).extract(out)
        graph = CircuitTracer(min_weight=0.0).trace(out, feats)
        by_id = {f.id: f for f in feats}
        out_edges = [
            e
            for e in graph.in_edges(CircuitTracer.output_node_id())
            if e.method == "last_layer_to_output"
        ]
        assert out_edges
        for e in out_edges:
            assert e.weight == pytest.approx(by_id[e.src].meta["raw_norm"])

    def test_legacy_l2_residual_accounting_unchanged(self):
        out = _sink_output()
        feats = FeatureExtractor(top_k=8, scoring="l2").extract(out)
        graph = CircuitTracer(min_weight=0.0).trace(out, feats)
        assert "excluded_positions" not in graph.meta
        total = float(np.linalg.norm(out.activations.astype(np.float64)) ** 2)
        explained = sum(f.score**2 for f in feats)
        err = graph.node(CircuitTracer.error_node_id())
        assert err is not None
        assert err.meta["unexplained_fraction"] == pytest.approx(
            (total - explained) / total, rel=1e-5
        )
        assert "excluded_positions" not in err.meta


class _BlackBoxWrapper(BaseProvider):
    """Strips activations so the extractor takes the black-box path.

    Returns deterministic top-token probabilities so masking has a real effect.
    """

    evidence_kind = "black_box"

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "bb_test"

    def run(self, prompt: str, **_) -> ProviderOutput:
        self.calls.append(prompt)
        tokens = prompt.split() or ["<empty>"]
        # Prob of "X" is high when "Y" is unmasked in the prompt, else low.
        prob_x = 0.9 if "Y" in tokens else 0.1
        return ProviderOutput(
            prompt=prompt,
            tokens=tokens,
            token_ids=[],
            top_tokens=[("X", prob_x), ("Z", 1.0 - prob_x)],
            evidence_kind="black_box",
        )


class TestBlackBox:
    def test_masking_produces_real_importance_score(self):
        bb = _BlackBoxWrapper()
        ex = FeatureExtractor(top_k=5, blackbox_budget=4)
        out = bb.run("A Y B C")
        feats = ex.extract(out, provider=bb)
        # The 'Y' token should have the highest causal importance score.
        y_feat = next(f for f in feats if f.label.endswith("Y"))
        other = [f for f in feats if not f.label.endswith("Y")]
        for f in other:
            assert y_feat.score > f.score, f"Y importance ({y_feat.score}) not > other ({f.score})"
        assert all(f.evidence_kind == "black_box" for f in feats)
        assert all(f.meta.get("method") == "token_masking" for f in feats)

    def test_mask_tokens_were_used(self):
        bb = _BlackBoxWrapper()
        ex = FeatureExtractor(top_k=5, blackbox_budget=3)
        out = bb.run("A B C")
        ex.extract(out, provider=bb)
        # At least one provider call masked a token.
        assert any(MASK_TOKEN in call for call in bb.calls)

    def test_pairwise_interactions(self):
        bb = _BlackBoxWrapper()
        ex = FeatureExtractor(top_k=5, blackbox_budget=3)
        pairs = ex.compute_pairwise_interactions(bb, "A Y B", budget=3)
        # n=3 → 3 unordered pairs.
        assert len(pairs) == 3
        for (i, j), score in pairs.items():
            assert i < j
            assert isinstance(score, float)


# ---------------------------------------------------------------------------
# SAE features: hook-point mapping, multi-SAE extraction, meta contract
# ---------------------------------------------------------------------------

_TINY_PROMPT = "the capital of the state containing dallas is"
_SITES = ("resid_pre", "resid_post", "mlp_out", "attn_out")

#: The meta keys every SAE feature carries (consumed by attribution / patching code).
_SAE_META_KEYS = {
    "method",
    "sae_name",
    "sae_slot",
    "sae_feature_id",
    "sae_layer",
    "sae_site",
    "sae_hook_name",
    "sae_id",
    "sae_release",
    "sae_architecture",
    "activation",
    "position",
    "activation_source",
    "activation_layer",
    "raw_norm",
    "excluded_positions",
}


def _independent_capture(model, token_ids: list[int]) -> dict[tuple[int, str], np.ndarray]:
    """Every SAE input of a GPT-2, captured with raw torch hooks + output_hidden_states.

    Deliberately independent of HookedModel: ``hidden_states[0]`` is the
    embedding output, ``hidden_states[l]`` (``0 < l < L``) the output of block
    ``l - 1``; the last block's output is taken from a forward hook because
    ``hidden_states[L]`` has the final LayerNorm applied.
    """
    import torch

    cap: dict[tuple[int, str], np.ndarray] = {}

    def keep(key, pick):
        def hook(_mod, _inp, out):
            cap[key] = pick(out)[0].detach().float().numpy()

        return hook

    blocks = model.transformer.h
    handles = []
    for layer, blk in enumerate(blocks):
        first = lambda o: o[0] if isinstance(o, tuple) else o  # noqa: E731
        handles.append(blk.register_forward_hook(keep((layer, "resid_post"), first)))
        handles.append(blk.mlp.register_forward_hook(keep((layer, "mlp_out"), first)))
        handles.append(blk.attn.register_forward_hook(keep((layer, "attn_out"), first)))
    try:
        with torch.no_grad():
            hs = model(torch.tensor([token_ids]), output_hidden_states=True).hidden_states
    finally:
        for h in handles:
            h.remove()
    n_layers = len(blocks)
    for layer in range(n_layers):
        cap[(layer, "resid_pre")] = hs[layer][0].float().numpy()
        if layer < n_layers - 1:  # sanity of the capture itself
            np.testing.assert_allclose(cap[(layer, "resid_post")], hs[layer + 1][0].numpy())
    return cap


def _identity_sae(d: int, layer: int | None = None, site: str | None = None, **kw):
    """Relu SAE with codes ``[relu(x), relu(-x)]``: every input coordinate is readable."""
    from LLmThoughtLens.features.sae import SparseAutoencoder

    eye = np.eye(d, dtype=np.float32)
    return SparseAutoencoder.from_weights(
        np.concatenate([eye, -eye], axis=1),
        np.concatenate([eye, -eye], axis=0),
        architecture="relu",
        apply_b_dec_to_input=False,
        hook_layer=layer,
        hook_site=site,
        **kw,
    )


def _random_sae(d: int, f: int, layer: int | None, site: str | None, seed: int = 0, **kw):
    from LLmThoughtLens.features.sae import SparseAutoencoder

    rng = np.random.default_rng(seed)
    return SparseAutoencoder.from_weights(
        (rng.standard_normal((d, f)) / np.sqrt(d)).astype(np.float32),
        rng.standard_normal((f, d)).astype(np.float32),
        rng.standard_normal(f).astype(np.float32) * 0.01,
        architecture="relu",
        apply_b_dec_to_input=False,
        hook_layer=layer,
        hook_site=site,
        **kw,
    )


@pytest.fixture(scope="module")
def tiny_hf():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from _tiny_hf import make_provider

    provider = make_provider(perturb_norms=True)
    out = provider.run(_TINY_PROMPT)
    return provider, out, _independent_capture(provider._model, out.token_ids)


class TestSAEHookMapping:
    """``blocks.L.hook_<site>`` -> the exact tensor, checked against independent captures."""

    @pytest.mark.parametrize("layer", [0, 1])
    @pytest.mark.parametrize("site", _SITES)
    def test_mapping_matches_independent_capture(self, tiny_hf, layer, site):
        provider, out, refs = tiny_hf
        x, prov = sae_input_activations(out, layer, site, provider)
        np.testing.assert_allclose(x, refs[(layer, site)], atol=1e-6)
        expected = {
            ("resid_pre", 0): ("embeddings", None),
            ("resid_pre", 1): ("activations", 0),
            ("resid_post", 0): ("activations", 0),
            ("resid_post", 1): ("activations", 1),
        }.get((site, layer), ("recomputed", None))
        assert (prov["activation_source"], prov["activation_layer"]) == expected

    def test_resid_pre_is_not_off_by_one(self, tiny_hf):
        _, out, refs = tiny_hf
        acts = out.activations
        # The three candidate tensors really differ, so the mapping test is meaningful.
        assert np.abs(acts[1] - acts[0]).max() > 1e-3
        assert np.abs(out.embeddings - acts[0]).max() > 1e-3
        np.testing.assert_array_equal(sae_input_activations(out, 1, "resid_pre")[0], acts[0])
        np.testing.assert_array_equal(sae_input_activations(out, 0, "resid_post")[0], acts[0])
        np.testing.assert_array_equal(sae_input_activations(out, 0, "resid_pre")[0], out.embeddings)
        # resid_post of the last layer is the true residual (no final LayerNorm)
        np.testing.assert_allclose(acts[-1], refs[(1, "resid_post")], atol=1e-6)

    def test_identity_saes_report_their_exact_inputs(self, tiny_hf):
        provider, out, refs = tiny_hf
        d = out.activations.shape[-1]
        ex = FeatureExtractor(top_k=10_000)
        ex.attach_saes([_identity_sae(d, layer, site) for layer in (0, 1) for site in _SITES])
        feats = ex.extract(out, provider=provider)
        assert {f.meta["sae_name"] for f in feats} == {
            f"blocks.{layer}.hook_{site}" for layer in (0, 1) for site in _SITES
        }
        for f in feats:
            ref = refs[(f.meta["sae_layer"], f.meta["sae_site"])][f.meta["position"]]
            fid = f.meta["sae_feature_id"]
            value = ref[fid] if fid < d else -ref[fid - d]
            assert f.score == pytest.approx(value, rel=1e-5, abs=1e-6)
            assert f.meta["activation"] == f.score and f.token_idx == f.meta["position"]
            site, layer = f.meta["sae_site"], f.meta["sae_layer"]
            assert f.layer == (max(layer - 1, 0) if site == "resid_pre" else layer)

    def test_unavailable_sites_raise_instead_of_substituting(self):
        out = MockProvider(n_layers=3, d_model=8, seed=1).run("a b c")
        assert out.embeddings is None
        with pytest.raises(SAEHookUnavailableError, match="blocks.0.hook_resid_pre"):
            sae_input_activations(out, 0, "resid_pre")
        with pytest.raises(SAEHookUnavailableError, match="blocks.1.hook_mlp_out") as info:
            sae_input_activations(out, 1, "mlp_out")
        assert isinstance(info.value, NotImplementedError)
        with pytest.raises(SAEHookUnavailableError, match="hook_attn_out"):
            sae_input_activations(out, 1, "attn_out", MockProvider())  # no HookedModel
        with pytest.raises(IndexError):
            sae_input_activations(out, 3, "resid_post")
        with pytest.raises(ValueError, match="site"):
            sae_input_activations(out, 0, "resid_mid")
        bb = _BlackBoxWrapper().run("a b")
        with pytest.raises(SAEHookUnavailableError, match="black-box"):
            sae_input_activations(bb, 0, "resid_post")

    def test_extract_raises_for_unreadable_sae_site(self):
        pytest.importorskip("torch")
        out = MockProvider(n_layers=3, d_model=8, seed=1).run("a b c")
        ex = FeatureExtractor()
        ex.attach_sae(_identity_sae(8, 0, "resid_pre"))
        with pytest.raises(SAEHookUnavailableError, match="embeddings"):
            ex.extract(out)

    def test_recompute_refuses_when_it_cannot_reproduce(self, tiny_hf):
        import dataclasses

        provider, out, _ = tiny_hf
        tampered = dataclasses.replace(out, activations=out.activations + 1.0)
        with pytest.raises(SAEHookUnavailableError, match="did not reproduce"):
            sae_input_activations(tampered, 0, "mlp_out", provider)
        intervened = dataclasses.replace(out, meta={**out.meta, "n_intervention_hooks": 1})
        with pytest.raises(SAEHookUnavailableError, match="interventions"):
            sae_input_activations(intervened, 0, "mlp_out", provider)
        no_ids = dataclasses.replace(out, token_ids=[])
        with pytest.raises(SAEHookUnavailableError, match="token ids"):
            sae_input_activations(no_ids, 0, "mlp_out", provider)

        class _Gemma2Provider:
            hooked = type("H", (), {"family": "gemma2"})()

        with pytest.raises(SAEHookUnavailableError, match="post-sublayer norm"):
            sae_input_activations(out, 0, "mlp_out", _Gemma2Provider())  # type: ignore[arg-type]


class TestMultiSAEExtraction:
    def test_ids_unique_and_meta_contract(self, tiny_hf):
        provider, out, _ = tiny_hf
        d = out.activations.shape[-1]
        saes = [
            _random_sae(d, 48, 0, "resid_pre", seed=1, sae_id="a", release="rel"),
            _random_sae(d, 48, 1, "resid_pre", seed=2),
            _random_sae(d, 64, 1, "resid_post", seed=3),
            _random_sae(d, 40, 0, "mlp_out", seed=4),
            _random_sae(d, 48, 1, "resid_post", seed=5),  # same hook point: name gets "#2"
        ]
        ex = FeatureExtractor(top_k=10_000)
        atts = ex.attach_saes(saes)
        assert [a.name for a in atts][-1] == "blocks.1.hook_resid_post#2"
        assert list(ex.sae_map) == [a.name for a in atts] and ex.saes == tuple(atts)
        feats = ex.extract(out, provider=provider)
        assert len({f.id for f in feats}) == len(feats)
        assert all(0 <= f.id < 1_000_000_000 for f in feats)
        assert {f.meta["sae_slot"] for f in feats} == set(range(len(saes)))
        for f in feats:
            assert set(f.meta) >= _SAE_META_KEYS, _SAE_META_KEYS - set(f.meta)
            att = ex.saes[f.meta["sae_slot"]]
            assert f.meta["sae_name"] == att.name
            assert (f.meta["sae_layer"], f.meta["sae_site"]) == (att.layer, att.site)
            assert f.meta["sae_hook_name"] == att.hook_name
            assert f.id % att.sae.config.d_sae == f.meta["sae_feature_id"]
            assert f.label == f"{att.name}/feature_{f.meta['sae_feature_id']}"
            assert f.evidence_kind == "white_box" and f.node_type == "feature"
            w = float(np.linalg.norm(att.sae.W_dec[:, f.meta["sae_feature_id"]].numpy()))
            assert f.meta["raw_norm"] == pytest.approx(f.score * w, rel=1e-5)
        first = next(f for f in feats if f.meta["sae_slot"] == 0)
        assert (first.meta["sae_id"], first.meta["sae_release"]) == ("a", "rel")
        assert first.meta["sae_architecture"] == "relu"
        scores = [f.score for f in feats]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_global_vs_per_sae(self, tiny_hf):
        provider, out, _ = tiny_hf
        d = out.activations.shape[-1]
        big = _random_sae(d, 48, 1, "resid_post", seed=6)
        big.W_enc.mul_(50.0)  # dominates every global ranking
        small = _random_sae(d, 48, 1, "resid_pre", seed=7)
        ex = FeatureExtractor(top_k=5)
        ex.attach_saes([big, small])
        assert {f.meta["sae_slot"] for f in ex.extract(out)} == {0}
        ex = FeatureExtractor(top_k=5, top_k_per_sae=3)
        ex.attach_saes([big, small])
        feats = ex.extract(out)
        assert [sum(f.meta["sae_slot"] == s for f in feats) for s in (0, 1)] == [3, 3]
        with pytest.raises(ValueError, match="top_k_per_sae"):
            FeatureExtractor(top_k_per_sae=0)

    def test_raw_norm_uses_output_scale(self, tiny_hf):
        _, out, _ = tiny_hf
        d = out.activations.shape[-1]
        sae = _random_sae(
            d, 48, 1, "resid_post", seed=8, normalize_activations="constant_norm_rescale"
        )
        ex = FeatureExtractor(top_k=50)
        ex.attach_sae(sae)
        x = out.activations[1]
        tok_scale = np.linalg.norm(x, axis=-1) / np.sqrt(d)
        dec = np.linalg.norm(sae.W_dec.numpy(), axis=0)
        for f in ex.extract(out):
            expected = f.score * dec[f.meta["sae_feature_id"]] * tok_scale[f.token_idx]
            assert f.meta["raw_norm"] == pytest.approx(expected, rel=1e-4)

    def test_labels_and_exclusions(self, tiny_hf):
        _, out, _ = tiny_hf
        d = out.activations.shape[-1]
        sae = _identity_sae(d, 1, "resid_post")
        top = FeatureExtractor(top_k=1)
        top.attach_sae(sae)
        best = top.extract(out)[0]
        sae.set_label(best.meta["sae_feature_id"], "texas-ish")
        assert top.extract(out)[0].label == "texas-ish"
        ex = FeatureExtractor(top_k=1000, exclude_positions=[0, -1])
        ex.attach_sae(sae)
        feats = ex.extract(out)
        n = len(out.tokens)
        assert {f.token_idx for f in feats}.isdisjoint({0, n - 1})
        assert all(f.meta["excluded_positions"] == [0, n - 1] for f in feats)
        assert ex.last_excluded_positions == [0, n - 1]

    def test_sink_outliers_excluded_by_default_with_opt_out(self):
        # Same exclusion semantics as residual-site scoring (Phase B): the sink is
        # excluded by default; exclude_outlier_positions=False / scoring="l2" opt out.
        pytest.importorskip("torch")
        out = _sink_output()
        sae = _identity_sae(out.activations.shape[-1], 2, "resid_post")
        default = FeatureExtractor(top_k=5)
        default.attach_sae(sae)
        feats = default.extract(out)
        assert feats and _SINK_POS not in {f.token_idx for f in feats}
        assert all(f.meta["outlier_positions"] == [_SINK_POS] for f in feats)
        assert all(f.meta["excluded_positions"] == [_SINK_POS] for f in feats)
        assert default.last_outlier_positions == [_SINK_POS]
        for opt_out in (
            FeatureExtractor(top_k=5, exclude_outlier_positions=False),
            FeatureExtractor(top_k=5, scoring="l2"),
        ):
            opt_out.attach_sae(sae)
            feats = opt_out.extract(out)
            assert {f.token_idx for f in feats} == {_SINK_POS}
            assert all(f.meta["outlier_positions"] == [_SINK_POS] for f in feats)
            assert all(f.meta["excluded_positions"] == [] for f in feats)


class TestSAEBosCaveat:
    """SAELens SAEs were trained with BOS at position 0; a prompt without it is flagged."""

    def _provider(self, bos: int | None):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from _tiny_hf import make_provider

        provider = make_provider(perturb_norms=True)
        provider._tokenizer.bos_token_id = bos
        return provider

    @pytest.mark.parametrize(
        ("source", "extra", "bos", "warned"),
        [
            ("saelens", {}, 1, True),  # SAELens default prepend_bos=True, no BOS in prompt
            ("saelens", {"prepend_bos": False}, 1, False),
            ("native", {}, 1, False),  # unknown provenance: no claim either way
            ("gemma_scope", {"prepend_bos": True}, 1, True),
            ("saelens", {}, None, False),  # tokenizer has no BOS id: cannot tell
        ],
    )
    def test_warning_recorded_only_on_mismatch(self, source, extra, bos, warned):
        provider = self._provider(bos)
        out = provider.run(_TINY_PROMPT)
        d = out.activations.shape[-1]
        sae = _identity_sae(d, 1, "resid_post", source_format=source, extra=dict(extra))
        ex = FeatureExtractor(top_k=5)
        ex.attach_sae(sae)
        feats = ex.extract(out, provider=provider)
        assert feats
        assert all(("sae_input_warning" in f.meta) is warned for f in feats)
        assert bool(ex.last_sae_warnings) is warned
        if warned:
            assert "BOS" in feats[0].meta["sae_input_warning"]
            ex.extract(MockProvider(d_model=d).run("x y"))  # record resets per call
            assert ex.last_sae_warnings == []

    def test_prompt_starting_with_bos_is_fine_and_meta_overrides(self):
        import dataclasses

        provider = self._provider(1)
        out = provider.run(_TINY_PROMPT)
        d = out.activations.shape[-1]
        sae = _identity_sae(d, 1, "resid_post", source_format="saelens")
        ex = FeatureExtractor(top_k=5)
        ex.attach_sae(sae)
        starts = dataclasses.replace(out, token_ids=[1, *out.token_ids[1:]])
        assert not any("sae_input_warning" in f.meta for f in ex.extract(starts, provider))
        flagged = dataclasses.replace(out, meta={**out.meta, "bos_token_id": 99})
        assert all("sae_input_warning" in f.meta for f in ex.extract(flagged))


class TestAttachAPI:
    def test_defaults_come_from_sae_metadata(self):
        pytest.importorskip("torch")
        ex = FeatureExtractor()
        att = ex.attach_sae(_identity_sae(8, 3, "resid_pre"))
        assert isinstance(att, SAEAttachment)
        assert (att.layer, att.site, att.name) == (3, "resid_pre", "blocks.3.hook_resid_pre")
        assert att.residual_layer() == 2 and ex.sae_layer == 3 and ex.sae is att.sae
        att = ex.attach_sae(_identity_sae(8, 3, "resid_pre"), 3)  # Scope's positional form
        assert (att.layer, att.site) == (3, "resid_pre")
        assert len(ex.saes) == 1  # attach_sae replaces

    def test_metadata_free_sae_keeps_legacy_meaning(self):
        out = MockProvider(n_layers=4, d_model=16, seed=0).run("one two three")
        ex = FeatureExtractor(top_k=6)
        with pytest.raises(ValueError, match="layer is required"):
            ex.attach_sae(_DuckSAE())  # type: ignore[arg-type]
        att = ex.attach_sae(_DuckSAE(), 2)  # type: ignore[arg-type]
        assert (att.site, att.residual_layer()) == ("resid_post", 2)
        feats = ex.extract(out)
        assert feats and all(f.layer == 2 and f.meta["activation_layer"] == 2 for f in feats)
        assert all(f.id % 3 == f.meta["sae_feature_id"] for f in feats)
        expected = np.abs(out.activations[2][:, 0]) + 0.5  # _DuckSAE reads activations[2]
        for f in feats:
            assert f.score == pytest.approx(float(expected[f.token_idx]), rel=1e-6)
            assert "raw_norm" not in f.meta  # duck SAE exposes no decoder

    def test_add_and_attach_many(self):
        pytest.importorskip("torch")
        ex = FeatureExtractor()
        ex.add_sae(_identity_sae(8, 1, "resid_post"))
        ex.add_sae(_identity_sae(8, 2, "resid_pre"), name="mine")
        assert [a.name for a in ex.saes] == ["blocks.1.hook_resid_post", "mine"]
        with pytest.raises(ValueError, match="already attached"):
            ex.add_sae(_identity_sae(8, 2, "resid_pre"), name="mine")
        atts = ex.attach_saes({0: _DuckSAE(), 2: _DuckSAE()})
        assert [(a.layer, a.site) for a in atts] == [(0, "resid_post"), (2, "resid_post")]
        atts = ex.attach_saes({"x": _identity_sae(8, 1, "mlp_out")})
        assert [(a.name, a.layer, a.site) for a in atts] == [("x", 1, "mlp_out")]
        atts = ex.attach_saes([(_DuckSAE(), 1), (_DuckSAE(), 1, "resid_pre"), atts[0]])
        assert [(a.layer, a.site, a.name) for a in atts] == [
            (1, "resid_post", "blocks.1.hook_resid_post"),
            (1, "resid_pre", "blocks.1.hook_resid_pre"),
            (1, "mlp_out", "x"),
        ]
        with pytest.raises(TypeError, match="mapping keys"):
            ex.attach_saes({1.5: _DuckSAE()})
        with pytest.raises(TypeError, match="expected"):
            ex.attach_saes([(_DuckSAE(),)])
        with pytest.raises(TypeError, match="encode"):
            ex.attach_sae(object(), 1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown SAE site"):
            ex.attach_sae(_DuckSAE(), 1, site="resid_mid")  # type: ignore[arg-type]
        ex.detach_saes()
        assert ex.saes == () and ex.sae is None and ex.sae_layer == -1

    def test_hook_point_disagreement_warns_and_unreadable_hook_raises(self):
        pytest.importorskip("torch")
        ex = FeatureExtractor()
        with pytest.warns(UserWarning, match="attached at blocks.4.hook_resid_pre"):
            ex.attach_sae(_identity_sae(8, 3, "resid_pre"), 4)
        z_sae = _identity_sae(8, hook_name="blocks.2.attn.hook_z")
        with pytest.raises(ValueError, match="hook_z"):
            ex.attach_sae(z_sae)
        att = ex.attach_sae(z_sae, 2, site="resid_post")  # explicit override is allowed
        assert att.site == "resid_post"

    def test_out_of_range_saes_are_skipped(self):
        pytest.importorskip("torch")
        out = MockProvider(n_layers=3, d_model=8, seed=2).run("a b c d")
        ex = FeatureExtractor(top_k=4)
        ex.attach_saes([_identity_sae(8, 7, "resid_post"), _identity_sae(8, 1, "resid_post")])
        with pytest.warns(RuntimeWarning, match="skipped"):
            feats = ex.extract(out)
        assert feats and all(f.meta["sae_slot"] == 1 for f in feats)
        assert [name for name, _ in ex.last_skipped_saes] == ["blocks.7.hook_resid_post"]
        ex.attach_sae(_identity_sae(8, 9, "resid_post"))
        with pytest.warns(RuntimeWarning):
            feats = ex.extract(out)
        assert feats and all(f.meta["method"] == "centered_norm" for f in feats)  # fallback
        ex.attach_sae(_identity_sae(4, 1, "resid_post"))
        with pytest.raises(ValueError, match="d_in=4 but .* d_model=8"):
            ex.extract(out)


class TestSAEPipelineIntegration:
    def test_scope_attach_sae_end_to_end(self, tiny_hf):
        from LLmThoughtLens.scope import Scope

        provider, out, _ = tiny_hf
        d = out.activations.shape[-1]
        scope = Scope(provider)
        sae = _random_sae(d, 48, 1, "resid_pre", seed=9)
        scope.attach_sae(sae, 1)
        result = scope.trace_full(_TINY_PROMPT)
        assert result.features and all(f.meta["method"] == "sae" for f in result.features)
        assert all(f.meta["sae_layer"] == 1 and f.layer == 0 for f in result.features)
        assert result.supernodes

    def test_supernodes_with_sae_map(self, tiny_hf):
        from LLmThoughtLens.circuits.supernodes import SupernodeGrouper

        provider, out, _ = tiny_hf
        d = out.activations.shape[-1]
        ex = FeatureExtractor(top_k=30)
        ex.attach_saes([_identity_sae(d, 1, "resid_post"), _identity_sae(d, 1, "resid_pre")])
        feats = ex.extract(out, provider=provider)
        assert len({f.meta["sae_name"] for f in feats}) == 2
        groups = SupernodeGrouper(similarity_threshold=0.99, sae=ex.sae_map).group(feats, out)
        for g in groups:  # identity SAE: same dictionary index => identical direction
            assert len({f.meta["sae_feature_id"] for f in g.features}) == 1
        # one SAE for features from two SAEs is ambiguous => activation clustering
        single = SupernodeGrouper(similarity_threshold=0.99, sae=ex.sae)
        assert sum(len(g.features) for g in single.group(feats, out)) == len(feats)


@pytest.mark.slow
class TestRealGPT2MultiSAE:
    """Hook mapping + multi-SAE extraction on real gpt2 (random SAE weights, no download)."""

    def test_mapping_and_extraction(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from LLmThoughtLens.models.hooked import load_hf_model
        from LLmThoughtLens.providers.huggingface_provider import HuggingFaceProvider

        try:
            model, tok, device = load_hf_model("gpt2", device="cpu", local_files_only=True)
        except Exception as exc:
            pytest.skip(f"gpt2 not available offline: {exc}")
        provider = HuggingFaceProvider(model_name="gpt2", device="cpu")
        provider._model, provider._tokenizer, provider._device = model, tok, device
        out = provider.run("The capital of the state containing Dallas is")
        refs = _independent_capture(model, out.token_ids)
        keys = [(0, "resid_pre"), (6, "resid_pre"), (6, "resid_post"), (11, "resid_post"),
                (3, "mlp_out"), (8, "attn_out")]  # fmt: skip
        for layer, site in keys:
            x, _ = sae_input_activations(out, layer, site, provider)
            scale = float(np.abs(refs[(layer, site)]).max())
            np.testing.assert_allclose(x, refs[(layer, site)], atol=1e-4 * max(scale, 1.0))
        ex = FeatureExtractor(top_k=40, top_k_per_sae=5, exclude_positions=[0])
        ex.attach_saes(
            [_random_sae(768, 1024, layer, site, seed=i) for i, (layer, site) in enumerate(keys)]
        )
        feats = ex.extract(out, provider=provider)
        assert len(feats) == 5 * len(keys) and len({f.id for f in feats}) == len(feats)
        assert {(f.meta["sae_layer"], f.meta["sae_site"]) for f in feats} == set(keys)
        assert all(f.token_idx != 0 for f in feats)
        # GPT-2's tokenizer adds no BOS: a SAELens-trained SAE gets the caveat.
        ex.attach_sae(_random_sae(768, 256, 6, "resid_pre", source_format="saelens"))
        flagged = ex.extract(out, provider=provider)
        assert flagged and all("sae_input_warning" in f.meta for f in flagged)
