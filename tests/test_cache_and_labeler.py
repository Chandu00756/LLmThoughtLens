"""Tests for :class:`ActivationCache` (SAE training data collection) and
:class:`FeatureLabeler` (LLM-driven auto-labelling of SAE features).

The cache is exercised against the deterministic :class:`MockProvider`, so
every stored activation can be compared byte-for-byte with a direct
``provider.run`` call.  The labeler is driven by a scripted SAE stub whose
codes we control exactly, plus a recording labelling provider, so we can
assert which contexts are shown to the labelling LLM and how its reply is
sanitised.  One integration test runs a real (tiny) torch SAE end to end.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from LLmThoughtLens.features.cache import ActivationCache
from LLmThoughtLens.features.labeler import FeatureLabeler, _clean_label
from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.providers.mock_provider import MockProvider

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _CountingMock(MockProvider):
    """MockProvider that records every prompt it is asked to run."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.prompts: list[str] = []

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        self.prompts.append(prompt)
        return super().run(prompt, **kwargs)


class _BlackBox(BaseProvider):
    evidence_kind = "black_box"

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:  # pragma: no cover
        return ProviderOutput(prompt=prompt)


class _WhiteBoxWithoutActivations(BaseProvider):
    """Claims white-box support but returns no activations (misconfigured HF)."""

    evidence_kind = "white_box"

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        return ProviderOutput(prompt=prompt, tokens=prompt.split(), evidence_kind="white_box")


class _ScriptedSAE:
    """Minimal SAE stand-in: ``encode`` returns pre-baked codes."""

    def __init__(self, codes: np.ndarray) -> None:
        self.codes = np.asarray(codes, dtype=np.float32)
        self._labels: dict[int, str] = {}
        self.encode_calls = 0

    def encode(self, activations: np.ndarray) -> np.ndarray:
        self.encode_calls += 1
        assert activations.shape[0] == self.codes.shape[0]
        return self.codes

    def set_label(self, feature_id: int, label: str) -> None:
        self._labels[int(feature_id)] = str(label)

    @property
    def labels(self) -> dict[int, str]:
        return dict(self._labels)


class _RecordingLabeler(BaseProvider):
    """Labelling provider that replies with a fixed completion and logs prompts."""

    evidence_kind = "black_box"

    def __init__(self, reply_tokens: list[str], completion: str = "") -> None:
        self.reply_tokens = reply_tokens
        self.completion = completion
        self.prompts: list[str] = []

    def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
        self.prompts.append(prompt)
        return ProviderOutput(
            prompt=prompt,
            tokens=list(self.reply_tokens),
            meta={"completion": self.completion},
        )


# ===========================================================================
# ActivationCache
# ===========================================================================


class TestActivationCacheCollect:
    def test_rejects_black_box_provider(self):
        with pytest.raises(ValueError, match="white-box"):
            ActivationCache(_BlackBox())

    def test_collect_stores_exact_layer_activations(self):
        provider = MockProvider(n_layers=3, n_heads=2, d_model=8, seed=1)
        prompts = ["the cat sat", "hello brave new world"]
        cache = ActivationCache(provider, layer=2)
        assert cache.collect(prompts) is cache  # chaining

        expected = np.concatenate([provider.run(p).activations[2] for p in prompts], axis=0)
        arr = cache.array()
        assert arr.shape == (7, 8)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, expected)
        assert len(cache) == 7
        assert cache.d_model == 8

    def test_max_tokens_truncates_and_stops_running_prompts(self):
        provider = _CountingMock(n_layers=2, n_heads=1, d_model=4, seed=0)
        cache = ActivationCache(provider, layer=0, max_tokens=5)
        cache.collect(["a b c", "d e f g", "never run"])
        assert len(cache) == 5
        assert cache.array().shape == (5, 4)
        # The third prompt is skipped because the cap was already reached.
        assert provider.prompts == ["a b c", "d e f g"]
        second = provider.run("d e f g").activations[0][:2]
        np.testing.assert_array_equal(cache.array()[3:], second)

    def test_collect_stores_provider_tokens_and_row_mapping(self):
        provider = MockProvider(n_layers=1, n_heads=1, d_model=4)
        cache = ActivationCache(provider).collect(["the cat sat", "hello world"])
        assert cache._tokens == ["the", "cat", "sat", "hello", "world"]
        assert cache._token_prompt == [0, 0, 0, 1, 1]
        assert cache._token_pos == [0, 1, 2, 0, 1]
        assert len(cache._tokens) == cache.array().shape[0]

    def test_max_tokens_truncates_stored_tokens_too(self, tmp_path):
        cache = ActivationCache(MockProvider(n_layers=1, n_heads=1, d_model=4), max_tokens=4)
        cache.collect(["a b c", "d e f g"])
        cache.save(tmp_path / "t.npz")
        streams, positions = ActivationCache.token_streams(ActivationCache.load(tmp_path / "t.npz"))
        assert streams == [["a", "b", "c"], ["d"]]
        assert positions == [(0, 0), (0, 1), (0, 2), (1, 0)]

    def test_provider_with_misaligned_tokens_warns_and_pads(self, tmp_path):
        class _Misaligned(MockProvider):
            def run(self, prompt: str, **kwargs: Any) -> ProviderOutput:
                out = super().run(prompt, **kwargs)
                out.tokens = out.tokens[:1]  # fewer token strings than activation rows
                return out

        cache = ActivationCache(_Misaligned(n_layers=1, n_heads=1, d_model=4))
        with pytest.warns(RuntimeWarning, match="1 tokens but 3 activation rows"):
            cache.collect(["x y z"])
        assert cache._tokens == ["x", "<?>", "<?>"]
        cache.save(tmp_path / "m.npz")
        assert ActivationCache.load(tmp_path / "m.npz")["meta"]["tokens_aligned"] is False

    def test_tiny_huggingface_tokens_are_the_tokenizers_own(self, tmp_path):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from _tiny_hf import N_EMBD, WordTokenizer, make_provider

        provider = make_provider()
        cache = ActivationCache(provider, layer=1, max_tokens=5)
        cache.collect(["paris is the capital", "of france"])
        tok = WordTokenizer()
        words = ["paris", "is", "the", "capital", "of"]  # capped at 5 tokens
        assert cache._tokens == [tok.decode([tok.word_id(w)]) for w in words]
        assert cache.array().shape == (5, N_EMBD)
        cache.save(tmp_path / "hf.npz")
        loaded = ActivationCache.load(tmp_path / "hf.npz")
        assert loaded["meta"]["provider"] == "huggingface"
        streams, _ = ActivationCache.token_streams(loaded)
        assert [len(s) for s in streams] == [4, 1]

    def test_custom_dtype(self):
        cache = ActivationCache(MockProvider(n_layers=1, n_heads=1, d_model=4), dtype=np.float16)
        cache.collect(["x y"])
        assert cache.array().dtype == np.float16

    def test_layer_out_of_range(self):
        cache = ActivationCache(MockProvider(n_layers=2, n_heads=1, d_model=4), layer=5)
        with pytest.raises(IndexError, match="layer=5 out of range"):
            cache.collect(["hello"])

    def test_missing_activations_raises(self):
        cache = ActivationCache(_WhiteBoxWithoutActivations())
        with pytest.raises(RuntimeError, match="no activations"):
            cache.collect(["hello"])

    def test_verbose_reports_progress(self, capsys):
        cache = ActivationCache(MockProvider(n_layers=1, n_heads=1, d_model=4))
        cache.collect(["one two", "three"], verbose=True)
        out = capsys.readouterr().out
        assert "prompt 1: +2 tokens (total 2)" in out
        assert "prompt 2: +1 tokens (total 3)" in out

    def test_empty_cache_array(self):
        cache = ActivationCache(MockProvider(n_layers=1, n_heads=1, d_model=4))
        assert cache.array().shape == (0, 0)
        assert len(cache) == 0
        assert cache.d_model == 0


class TestActivationCachePersistence:
    def test_save_load_roundtrip(self, tmp_path):
        provider = MockProvider(n_layers=2, n_heads=2, d_model=6, seed=3)
        cache = ActivationCache(provider, layer=1).collect(["the capital of France is"])
        path = tmp_path / "acts.npz"
        cache.save(path)

        loaded = ActivationCache.load(path)
        np.testing.assert_array_equal(loaded["activations"], cache.array())
        assert loaded["meta"] == {
            "layer": 1,
            "n_tokens": 5,
            "d_model": 6,
            "provider": "mock",
            "model_id": provider.model_id,
            "tokens_aligned": True,
        }
        assert loaded["tokens"] == ["the", "capital", "of", "France", "is"]
        assert loaded["token_prompt"].tolist() == [0] * 5
        assert loaded["token_pos"].tolist() == [0, 1, 2, 3, 4]
        assert loaded["prompts"] == ["the capital of France is"]
        assert "prompt_lines" not in loaded  # collect() was not given line numbers
        assert ActivationCache.has_tokens(loaded)

    def test_save_empty_cache(self, tmp_path):
        cache = ActivationCache(MockProvider(n_layers=1, n_heads=1, d_model=4))
        path = tmp_path / "empty.npz"
        cache.save(path)
        loaded = ActivationCache.load(path)
        assert loaded["activations"].shape == (0, 0)
        assert loaded["meta"]["n_tokens"] == 0
        assert loaded["meta"]["d_model"] == 0
        assert loaded["tokens"] == [] and loaded["prompts"] == []
        assert ActivationCache.token_streams(loaded) == ([], [])

    def test_line_numbers_roundtrip(self, tmp_path):
        cache = ActivationCache(MockProvider(n_layers=1, n_heads=1, d_model=4))
        cache.collect(["a b", "c"], line_numbers=[3, 7])
        cache.save(tmp_path / "a.npz")
        loaded = ActivationCache.load(tmp_path / "a.npz")
        assert loaded["prompt_lines"].tolist() == [3, 7]
        assert loaded["token_prompt"].tolist() == [0, 0, 1]

    def test_line_numbers_must_be_parallel_to_prompts(self):
        cache = ActivationCache(MockProvider(n_layers=1, n_heads=1, d_model=4))
        with pytest.raises(ValueError, match="parallel to prompts"):
            cache.collect(["a", "b"], line_numbers=[1])

    def test_legacy_file_without_tokens_still_loads(self, tmp_path):
        path = tmp_path / "old.npz"
        np.savez_compressed(path, activations=np.ones((3, 2)), meta__layer=4)
        loaded = ActivationCache.load(path)
        assert loaded["activations"].shape == (3, 2)
        assert loaded["meta"] == {"layer": 4}
        assert not ActivationCache.has_tokens(loaded)
        with pytest.raises(ValueError, match="no stored tokens"):
            ActivationCache.token_streams(loaded)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("tokens", ["a", "b"], "3 activation rows but 2 tokens"),
            ("token_pos", np.array([0, 2, 1]), "out of order"),
            ("token_prompt", np.array([0, -1, 0]), "out of order"),
        ],
    )
    def test_token_streams_rejects_inconsistent_arrays(self, field, value, match):
        data = {
            "activations": np.zeros((3, 2)),
            "tokens": ["a", "b", "c"],
            "token_prompt": np.array([0, 0, 0]),
            "token_pos": np.array([0, 1, 2]),
            "prompts": ["a b c"],
        }
        assert ActivationCache.token_streams(data) == ([["a", "b", "c"]], [(0, 0), (0, 1), (0, 2)])
        data[field] = value
        with pytest.raises(ValueError, match=match):
            ActivationCache.token_streams(data)


# ===========================================================================
# FeatureLabeler
# ===========================================================================

STREAMS = [["the", "cat", "sat"], ["on", "a", "mat"]]  # 6 flat positions


def _codes(col0: list[float], col1: list[float] | None = None) -> np.ndarray:
    cols = [col0, col1 if col1 is not None else [0.0] * len(col0)]
    return np.array(cols, dtype=np.float32).T  # (N, 2)


class TestFeatureLabelerSingleFeature:
    def test_prompt_contains_top_contexts_in_activation_order(self):
        # Feature 0 fires hardest on "mat" (flat 5), then "cat" (flat 1); zeros are skipped.
        sae = _ScriptedSAE(_codes([0.0, 0.5, 0.0, 0.0, 0.0, 2.0]))
        labeler_llm = _RecordingLabeler(["Animals"])
        fl = FeatureLabeler(sae, labeler_llm, top_n_contexts=5, context_window=1)

        label = fl.label_feature(0, np.zeros((6, 4)), STREAMS)

        assert label == "animals"
        assert sae.labels == {0: "animals"}
        (prompt,) = labeler_llm.prompts
        assert "You will be shown 2 short text snippets" in prompt
        snippet_lines = [ln for ln in prompt.splitlines() if ln.startswith("- ")]
        assert snippet_lines == ["- a «mat»", "- the «cat» sat"]

    def test_top_n_contexts_limits_snippets(self):
        sae = _ScriptedSAE(_codes([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        labeler_llm = _RecordingLabeler(["x"])
        FeatureLabeler(sae, labeler_llm, top_n_contexts=2, context_window=0).label_feature(
            0, np.zeros((6, 4)), STREAMS
        )
        snippet_lines = [ln for ln in labeler_llm.prompts[0].splitlines() if ln.startswith("- ")]
        assert snippet_lines == ["- «mat»", "- «a»"]

    def test_dead_feature_gets_fallback_without_calling_llm(self):
        sae = _ScriptedSAE(_codes([0.0] * 6))
        labeler_llm = _RecordingLabeler(["ignored"])
        label = FeatureLabeler(sae, labeler_llm).label_feature(0, np.zeros((6, 4)), STREAMS)
        assert label == "unlabelled_feature"
        assert labeler_llm.prompts == []
        assert sae.labels == {}

    def test_unusable_llm_reply_falls_back(self):
        sae = _ScriptedSAE(_codes([1.0, 0, 0, 0, 0, 0]))
        label = FeatureLabeler(sae, _RecordingLabeler(["!!!", "???"])).label_feature(
            0, np.zeros((6, 4)), STREAMS
        )
        assert label == "unlabelled_feature"
        assert sae.labels == {0: "unlabelled_feature"}

    def test_completion_meta_used_when_provider_returns_no_tokens(self):
        sae = _ScriptedSAE(_codes([1.0, 0, 0, 0, 0, 0]))
        llm = _RecordingLabeler([], completion="Sentence Start\nextra explanation")
        label = FeatureLabeler(sae, llm).label_feature(0, np.zeros((6, 4)), STREAMS)
        assert label == "sentence start"

    def test_multi_word_label_from_black_box_labeler_keeps_spaces(self):
        """Regression: tokens were joined with '' ('proper nouns' -> 'propernouns')."""
        sae = _ScriptedSAE(_codes([1.0, 0, 0, 0, 0, 0]))
        # Exactly what OpenAIProvider/AnthropicProvider return for the reply "proper nouns".
        llm = _RecordingLabeler(["proper", "nouns"], completion="proper nouns")
        label = FeatureLabeler(sae, llm).label_feature(0, np.zeros((6, 4)), STREAMS)
        assert label == "proper nouns"

    def test_black_box_tokens_are_joined_with_spaces_without_completion(self):
        sae = _ScriptedSAE(_codes([1.0, 0, 0, 0, 0, 0]))
        llm = _RecordingLabeler(["proper", "nouns"], completion="")
        assert FeatureLabeler(sae, llm).label_feature(0, np.zeros((6, 4)), STREAMS) == (
            "proper nouns"
        )

    def test_completion_wins_over_tokens(self):
        sae = _ScriptedSAE(_codes([1.0, 0, 0, 0, 0, 0]))
        llm = _RecordingLabeler(["ignored"], completion="Geographic Names")
        assert FeatureLabeler(sae, llm).label_feature(0, np.zeros((6, 4)), STREAMS) == (
            "geographic names"
        )

    def test_white_box_labeler_does_not_echo_the_prompt(self):
        """White-box providers return the *prompt's* tokens; use their prediction."""
        sae = _ScriptedSAE(_codes([1.0, 0, 0, 0, 0, 0]))
        llm = _CountingMock(n_layers=1, n_heads=1, d_model=4)
        label = FeatureLabeler(sae, llm).label_feature(0, np.zeros((6, 4)), STREAMS)
        (prompt,) = llm.prompts
        out = MockProvider(n_layers=1, n_heads=1, d_model=4).run(prompt)
        assert out.tokens[:3] == ["You", "will", "be"]  # what the old code labelled with
        assert label == (_clean_label(out.output_token) or "unlabelled_feature")
        assert "you will be" not in label

    def test_token_count_mismatch_raises_clear_error(self):
        sae = _ScriptedSAE(_codes([1.0, 0, 0, 0, 0, 0]))
        llm = _RecordingLabeler(["x"])
        with pytest.raises(ValueError, match="activations has 6 rows but token_streams contain 5"):
            FeatureLabeler(sae, llm).label_feature(
                0, np.zeros((6, 4)), [["a", "b"], ["c", "d", "e"]]
            )
        with pytest.raises(ValueError, match="must align 1:1"):
            FeatureLabeler(sae, llm).label_all(np.zeros((6, 4)), [["a"] * 7])
        assert llm.prompts == [] and sae.encode_calls == 0  # validated before any work

    def test_explicit_positions_map_rows_to_tokens(self):
        # Rows are deliberately NOT in flattened order.
        positions = [(1, 2), (0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        sae = _ScriptedSAE(_codes([3.0, 0, 0, 0, 0, 0]))  # fires on row 0 -> "mat"
        llm = _RecordingLabeler(["Floor"])
        fl = FeatureLabeler(sae, llm, context_window=1)
        assert fl.label_feature(0, np.zeros((6, 4)), STREAMS, positions=positions) == "floor"
        assert [ln for ln in llm.prompts[0].splitlines() if ln.startswith("- ")] == ["- a «mat»"]

    @pytest.mark.parametrize(
        ("positions", "match"),
        [
            ([(0, 0)] * 5, "positions has 5 entries"),
            ([(0, 0)] * 5 + [(2, 0)], r"positions\[5\] = \(2, 0\) is outside"),
            ([(0, 0)] * 5 + [(1, 3)], r"positions\[5\] = \(1, 3\) is outside"),
            ([(0, 0)] * 5 + [(0, -1)], "is outside"),
        ],
    )
    def test_invalid_explicit_positions_raise(self, positions, match):
        sae = _ScriptedSAE(_codes([1.0, 0, 0, 0, 0, 0]))
        with pytest.raises(ValueError, match=match):
            FeatureLabeler(sae, _RecordingLabeler(["x"])).label_feature(
                0, np.zeros((6, 4)), STREAMS, positions=positions
            )


class TestFeatureLabelerBulk:
    def test_label_all_encodes_once_and_accepts_positions(self):
        sae = _ScriptedSAE(_codes([0, 0, 0, 0, 0, 1.0], [2.0, 0, 0, 0, 0, 0]))
        llm = _RecordingLabeler(["Thing"])
        positions = [(1, 2), (0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        labels = FeatureLabeler(sae, llm, context_window=0).label_all(
            np.zeros((6, 4)), STREAMS, positions=positions
        )
        assert labels == {0: "thing", 1: "thing"}
        assert sae.encode_calls == 1
        snippets = [ln for p in llm.prompts for ln in p.splitlines() if ln.startswith("- ")]
        assert snippets == ["- «a»", "- «mat»"]  # row 5 -> (1, 1) "a"; row 0 -> (1, 2) "mat"

    def test_label_all_only_visits_live_features(self, capsys):
        sae = _ScriptedSAE(_codes([0, 0, 0, 0, 0, 0], [0, 0, 3.0, 0, 0, 0]))
        llm = _RecordingLabeler(["Verbs"])
        labels = FeatureLabeler(sae, llm).label_all(np.zeros((6, 4)), STREAMS, verbose=True)
        assert labels == {1: "verbs"}
        assert len(llm.prompts) == 1
        assert "feature     1 → verbs" in capsys.readouterr().out

    def test_label_all_explicit_ids_include_dead_features(self):
        sae = _ScriptedSAE(_codes([0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]))
        labels = FeatureLabeler(sae, _RecordingLabeler(["Noun"])).label_all(
            np.zeros((6, 4)), STREAMS, feature_ids=[0, 1]
        )
        assert labels == {0: "unlabelled_feature", 1: "noun"}

    def test_save_labels_writes_sae_labels_as_json(self, tmp_path):
        sae = _ScriptedSAE(_codes([1.0, 0, 0, 0, 0, 0]))
        fl = FeatureLabeler(sae, _RecordingLabeler(["Cats"]))
        fl.label_feature(0, np.zeros((6, 4)), STREAMS)
        out = tmp_path / "labels.json"
        fl.save_labels(out)
        assert json.loads(out.read_text()) == {"0": "cats"}

    def test_flat_to_positions(self):
        fl = FeatureLabeler(_ScriptedSAE(np.zeros((1, 1))), _RecordingLabeler([]))
        assert fl._flat_to_positions([["a", "b"], [], ["c"]]) == [(0, 0), (0, 1), (2, 0)]


class TestCleanLabel:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Proper Nouns.  ", "proper nouns"),
            ('"Quoted, label!"', "quoted label"),
            ("First line\nsecond line", "first line"),
            ("one two three four five six seven", "one two three four five"),
            ("snake_case-label", "snake_case-label"),
            ("", ""),
            ("!!!", ""),
        ],
    )
    def test_sanitises(self, raw, expected):
        assert _clean_label(raw) == expected

    def test_truncates_to_sixty_chars(self):
        out = _clean_label("a" * 100)
        assert out == "a" * 60


class TestFeatureLabelerWithRealSAE:
    def test_end_to_end_with_torch_sae(self):
        pytest.importorskip("torch")
        from LLmThoughtLens.features.sae import SAEConfig, SparseAutoencoder

        provider = MockProvider(n_layers=2, n_heads=1, d_model=8, seed=5)
        corpus = ["the cat sat on the mat", "dogs chase cats"]
        cache = ActivationCache(provider, layer=1).collect(corpus)
        acts = cache.array()
        streams = [p.split() for p in corpus]
        assert acts.shape[0] == sum(len(s) for s in streams)

        sae = SparseAutoencoder(SAEConfig(input_dim=8, dict_size=16, k=2, device="cpu", seed=0))
        llm = _RecordingLabeler(["Feline"])
        labels = FeatureLabeler(sae, llm, top_n_contexts=3).label_all(acts, streams)

        codes = sae.encode(acts)
        live = {int(i) for i in np.where((codes != 0).any(axis=0))[0]}
        assert set(labels) == live
        assert live, "TopK SAE must activate at least one feature"
        assert set(labels.values()) == {"feline"}
        assert sae.labels == labels
        assert len(llm.prompts) == len(live)
