"""Tests for providers — MockProvider, BaseProvider contract, ProviderOutput envelope."""

from __future__ import annotations

import numpy as np
import pytest
from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput
from LLmThoughtLens.providers.mock_provider import MockProvider

# ---------------------------------------------------------------------------
# ProviderOutput contract
# ---------------------------------------------------------------------------


class TestProviderOutput:
    def test_defaults(self):
        out = ProviderOutput(prompt="test")
        assert out.prompt == "test"
        assert out.tokens == []
        assert out.token_ids == []
        assert out.activations is None
        assert out.attentions is None
        assert out.logits is None
        assert out.top_tokens == []
        assert out.meta == {}
        assert out.evidence_kind == "black_box"

    def test_accessors_for_empty(self):
        out = ProviderOutput(prompt="x")
        assert out.n_tokens == 0
        assert out.n_layers == 0
        assert out.d_model == 0
        assert out.output_token == ""
        assert out.output_prob == 0.0
        assert out.has_internals is False

    def test_with_values(self):
        out = ProviderOutput(
            prompt="hi",
            tokens=["hi"],
            token_ids=[1],
            meta={"model": "mock"},
            top_tokens=[("hi", 0.42)],
            evidence_kind="white_box",
        )
        assert out.tokens == ["hi"]
        assert out.token_ids == [1]
        assert out.meta["model"] == "mock"
        assert out.output_token == "hi"
        assert out.output_prob == pytest.approx(0.42)
        assert out.evidence_kind == "white_box"


# ---------------------------------------------------------------------------
# BaseProvider ABC
# ---------------------------------------------------------------------------


class TestBaseProviderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseProvider()  # type: ignore[abstract]

    def test_concrete_subclass_requires_run(self):
        class Incomplete(BaseProvider):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_ok(self):
        class Minimal(BaseProvider):
            def run(self, prompt, **kwargs):
                return ProviderOutput(prompt=prompt)

        m = Minimal()
        out = m.run("hello")
        assert out.prompt == "hello"
        # Defaults
        assert m.evidence_kind == "black_box"
        assert m.supports_internals is False

    def test_run_with_intervention_default_delegates(self):
        class Minimal(BaseProvider):
            def run(self, prompt, **kwargs):
                return ProviderOutput(prompt=prompt + "!" + str(kwargs.get("temperature", "")))

        m = Minimal()
        out = m.run_with_intervention("x", interventions=None, temperature=0.5)
        assert out.prompt == "x!0.5"


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------


class TestMockProvider:
    def test_instantiation_defaults(self):
        mp = MockProvider()
        assert mp.n_layers == 6
        assert mp.d_model == 64
        assert mp.seed == 42

    def test_d_model_divides_heads(self):
        with pytest.raises(ValueError):
            MockProvider(n_layers=2, n_heads=3, d_model=8)  # 8 not divisible by 3

    def test_run_returns_provider_output(self, mock_provider: MockProvider):
        out = mock_provider.run("Hello world")
        assert isinstance(out, ProviderOutput)

    def test_evidence_kind_is_white_box(self, mock_provider: MockProvider):
        out = mock_provider.run("hi")
        assert out.evidence_kind == "white_box"
        assert out.has_internals is True

    def test_tokens_match_words(self, mock_provider: MockProvider):
        out = mock_provider.run("foo bar baz")
        assert out.tokens == ["foo", "bar", "baz"]

    def test_token_ids_length(self, mock_provider: MockProvider):
        out = mock_provider.run("a b c d")
        assert len(out.token_ids) == len(out.tokens)

    def test_activations_shape(self, mock_provider: MockProvider):
        out = mock_provider.run("hello world")
        assert out.activations is not None
        assert out.activations.shape == (
            mock_provider.n_layers,
            len(out.tokens),
            mock_provider.d_model,
        )

    def test_activations_dtype(self, mock_provider: MockProvider):
        out = mock_provider.run("hello")
        assert out.activations.dtype == np.float32

    def test_attentions_shape(self, mock_provider: MockProvider):
        out = mock_provider.run("a b c")
        assert out.attentions.shape == (
            mock_provider.n_layers,
            mock_provider.n_heads,
            len(out.tokens),
            len(out.tokens),
        )

    def test_attentions_sum_to_one(self, mock_provider: MockProvider):
        out = mock_provider.run("test attention")
        sums = out.attentions.sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)

    def test_causal_attention_mask(self, mock_provider: MockProvider):
        out = mock_provider.run("a b c d")
        # Upper triangle (future positions) must be exactly zero.
        upper = np.triu(np.abs(out.attentions), k=1)
        assert float(upper.sum()) == pytest.approx(0.0)

    def test_logits_shape_is_last_token_only(self, mock_provider: MockProvider):
        out = mock_provider.run("hello world")
        # logits is the LAST-token vector only, not full (T, V).
        assert out.logits.shape == (mock_provider.vocab_size,)

    def test_top_tokens_are_real_probabilities(self, mock_provider: MockProvider):
        out = mock_provider.run("hello")
        assert isinstance(out.top_tokens, list) and len(out.top_tokens) > 0
        probs = [p for _, p in out.top_tokens]
        # Probabilities are in [0, 1] and the list is sorted descending.
        for p in probs:
            assert 0.0 <= p <= 1.0
        assert probs == sorted(probs, reverse=True)

    def test_determinism_same_prompt(self):
        mp = MockProvider(seed=7)
        a = mp.run("reproducible").activations
        b = mp.run("reproducible").activations
        np.testing.assert_array_equal(a, b)

    def test_different_prompts_differ(self):
        mp = MockProvider(seed=7)
        out_a = mp.run("aaa bbb")
        out_b = mp.run("ccc ddd")
        # Same shapes (same word count) → arrays must actually differ.
        assert not np.array_equal(out_a.activations, out_b.activations)

    def test_different_seeds_differ(self):
        a = MockProvider(seed=1).run("same prompt").activations
        b = MockProvider(seed=2).run("same prompt").activations
        assert not np.array_equal(a, b)

    def test_meta_keys(self, mock_provider: MockProvider):
        out = mock_provider.run("test")
        for key in ("provider", "model", "n_layers", "d_model", "seed", "evidence_note"):
            assert key in out.meta

    def test_name_property(self):
        assert MockProvider().name == "mock"

    def test_model_id(self):
        mp = MockProvider(n_layers=3, n_heads=2, d_model=8)
        assert mp.model_id == "mock-L3-H2-D8"

    def test_supports_internals(self):
        assert MockProvider().supports_internals is True
