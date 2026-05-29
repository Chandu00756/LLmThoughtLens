"""Tests for providers — MockProvider, BaseProvider contract, and ProviderOutput."""

from __future__ import annotations

import numpy as np
import pytest

from llmscope.providers.base import BaseProvider, ProviderOutput
from llmscope.providers.mock_provider import MockProvider


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

    def test_with_values(self):
        out = ProviderOutput(
            prompt="hi",
            tokens=["hi"],
            token_ids=[1],
            meta={"model": "mock"},
        )
        assert out.tokens == ["hi"]
        assert out.token_ids == [1]
        assert out.meta["model"] == "mock"


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


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------

class TestMockProvider:
    def test_instantiation(self):
        mp = MockProvider()
        assert mp.n_layers == 6
        assert mp.d_model == 64
        assert mp.seed == 42

    def test_run_returns_provider_output(self, mock_provider: MockProvider):
        out = mock_provider.run("Hello world")
        assert isinstance(out, ProviderOutput)

    def test_tokens_match_words(self, mock_provider: MockProvider):
        out = mock_provider.run("foo bar baz")
        assert out.tokens == ["foo", "bar", "baz"]

    def test_token_ids_length(self, mock_provider: MockProvider):
        out = mock_provider.run("a b c d")
        assert len(out.token_ids) == len(out.tokens)

    def test_activations_shape(self, mock_provider: MockProvider):
        out = mock_provider.run("hello world")
        n_tokens = len(out.tokens)
        assert out.activations is not None
        assert out.activations.shape == (
            mock_provider.n_layers,
            n_tokens,
            mock_provider.d_model,
        )

    def test_activations_dtype(self, mock_provider: MockProvider):
        out = mock_provider.run("hello")
        assert out.activations is not None
        assert out.activations.dtype == np.float32

    def test_attentions_shape(self, mock_provider: MockProvider):
        out = mock_provider.run("a b c")
        n_tokens = len(out.tokens)
        assert out.attentions is not None
        assert out.attentions.shape == (
            mock_provider.n_layers,
            mock_provider.n_heads,
            n_tokens,
            n_tokens,
        )

    def test_attentions_sum_to_one(self, mock_provider: MockProvider):
        out = mock_provider.run("test attention")
        assert out.attentions is not None
        sums = out.attentions.sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)

    def test_logits_shape(self, mock_provider: MockProvider):
        out = mock_provider.run("hello world")
        n_tokens = len(out.tokens)
        assert out.logits is not None
        assert out.logits.shape == (n_tokens, mock_provider.vocab_size)

    def test_top_tokens_is_list_of_tuples(self, mock_provider: MockProvider):
        out = mock_provider.run("hello")
        assert isinstance(out.top_tokens, list)
        assert len(out.top_tokens) > 0
        for tok, prob in out.top_tokens:
            assert isinstance(tok, str)
            assert 0.0 <= prob <= 1.0

    def test_determinism_same_prompt(self):
        mp = MockProvider(seed=7)
        out1 = mp.run("reproducible")
        out2 = mp.run("reproducible")
        np.testing.assert_array_equal(out1.activations, out2.activations)
        np.testing.assert_array_equal(out1.attentions, out2.attentions)

    def test_different_prompts_differ(self):
        mp = MockProvider(seed=7)
        out1 = mp.run("prompt one two")
        out2 = mp.run("completely different")
        # Different token counts mean different shapes — just check they're not identical
        assert out1.activations is not None
        assert out2.activations is not None
        assert out1.activations.shape != out2.activations.shape or not np.array_equal(
            out1.activations, out2.activations
        )

    def test_different_seeds_differ(self):
        mp1 = MockProvider(seed=1)
        mp2 = MockProvider(seed=2)
        out1 = mp1.run("same prompt")
        out2 = mp2.run("same prompt")
        assert out1.activations is not None
        assert out2.activations is not None
        assert not np.array_equal(out1.activations, out2.activations)

    def test_meta_keys(self, mock_provider: MockProvider):
        out = mock_provider.run("test")
        assert out.meta["provider"] == "MockProvider"
        assert "n_layers" in out.meta
        assert "d_model" in out.meta
        assert "seed" in out.meta

    def test_name_property(self):
        mp = MockProvider()
        assert mp.name == "mock"

    def test_repr(self):
        mp = MockProvider()
        assert "MockProvider" in repr(mp)
