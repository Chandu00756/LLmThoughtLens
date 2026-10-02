"""Behavioural tests for the pure-NumPy math helpers and the whitespace tokenizer utils.

These helpers sit under the feature extractor, circuit tracer and residual-stream
view, so every function is checked against an independently computed reference
value rather than just "returns an array".
"""

from __future__ import annotations

import numpy as np
import pytest
from LLmThoughtLens.utils.math_utils import (
    cosine_sim,
    l2_normalise,
    pca_2d,
    softmax,
    topk_indices,
    topk_mask,
)
from LLmThoughtLens.utils.tokenizer_utils import (
    MASK_TOKEN,
    mask_positions,
    replace_token,
    token_join,
    whitespace_tokens,
)

# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------


class TestSoftmax:
    def test_matches_reference_and_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        ref = np.exp(x) / np.exp(x).sum()
        out = softmax(x)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, ref, rtol=1e-6)
        assert out.sum() == pytest.approx(1.0, abs=1e-6)

    def test_numerically_stable_for_huge_logits(self):
        out = softmax(np.array([1000.0, 1000.0, -1000.0]))
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, [0.5, 0.5, 0.0], atol=1e-6)

    def test_axis_argument_normalises_along_requested_axis(self):
        x = np.arange(6, dtype=np.float64).reshape(2, 3)
        rows = softmax(x, axis=-1)
        cols = softmax(x, axis=0)
        np.testing.assert_allclose(rows.sum(axis=1), [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(cols.sum(axis=0), [1.0, 1.0, 1.0], atol=1e-6)
        # Shift invariance: every row of `x` differs by a constant, so rows are equal.
        np.testing.assert_allclose(rows[0], rows[1], atol=1e-6)


# ---------------------------------------------------------------------------
# cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSim:
    def test_identical_orthogonal_and_opposite(self):
        a = np.array([1.0, 2.0, 3.0])
        assert cosine_sim(a, a) == pytest.approx(1.0)
        assert cosine_sim(a, -a) == pytest.approx(-1.0)
        assert cosine_sim(np.array([1.0, 0.0]), np.array([0.0, 5.0])) == pytest.approx(0.0)

    def test_matches_reference_formula(self):
        a = np.array([0.3, -1.2, 4.0])
        b = np.array([2.0, 0.5, -0.7])
        ref = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cosine_sim(a, b) == pytest.approx(ref)

    def test_zero_vector_returns_zero_instead_of_nan(self):
        assert cosine_sim(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0
        assert cosine_sim(np.array([1.0, 2.0]), np.zeros(2)) == 0.0

    def test_multi_dimensional_inputs_are_flattened(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert cosine_sim(a, a.ravel()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# top-k helpers
# ---------------------------------------------------------------------------


class TestTopkIndices:
    def test_returns_indices_sorted_by_value_descending(self):
        x = np.array([0.1, 5.0, -2.0, 3.0, 4.0])
        assert topk_indices(x, 3).tolist() == [1, 4, 3]

    def test_k_larger_than_axis_is_clipped(self):
        x = np.array([2.0, 1.0, 3.0])
        assert topk_indices(x, 10).tolist() == [2, 0, 1]

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_raises(self, k):
        with pytest.raises(ValueError, match="k must be > 0"):
            topk_indices(np.arange(3.0), k)

    def test_two_dimensional_per_row(self):
        x = np.array([[1.0, 9.0, 5.0], [7.0, 2.0, 8.0]])
        idx = topk_indices(x, 2, axis=-1)
        assert idx.tolist() == [[1, 2], [2, 0]]


class TestTopkMask:
    def test_keeps_only_top_k_values(self):
        x = np.array([0.5, -1.0, 3.0, 2.0])
        out = topk_mask(x, 2)
        np.testing.assert_array_equal(out, [0.0, 0.0, 3.0, 2.0])

    def test_k_equal_to_length_returns_a_copy(self):
        x = np.array([1.0, 2.0])
        out = topk_mask(x, 5)
        np.testing.assert_array_equal(out, x)
        assert out is not x
        out[0] = 99.0
        assert x[0] == 1.0  # original untouched

    def test_non_positive_k_raises(self):
        with pytest.raises(ValueError):
            topk_mask(np.arange(4.0), 0)

    def test_row_wise_mask_preserves_shape(self):
        x = np.array([[1.0, 9.0, 5.0], [7.0, 2.0, 8.0]])
        out = topk_mask(x, 1, axis=-1)
        np.testing.assert_array_equal(out, [[0.0, 9.0, 0.0], [0.0, 0.0, 8.0]])


# ---------------------------------------------------------------------------
# L2 normalise / PCA
# ---------------------------------------------------------------------------


class TestL2Normalise:
    def test_rows_have_unit_norm_and_direction_preserved(self):
        x = np.array([[3.0, 4.0], [0.0, 2.0]])
        out = l2_normalise(x)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0])
        np.testing.assert_allclose(out[0], [0.6, 0.8])

    def test_zero_row_stays_zero(self):
        out = l2_normalise(np.zeros((1, 3)))
        np.testing.assert_array_equal(out, np.zeros((1, 3)))


class TestPCA2D:
    def test_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        out = pca_2d(rng.standard_normal((10, 6)))
        assert out.shape == (10, 2)
        assert out.dtype == np.float32

    def test_fewer_than_two_samples_returns_zeros(self):
        out = pca_2d(np.ones((1, 4)))
        np.testing.assert_array_equal(out, np.zeros((1, 2), dtype=np.float32))

    def test_collinear_points_project_onto_first_component_only(self):
        t = np.linspace(-2.0, 2.0, 9)
        direction = np.array([1.0, 2.0, 2.0]) / 3.0
        x = np.outer(t, direction) + np.array([5.0, -1.0, 0.5])
        out = pca_2d(x)
        # All variance lands on PC1; PC2 is numerically zero.
        np.testing.assert_allclose(np.abs(out[:, 0]), np.abs(t), atol=1e-5)
        np.testing.assert_allclose(out[:, 1], 0.0, atol=1e-5)
        # Projection is centred.
        assert out[:, 0].mean() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# tokenizer utils
# ---------------------------------------------------------------------------


class TestTokenizerUtils:
    def test_whitespace_tokens_splits_on_any_whitespace(self):
        assert whitespace_tokens("the  cat\tsat\n") == ["the", "cat", "sat"]

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_whitespace_tokens_empty_sentinel(self, text):
        assert whitespace_tokens(text) == ["<empty>"]

    def test_replace_token_returns_copy(self):
        toks = ["a", "b", "c"]
        out = replace_token(toks, 1)
        assert out == ["a", MASK_TOKEN, "c"]
        assert toks == ["a", "b", "c"]
        assert replace_token(toks, 2, "Z") == ["a", "b", "Z"]

    @pytest.mark.parametrize("idx", [-1, 3])
    def test_replace_token_out_of_bounds_raises(self, idx):
        with pytest.raises(IndexError, match="out of bounds"):
            replace_token(["a", "b", "c"], idx)

    def test_mask_positions_ignores_out_of_range(self):
        toks = ["a", "b", "c", "d"]
        out = mask_positions(toks, [0, 2, 7, -1], replacement="_")
        assert out == ["_", "b", "_", "d"]
        assert toks == ["a", "b", "c", "d"]

    def test_token_join_roundtrip(self):
        assert token_join(whitespace_tokens("hello brave new world")) == "hello brave new world"
