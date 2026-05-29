"""Tests for FeatureIntervention — NumPy + torch paths."""

from __future__ import annotations

import numpy as np
from LLmThoughtLens.features.intervention import FeatureIntervention
from LLmThoughtLens.features.sae import SAEConfig, SparseAutoencoder


class TestNumpyPath:
    def test_clamp_zeros_dim_no_sae(self):
        acts = np.ones((2, 3, 4), dtype=np.float32)
        inter = FeatureIntervention.clamp(feature_id=2, value=0.0, layer=1, token_idx=2)
        out = inter.apply_numpy(acts)
        # Coordinate index 2 (% d_model=4 → 2) at (1,2,:) becomes 0.
        assert out[1, 2, 2] == 0.0
        # Untouched cells stay 1.
        assert out[0, 0, 0] == 1.0

    def test_inhibit_reduces_projection(self):
        rng = np.random.default_rng(0)
        acts = rng.standard_normal((2, 3, 4)).astype(np.float32)
        inter = FeatureIntervention.inhibit(feature_id=1, scale=1.0, layer=0, token_idx=1)
        out = inter.apply_numpy(acts)
        before = float(acts[0, 1, 1])
        after = float(out[0, 1, 1])
        assert abs(after) <= abs(before) + 1e-6

    def test_amplify_grows_projection(self):
        acts = np.ones((1, 1, 4), dtype=np.float32)
        inter = FeatureIntervention.amplify(feature_id=0, scale=3.0, layer=0, token_idx=0)
        out = inter.apply_numpy(acts)
        assert out[0, 0, 0] > acts[0, 0, 0]

    def test_apply_alias(self):
        acts = np.ones((1, 1, 4), dtype=np.float32)
        inter = FeatureIntervention.clamp(0, 0.0)
        assert np.allclose(inter.apply(acts), inter.apply_numpy(acts))


class TestSAEPath:
    def test_uses_decoder_direction(self):
        sae = SparseAutoencoder(
            SAEConfig(input_dim=8, dict_size=16, k=2, n_steps=30, batch_size=16)
        )
        # Quick fit so the decoder columns settle to unit norm.
        rng = np.random.default_rng(1)
        X = rng.standard_normal((128, 8)).astype(np.float32)
        sae.fit(X, verbose=False)
        direction = sae.feature_direction(3)

        acts = rng.standard_normal((1, 1, 8)).astype(np.float32)
        before_proj = float(np.dot(acts[0, 0], direction))
        inter = FeatureIntervention.clamp(feature_id=3, value=0.0, layer=0, token_idx=0, sae=sae)
        out = inter.apply_numpy(acts)
        after_proj = float(np.dot(out[0, 0], direction))
        assert abs(after_proj) < abs(before_proj) + 1e-5
        assert abs(after_proj) < 1e-5


class TestTorchPath:
    def test_apply_torch_zeros_projection(self):
        import torch

        d_model = 8
        hidden = torch.randn(1, 4, d_model)
        inter = FeatureIntervention.clamp(feature_id=2, value=0.0, layer=0)
        out = inter.apply_torch(hidden)
        assert out.shape == hidden.shape
        # The same coord must be ~0 after clamp.
        assert abs(float(out[0, 0, 2 % d_model])) < 1e-6
