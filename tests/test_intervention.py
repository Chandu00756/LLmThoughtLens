"""Tests for FeatureIntervention — NumPy + torch paths."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from LLmThoughtLens.features.intervention import FeatureIntervention
from LLmThoughtLens.features.sae import SAEConfig, SparseAutoencoder

# The NumPy path is core; the SAE and live-tensor paths need the torch extra.
needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch extra not installed"
)


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


@needs_torch
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


@needs_torch
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


class TestSiteField:
    def test_default_site_is_legacy_mlp_in(self):
        inter = FeatureIntervention.inhibit(0)
        assert inter.site == "mlp_in"
        assert "site" not in repr(inter)

    def test_site_validation_and_repr(self):
        inter = FeatureIntervention.clamp(1, 0.0, site="resid_post")
        assert inter.site == "resid_post"
        assert "site='resid_post'" in repr(inter)
        with pytest.raises(ValueError, match="unknown intervention site"):
            FeatureIntervention(feature_id=0, site="attn_out")  # type: ignore[arg-type]

    def test_numpy_path_ignores_site(self):
        acts = np.ones((2, 3, 4), dtype=np.float32)
        a = FeatureIntervention.clamp(2, 0.0, layer=1, token_idx=2).apply_numpy(acts)
        b = FeatureIntervention.clamp(2, 0.0, layer=1, token_idx=2, site="resid_pre")
        np.testing.assert_array_equal(a, b.apply_numpy(acts))


@needs_torch
class TestResidualSites:
    """``site`` picks the hook point inside :func:`intervention_context`."""

    @staticmethod
    def _blocks():
        import torch.nn as nn

        class _Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mlp = nn.Identity()

            def forward(self, x):  # type: ignore[override]
                return x + self.mlp(x)

        return [_Block(), _Block()]

    def _run(self, blocks, x):
        for b in blocks:
            x = b(x)
        return x

    def test_resid_post_edits_block_output(self):
        import torch
        from LLmThoughtLens.features.intervention import intervention_context

        blocks = self._blocks()
        x = torch.ones(1, 3, 4)
        spec = FeatureIntervention.clamp(1, value=7.0, layer=0, site="resid_post")
        seen = []
        h = blocks[1].register_forward_pre_hook(lambda _m, args: seen.append(args[0].clone()))
        with intervention_context(blocks, [spec]) as ctx:
            assert ctx.n_installed == 1
            assert len(blocks[0]._forward_hooks) == 1 and not blocks[0].mlp._forward_pre_hooks
            out = self._run(blocks, x)
        h.remove()
        # Block 0 output (2.0 everywhere) has coordinate 1 clamped to 7 on every token.
        assert torch.allclose(seen[0][..., 1], torch.full((1, 3), 7.0))
        assert torch.allclose(seen[0][..., 0], torch.full((1, 3), 2.0))
        assert torch.allclose(out[..., 1], torch.full((1, 3), 14.0))
        assert not blocks[0]._forward_hooks

    def test_resid_pre_edits_block_input_at_one_token(self):
        import torch
        from LLmThoughtLens.features.intervention import intervention_context

        blocks = self._blocks()
        x = torch.ones(1, 3, 4)
        spec = FeatureIntervention.clamp(0, value=0.0, layer=1, token_idx=2, site="resid_pre")
        with intervention_context(blocks, [spec]):
            assert len(blocks[1]._forward_pre_hooks) == 1
            out = self._run(blocks, x)
        # Block 1 input coordinate 0 at token 2 became 0 -> output 0 there; elsewhere 4.
        assert float(out[0, 2, 0]) == 0.0
        assert float(out[0, 1, 0]) == 4.0 and float(out[0, 2, 1]) == 4.0
        assert not blocks[1]._forward_pre_hooks

    def test_mixed_sites_and_cleanup_on_error(self):
        import torch
        from LLmThoughtLens.features.intervention import intervention_context

        blocks = self._blocks()
        specs = [
            FeatureIntervention.inhibit(0, layer=0),
            FeatureIntervention.inhibit(0, layer=0, site="resid_pre"),
            FeatureIntervention.inhibit(0, layer=1, site="resid_post"),
        ]
        with pytest.raises(RuntimeError), intervention_context(blocks, specs) as ctx:
            assert ctx.n_installed == 3
            self._run(blocks, torch.ones(1, 2, 4))
            raise RuntimeError("fail mid-forward")
        for b in blocks:
            assert not b._forward_hooks and not b._forward_pre_hooks
            assert not b.mlp._forward_pre_hooks
        with intervention_context([], specs) as ctx:
            assert ctx.n_installed == 0

    def test_tiny_gpt2_resid_post_matches_hooked_model_capture(self):
        pytest.importorskip("transformers")
        from LLmThoughtLens.features.intervention import intervention_context
        from LLmThoughtLens.models import HookedModel

        from _tiny_hf import WordTokenizer, make_tiny_gpt2

        hm = HookedModel(make_tiny_gpt2(), WordTokenizer())
        spec = FeatureIntervention.clamp(3, value=9.0, layer=0, token_idx=1, site="resid_post")
        with intervention_context(hm.blocks, [spec]):
            res = hm.forward("a b c")
        assert abs(float(res.resid_post[0, 1, 3]) - 9.0) < 1e-5
        assert abs(float(res.resid_pre[1, 1, 3]) - 9.0) < 1e-5
