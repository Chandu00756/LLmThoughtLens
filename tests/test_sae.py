"""Tests for the TopK SparseAutoencoder."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from LLmThoughtLens.features.sae import SAEConfig, SparseAutoencoder

# Every test here builds a SparseAutoencoder, which needs the torch extra.
pytest.importorskip("torch")


def _synthetic_corpus(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_codes = 5
    codes = rng.standard_normal((400, n_codes)).astype(np.float32)
    codes = (np.abs(codes) * (rng.random((400, n_codes)) > 0.6)).astype(np.float32)
    D = np.linalg.qr(rng.standard_normal((16, n_codes)).astype(np.float32))[0]
    return (codes @ D.T).astype(np.float32)


class TestTopKBehaviour:
    def test_encode_respects_k(self):
        sae = SparseAutoencoder(
            SAEConfig(input_dim=16, dict_size=32, k=4, n_steps=10, batch_size=8)
        )
        x = np.random.RandomState(0).randn(7, 16).astype(np.float32)
        z = sae.encode(x)
        nnz = (z != 0).sum(axis=-1)
        assert (nnz <= 4).all(), f"L0 violates TopK cap: {nnz}"

    def test_decode_shape(self):
        sae = SparseAutoencoder(SAEConfig(input_dim=16, dict_size=24, k=3, n_steps=5))
        z = np.zeros((4, 24), dtype=np.float32)
        z[:, 0] = 1.0
        x_hat = sae.decode(z)
        assert x_hat.shape == (4, 16)


class TestTraining:
    def test_loss_decreases(self):
        X = _synthetic_corpus()
        sae = SparseAutoencoder(
            SAEConfig(
                input_dim=16,
                dict_size=32,
                k=4,
                n_steps=300,
                batch_size=64,
                lr=1e-3,
                l1_coeff=1e-3,
            )
        )
        before = sae.reconstruction_loss(X)
        sae.fit(X, verbose=False)
        after = sae.reconstruction_loss(X)
        assert after < before, f"training did not improve loss: {before} -> {after}"

    def test_sparsity_stats(self):
        X = _synthetic_corpus()
        sae = SparseAutoencoder(
            SAEConfig(input_dim=16, dict_size=32, k=4, n_steps=200, batch_size=64)
        )
        sae.fit(X, verbose=False)
        stats = sae.sparsity_stats(X)
        assert stats["l0_mean"] <= 4.5
        assert 0.0 <= stats["dead_fraction"] <= 1.0
        assert 0.0 <= stats["explained_variance"] <= 1.0

    def test_decoder_unit_norm_after_fit(self):
        X = _synthetic_corpus()
        sae = SparseAutoencoder(
            SAEConfig(input_dim=16, dict_size=24, k=3, n_steps=100, batch_size=32)
        )
        sae.fit(X, verbose=False)
        norms = sae.W_dec.norm(dim=0).cpu().numpy()
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-3)


class TestPersistence:
    def test_save_and_load_roundtrip(self):
        X = _synthetic_corpus()
        sae = SparseAutoencoder(
            SAEConfig(input_dim=16, dict_size=24, k=3, n_steps=50, batch_size=32)
        )
        sae.fit(X, verbose=False)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sae.pt"
            sae.save(path)
            loaded = SparseAutoencoder.load(path)
            z1 = sae.encode(X[:5])
            z2 = loaded.encode(X[:5])
            np.testing.assert_allclose(z1, z2, atol=1e-6)

    def test_label_save_load(self):
        sae = SparseAutoencoder(SAEConfig(input_dim=4, dict_size=8, k=2, n_steps=5))
        sae.set_label(2, "my feature")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sae.pt"
            sae.save(path)
            loaded = SparseAutoencoder.load(path)
            assert loaded.labels[2] == "my feature"


class TestDirections:
    def test_feature_direction_is_unit_norm(self):
        sae = SparseAutoencoder(SAEConfig(input_dim=16, dict_size=24, k=3, n_steps=20))
        v = sae.feature_direction(7)
        assert v.shape == (16,)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Pretrained architectures: hand-written NumPy references (published orientation)
# ---------------------------------------------------------------------------

D_IN, D_SAE = 12, 40


def _published_weights(seed: int = 0) -> dict[str, np.ndarray]:
    """SAELens / Gemma Scope orientation: W_enc (d_in, d_sae), W_dec (d_sae, d_in)."""
    rng = np.random.default_rng(seed)
    return {
        "W_enc": (rng.standard_normal((D_IN, D_SAE)) / np.sqrt(D_IN)).astype(np.float32),
        "W_dec": (rng.standard_normal((D_SAE, D_IN)) / np.sqrt(D_SAE)).astype(np.float32),
        "b_enc": (0.1 * rng.standard_normal(D_SAE)).astype(np.float32),
        "b_dec": (0.3 * rng.standard_normal(D_IN)).astype(np.float32),
        "threshold": rng.uniform(0.05, 0.6, D_SAE).astype(np.float32),
    }


def _ref_preprocess(
    x: np.ndarray,
    b_dec: np.ndarray,
    *,
    apply_b_dec: bool = True,
    norm: str = "none",
    factor: float | None = None,
    center: bool = False,
) -> np.ndarray:
    x = x.astype(np.float64)
    if center:
        x = x - x.mean(-1, keepdims=True)
    if norm == "expected_average_only_in":
        assert factor is not None
        x = x * factor
    elif norm == "constant_norm_rescale":
        x = x * np.sqrt(x.shape[-1]) / np.linalg.norm(x, axis=-1, keepdims=True)
    elif norm == "layer_norm":
        xc = x - x.mean(-1, keepdims=True)
        x = xc / (xc.std(-1, ddof=1, keepdims=True) + 1e-5)
    if apply_b_dec:
        x = x - b_dec
    return x


def _ref_encode(
    x: np.ndarray, w: dict[str, np.ndarray], arch: str, k: int = 5, **pre
) -> np.ndarray:
    """Reference encoders written from the SAELens / Gemma Scope definitions."""
    pre_acts = _ref_preprocess(x, w["b_dec"], **pre) @ w["W_enc"] + w["b_enc"]
    if arch == "relu":
        return np.maximum(pre_acts, 0.0)
    if arch == "jumprelu":  # Gemma Scope: mask = pre > threshold; acts = mask * relu(pre)
        return (pre_acts > w["threshold"]) * np.maximum(pre_acts, 0.0)
    assert arch == "topk"  # SAELens: topk over pre-activations, then ReLU on the kept values
    out = np.zeros_like(pre_acts)
    idx = np.argsort(-pre_acts, axis=-1)[..., :k]
    np.put_along_axis(out, idx, np.maximum(np.take_along_axis(pre_acts, idx, -1), 0.0), -1)
    return out


def _pretrained(arch: str, w: dict[str, np.ndarray] | None = None, **kw) -> SparseAutoencoder:
    w = w or _published_weights()
    thr = w["threshold"] if arch == "jumprelu" else None
    return SparseAutoencoder.from_weights(
        w["W_enc"], w["W_dec"], w["b_enc"], w["b_dec"], thr, architecture=arch, **kw
    )


def _identity_weights(d: int, b_dec: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """An SAE that reconstructs its (pre-processed) input exactly: z = [relu(x), relu(-x)]."""
    eye = np.eye(d, dtype=np.float32)
    return {
        "W_enc": np.concatenate([eye, -eye], axis=1),
        "W_dec": np.concatenate([eye, -eye], axis=0),
        "b_enc": np.zeros(2 * d, dtype=np.float32),
        "b_dec": np.zeros(d, dtype=np.float32) if b_dec is None else b_dec.astype(np.float32),
        "threshold": np.zeros(2 * d, dtype=np.float32),
    }


def _x(n: int = 9, seed: int = 1, offset: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, D_IN)) + offset).astype(np.float32)


class TestArchitecturesMatchReference:
    @pytest.mark.parametrize("arch", ["relu", "jumprelu", "topk"])
    @pytest.mark.parametrize("apply_b_dec", [True, False])
    def test_encode_matches_reference(self, arch, apply_b_dec):
        w = _published_weights()
        sae = _pretrained(arch, w, k=5, apply_b_dec_to_input=apply_b_dec)
        x = _x()
        ref = _ref_encode(x, w, arch, k=5, apply_b_dec=apply_b_dec)
        np.testing.assert_allclose(sae.encode(x), ref, rtol=1e-5, atol=1e-5)
        assert (ref > 0).any()

    def test_jumprelu_threshold_is_not_relu(self):
        """Some features sit in (0, threshold): ReLU keeps them, JumpReLU must not."""
        w = _published_weights()
        x = _x(64)
        relu = _ref_encode(x, w, "relu")
        jump = _pretrained("jumprelu", w).encode(x)
        in_gap = (relu > 0) & (relu <= w["threshold"])
        assert in_gap.any(), "fixture must exercise the jump"
        assert (jump[in_gap] == 0).all()
        np.testing.assert_allclose(jump[~in_gap], relu[~in_gap], rtol=1e-5, atol=1e-5)

    def test_topk_caps_l0_and_keeps_largest(self):
        w = _published_weights()
        z = _pretrained("topk", w, k=3).encode(_x(20))
        assert ((z > 0).sum(-1) <= 3).all()

    def test_relu_decoder_is_published_transpose(self):
        w = _published_weights()
        sae = _pretrained("relu", w)
        np.testing.assert_array_equal(sae.W_enc.numpy(), w["W_enc"].T)
        np.testing.assert_array_equal(sae.W_dec.numpy(), w["W_dec"].T)
        z = np.zeros((1, D_SAE), dtype=np.float32)
        z[0, 7] = 2.0
        np.testing.assert_allclose(sae.decode(z)[0], 2.0 * w["W_dec"][7] + w["b_dec"], rtol=1e-6)
        dir7 = w["W_dec"][7] / np.linalg.norm(w["W_dec"][7])
        np.testing.assert_allclose(sae.feature_direction(7), dir7, rtol=1e-5)
        np.testing.assert_allclose(sae.decoder_norms(), np.linalg.norm(w["W_dec"], axis=1), 1e-5)
        # feature ids taken modulo d_sae (Feature.id % d_sae is the dictionary index)
        np.testing.assert_allclose(sae.feature_direction(7 + 3 * D_SAE), dir7, rtol=1e-5)

    def test_vector_and_batched_shapes(self):
        sae = _pretrained("relu")
        x = _x(6)
        assert sae.encode(x[0]).shape == (D_SAE,)
        codes, scale = sae.encode_with_output_scale(x.reshape(2, 3, D_IN))
        assert codes.shape == (2, 3, D_SAE) and scale.shape == (2, 3)
        np.testing.assert_allclose(codes.reshape(6, D_SAE), sae.encode(x), rtol=1e-6)
        assert sae.reconstruct(x.reshape(2, 3, D_IN)).shape == (2, 3, D_IN)
        with pytest.raises(ValueError, match="d_in"):
            sae.encode(np.zeros((2, D_IN + 1), dtype=np.float32))
        with pytest.raises(ValueError, match="d_sae"):
            sae.decode(np.zeros((2, D_SAE + 1), dtype=np.float32))


class TestNormalisation:
    @pytest.mark.parametrize(
        ("norm", "factor", "center"),
        [
            ("expected_average_only_in", 0.37, False),
            ("constant_norm_rescale", None, False),
            ("layer_norm", None, False),
            ("none", None, True),
            ("expected_average_only_in", 2.5, True),
        ],
    )
    def test_encode_matches_reference(self, norm, factor, center):
        w = _published_weights()
        sae = _pretrained(
            "relu", w, normalize_activations=norm, norm_scaling_factor=factor, center_input=center
        )
        x = _x(offset=3.0)
        ref = _ref_encode(x, w, "relu", norm=norm, factor=factor, center=center)
        np.testing.assert_allclose(sae.encode(x), ref, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize(
        ("norm", "factor", "center", "tol"),
        [
            ("none", None, False, 1e-5),
            ("expected_average_only_in", 0.37, False, 1e-5),
            ("constant_norm_rescale", None, False, 1e-5),
            ("layer_norm", None, False, 1e-4),
            ("none", None, True, 1e-5),
            ("constant_norm_rescale", None, True, 1e-5),
        ],
    )
    def test_reconstruct_undoes_preprocessing(self, norm, factor, center, tol):
        """An identity SAE reproduces its input in SAE space, so reconstruct() must be exact."""
        b_dec = np.linspace(-1, 1, D_IN).astype(np.float32)
        w = _identity_weights(D_IN, b_dec)
        sae = _pretrained(
            "relu", w, normalize_activations=norm, norm_scaling_factor=factor, center_input=center
        )
        x = _x(offset=2.0)
        np.testing.assert_allclose(sae.reconstruct(x), x, rtol=tol, atol=tol)

    @pytest.mark.parametrize(
        ("norm", "factor", "center"),
        [
            ("none", None, False),
            ("expected_average_only_in", 0.37, False),
            ("constant_norm_rescale", None, True),
            ("layer_norm", None, False),
        ],
    )
    def test_output_scale_maps_codes_to_model_units(self, norm, factor, center):
        """reconstruct(x) - offset == s * (z W_dec + b_dec) with s = output_scale_torch(x)."""
        import torch

        w = _published_weights()
        sae = _pretrained(
            "relu", w, normalize_activations=norm, norm_scaling_factor=factor, center_input=center
        )
        x = torch.as_tensor(_x(offset=1.5))
        z = sae.encode_torch(x)
        s = sae.output_scale_torch(x)
        assert tuple(s.shape) == (x.shape[0], 1)
        offset = x.mean(-1, keepdim=True) if (center or norm == "layer_norm") else 0.0
        recon = sae.reconstruct_torch(x)
        expected = s * (z @ sae.W_dec.T + sae.b_dec) + offset
        torch.testing.assert_close(recon, expected, rtol=1e-4, atol=1e-5)
        _, scale_np = sae.encode_with_output_scale(x.numpy())
        np.testing.assert_allclose(scale_np, s[:, 0].numpy(), rtol=1e-5)

    def test_expected_average_decode_divides_by_factor(self):
        w = _published_weights()
        sae = _pretrained(
            "relu", w, normalize_activations="expected_average_only_in", norm_scaling_factor=4.0
        )
        z = np.zeros((1, D_SAE), dtype=np.float32)
        z[0, 2] = 1.0
        np.testing.assert_allclose(sae.decode(z)[0], (w["W_dec"][2] + w["b_dec"]) / 4.0, 1e-6)

    def test_expected_average_requires_factor(self):
        with pytest.raises(ValueError, match="norm_scaling_factor"):
            _pretrained("relu", normalize_activations="expected_average_only_in")
        with pytest.raises(ValueError, match="> 0"):
            _pretrained(
                "relu", normalize_activations="expected_average_only_in", norm_scaling_factor=0.0
            )

    def test_unsupported_normalisation_is_rejected(self):
        with pytest.raises(ValueError, match="normalize_activations"):
            _pretrained("relu", normalize_activations="per_token_magic")

    def test_estimate_norm_scaling_factor(self):
        from LLmThoughtLens.features.sae import estimate_norm_scaling_factor

        x = _x(50, offset=1.0)
        factor = estimate_norm_scaling_factor(x)
        assert factor == pytest.approx(np.sqrt(D_IN) / np.linalg.norm(x, axis=-1).mean(), 1e-6)
        scaled = x * factor
        assert np.linalg.norm(scaled, axis=-1).mean() == pytest.approx(np.sqrt(D_IN), rel=1e-5)
        with pytest.raises(ValueError):
            estimate_norm_scaling_factor(np.zeros((0, 3)))
        with pytest.raises(ValueError):
            estimate_norm_scaling_factor(np.zeros((4, 3)))


class TestDifferentiableEncode:
    def test_encode_torch_matches_numpy_and_backprops(self):
        import torch

        w = _published_weights()
        sae = _pretrained("relu", w)
        x = torch.as_tensor(_x(5)).requires_grad_(True)
        z = sae.encode_torch(x)
        np.testing.assert_allclose(z.detach().numpy(), sae.encode(x.detach().numpy()), 1e-6)
        z.sum().backward()
        active = (z.detach() > 0).to(torch.float32)  # (N, d_sae)
        expected = active @ torch.as_tensor(w["W_enc"]).T  # sum of active encoder columns
        torch.testing.assert_close(x.grad, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("arch", ["jumprelu", "topk"])
    def test_gradient_flows_only_through_active_features(self, arch):
        import torch

        sae = _pretrained(arch, k=4)
        x = torch.as_tensor(_x(3)).requires_grad_(True)
        z = sae.encode_torch(x)
        live = int((z[0] > 0).nonzero()[0, 0])
        (dead_idx,) = (z[0] == 0).nonzero(as_tuple=True)
        z[0, live].backward(retain_graph=True)
        torch.testing.assert_close(x.grad[0], sae.W_enc[live], rtol=1e-6, atol=1e-7)
        assert torch.count_nonzero(x.grad[1:]) == 0
        x.grad = None
        z[0, int(dead_idx[0])].backward()
        assert x.grad is None or torch.count_nonzero(x.grad) == 0

    def test_reconstruct_torch_is_differentiable(self):
        import torch

        sae = _pretrained("relu", normalize_activations="layer_norm", center_input=True)
        x = torch.as_tensor(_x(4)).requires_grad_(True)
        sae.reconstruct_torch(x).pow(2).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestFromWeightsValidation:
    def test_transposed_encoder_is_rejected(self):
        w = _published_weights()
        with pytest.raises(ValueError, match="W_dec must be"):
            SparseAutoencoder.from_weights(w["W_enc"].T, w["W_dec"], architecture="relu")
        with pytest.raises(ValueError, match="W_dec must be"):
            SparseAutoencoder.from_weights(w["W_enc"], w["W_dec"].T, architecture="relu")
        with pytest.raises(ValueError, match="2-D"):
            SparseAutoencoder.from_weights(w["W_enc"][0], w["W_dec"], architecture="relu")

    def test_bias_and_threshold_shapes(self):
        w = _published_weights()
        with pytest.raises(ValueError, match="b_enc"):
            SparseAutoencoder.from_weights(w["W_enc"], w["W_dec"], w["b_dec"], architecture="relu")
        with pytest.raises(ValueError, match="b_dec"):
            SparseAutoencoder.from_weights(
                w["W_enc"], w["W_dec"], w["b_enc"], w["b_enc"], architecture="relu"
            )
        with pytest.raises(ValueError, match="needs a"):
            SparseAutoencoder.from_weights(w["W_enc"], w["W_dec"], architecture="jumprelu")
        with pytest.raises(ValueError, match="threshold must be"):
            SparseAutoencoder.from_weights(
                w["W_enc"], w["W_dec"], threshold=w["b_dec"], architecture="jumprelu"
            )
        with pytest.raises(ValueError, match="JumpReLU only"):
            SparseAutoencoder.from_weights(
                w["W_enc"], w["W_dec"], threshold=w["threshold"], architecture="relu"
            )
        with pytest.raises(TypeError, match="floating"):
            SparseAutoencoder.from_weights(
                w["W_enc"].astype(np.int64), w["W_dec"], architecture="relu"
            )

    def test_missing_biases_default_to_zero(self):
        w = _published_weights()
        sae = SparseAutoencoder.from_weights(w["W_enc"], w["W_dec"], architecture="relu")
        x = _x()
        np.testing.assert_allclose(sae.encode(x), np.maximum(x @ w["W_enc"], 0), 1e-5, 1e-6)

    def test_config_fields_are_checked(self):
        w = _published_weights()
        with pytest.raises(TypeError, match="unknown SAEConfig"):
            SparseAutoencoder.from_weights(w["W_enc"], w["W_dec"], architecure="relu")
        with pytest.raises(ValueError, match="input_dim"):
            SparseAutoencoder.from_weights(
                w["W_enc"], w["W_dec"], config=SAEConfig(input_dim=3, dict_size=D_SAE)
            )
        with pytest.raises(ValueError, match="dict_size"):
            SparseAutoencoder.from_weights(w["W_enc"], w["W_dec"], dict_size=7)
        sae = SparseAutoencoder.from_weights(
            w["W_enc"],
            w["W_dec"],
            config=SAEConfig(input_dim=D_IN, dict_size=D_SAE, seed=5),
            architecture="relu",
        )
        assert (sae.config.seed, sae.config.d_in, sae.config.d_sae) == (5, D_IN, D_SAE)

    def test_accepts_torch_tensors_of_any_float_dtype(self):
        import torch

        w = _published_weights()
        sae = SparseAutoencoder.from_weights(
            torch.as_tensor(w["W_enc"]).to(torch.bfloat16),
            torch.as_tensor(w["W_dec"]).to(torch.float16),
            architecture="relu",
        )
        assert sae.W_enc.dtype == torch.float32 and sae.W_dec.dtype == torch.float32


class TestConfigMetadata:
    def test_hook_name_round_trips(self):
        from LLmThoughtLens.features.sae import hook_name_for, parse_hook_name

        assert parse_hook_name("blocks.8.hook_resid_pre") == (8, "resid_pre")
        assert parse_hook_name(" blocks.11.hook_resid_post ") == (11, "resid_post")
        assert parse_hook_name("blocks.3.hook_mlp_out") == (3, "mlp_out")
        assert parse_hook_name("blocks.3.hook_attn_out") == (3, "attn_out")
        for other in ("blocks.3.attn.hook_z", "blocks.2.hook_resid_mid", "hook_embed", ""):
            assert parse_hook_name(other) is None
        assert hook_name_for(4, "attn_out") == "blocks.4.hook_attn_out"
        with pytest.raises(ValueError):
            hook_name_for(4, "resid_mid")
        with pytest.raises(ValueError):
            hook_name_for(-1, "resid_pre")

    def test_hook_fields_are_derived_and_checked(self):
        cfg = SAEConfig(hook_name="blocks.5.hook_resid_pre")
        assert (cfg.hook_layer, cfg.hook_site) == (5, "resid_pre")
        cfg = SAEConfig(hook_layer=2, hook_site="mlp_out")
        assert cfg.hook_name == "blocks.2.hook_mlp_out"
        cfg = SAEConfig(hook_name="blocks.3.attn.hook_z")  # recorded, but not a readable site
        assert cfg.hook_site is None and cfg.hook_layer is None
        with pytest.raises(ValueError, match="disagrees"):
            SAEConfig(hook_name="blocks.5.hook_resid_pre", hook_layer=6)
        with pytest.raises(ValueError, match="hook_site"):
            SAEConfig(hook_site="resid_mid")
        with pytest.raises(ValueError, match="hook_layer"):
            SAEConfig(hook_layer=-2)

    def test_architecture_and_k_validation(self):
        with pytest.raises(ValueError, match="architecture"):
            SAEConfig(architecture="gated")
        with pytest.raises(ValueError, match="k must be"):
            SAEConfig(k=0)
        assert SAEConfig(architecture="relu", k=0).k == 0  # k only matters for TopK

    def test_from_dict_ignores_unknown_keys(self):
        cfg = SAEConfig.from_dict({"input_dim": 4, "dict_size": 8, "from_the_future": 1})
        assert (cfg.d_in, cfg.d_sae) == (4, 8)

    def test_hook_point_and_repr(self):
        sae = _pretrained(
            "jumprelu", hook_name="blocks.3.hook_resid_post", sae_id="x/y", source_format="saelens"
        )
        assert sae.hook_point == (3, "resid_post")
        text = repr(sae)
        for part in ("saelens", "arch=jumprelu", "hook=blocks.3.hook_resid_post", "sae_id=x/y"):
            assert part in text
        assert "k=" in repr(_pretrained("topk", k=3))
        legacy = SparseAutoencoder(SAEConfig(input_dim=4, dict_size=8, k=2))
        assert (
            repr(legacy) == "SparseAutoencoder(untrained, input_dim=4, dict_size=8, k=2, steps=0)"
        )
        assert legacy.hook_point is None

    def test_to_returns_self_on_device(self):
        sae = _pretrained("jumprelu")
        assert sae.to("cpu") is sae and str(sae.device) == "cpu"
        assert str(sae.threshold.device) == "cpu"


class TestPretrainedPersistence:
    def test_round_trip_keeps_metadata_and_threshold(self, tmp_path):
        w = _published_weights()
        sae = _pretrained(
            "jumprelu",
            w,
            apply_b_dec_to_input=False,
            normalize_activations="expected_average_only_in",
            norm_scaling_factor=0.8,
            center_input=True,
            hook_name="blocks.7.hook_resid_pre",
            model_name="gpt2-small",
            release="rel",
            sae_id="blocks.7.hook_resid_pre",
            source_format="saelens",
            extra={"prepend_bos": True, "context_size": 128},
        )
        sae.set_label(4, "Texas")
        path = tmp_path / "sae.pt"
        sae.save(path)
        loaded = SparseAutoencoder.load(path)
        assert loaded.config == sae.config
        np.testing.assert_array_equal(loaded.threshold.numpy(), w["threshold"])
        x = _x()
        np.testing.assert_allclose(loaded.encode(x), sae.encode(x), atol=1e-6)
        assert loaded.labels == {4: "Texas"}
        assert loaded.hook_point == (7, "resid_pre")

    def test_legacy_trainer_payload_still_loads(self, tmp_path):
        """A file written by the pre-metadata trainer: 11 config keys, no threshold."""
        import torch

        rng = np.random.default_rng(3)
        legacy_cfg = {
            "input_dim": 6,
            "dict_size": 10,
            "k": 2,
            "lr": 2e-4,
            "batch_size": 32,
            "n_steps": 5,
            "l1_coeff": 8e-4,
            "seed": 0,
            "dead_window": 1000,
            "log_every": 200,
            "device": "cpu",
        }
        W_enc = rng.standard_normal((10, 6)).astype(np.float32)  # legacy (d_sae, d_in)
        W_dec = rng.standard_normal((6, 10)).astype(np.float32)  # legacy (d_in, d_sae)
        b_enc = rng.standard_normal(10).astype(np.float32)
        b_dec = rng.standard_normal(6).astype(np.float32)
        payload = {
            "config": legacy_cfg,
            "W_enc": torch.as_tensor(W_enc),
            "b_enc": torch.as_tensor(b_enc),
            "W_dec": torch.as_tensor(W_dec),
            "b_dec": torch.as_tensor(b_dec),
            "labels": {1: "one"},
            "trained": True,
            "steps_run": 5,
        }
        path = tmp_path / "legacy.pt"
        torch.save(payload, path)
        sae = SparseAutoencoder.load(path)
        assert sae.config.architecture == "topk" and sae.hook_point is None
        assert sae.threshold is None and sae.labels == {1: "one"}
        x = rng.standard_normal((4, 6)).astype(np.float32)
        ref = _ref_encode(
            x, {"W_enc": W_enc.T, "b_enc": b_enc, "b_dec": b_dec}, "topk", k=2
        )  # legacy TopK = TopK(ReLU(W_enc (x - b_dec) + b_enc))
        np.testing.assert_allclose(sae.encode(x), ref, rtol=1e-5, atol=1e-5)
        assert repr(sae).startswith("SparseAutoencoder(trained, input_dim=6")

    def test_corrupt_files_are_rejected(self, tmp_path):
        import torch

        sae = _pretrained("jumprelu")
        path = tmp_path / "sae.pt"
        sae.save(path)
        payload = torch.load(path, weights_only=False)
        payload["threshold"] = None
        torch.save(payload, path)
        with pytest.raises(ValueError, match="threshold"):
            SparseAutoencoder.load(path)
        payload = torch.load(path, weights_only=False)
        payload["config"]["architecture"] = "relu"
        payload["W_dec"] = payload["W_dec"][:, :5]
        torch.save(payload, path)
        with pytest.raises(ValueError, match="W_dec"):
            SparseAutoencoder.load(path)


class TestTrainingGuards:
    def test_only_trainable_architectures_fit(self):
        X = _synthetic_corpus()
        with pytest.raises(ValueError, match="not supported"):
            SparseAutoencoder(
                SAEConfig(input_dim=16, dict_size=8, architecture="jumprelu", n_steps=1)
            ).fit(X)
        cfg = SAEConfig(
            input_dim=16, dict_size=8, normalize_activations="constant_norm_rescale", n_steps=1
        )
        with pytest.raises(ValueError, match="normalisation"):
            SparseAutoencoder(cfg).fit(X)

    def test_relu_architecture_trains(self):
        X = _synthetic_corpus()
        sae = SparseAutoencoder(
            SAEConfig(
                input_dim=16,
                dict_size=32,
                architecture="relu",
                n_steps=200,
                batch_size=64,
                lr=1e-3,
                device="cpu",
            )
        )
        before = sae.reconstruction_loss(X)
        sae.fit(X)
        assert sae.reconstruction_loss(X) < before

    def test_identity_sae_stats(self):
        sae = _pretrained("relu", _identity_weights(D_IN))
        stats = sae.sparsity_stats(_x(30))
        assert stats["mse"] == pytest.approx(0.0, abs=1e-10)
        assert stats["explained_variance"] == pytest.approx(1.0)
        assert stats["l0_mean"] == pytest.approx(D_IN)
