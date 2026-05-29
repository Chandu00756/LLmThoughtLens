"""Tests for the TopK SparseAutoencoder."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from LLmThoughtLens.features.sae import SAEConfig, SparseAutoencoder


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
