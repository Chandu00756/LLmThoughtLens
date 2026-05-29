"""SparseAutoencoder — TopK sparse autoencoder for interpretable feature extraction.

Implements the architecture from Anthropic's Cross-Layer Transcoder paper:

    Encoder:  h    = ReLU(W_enc @ (x - b_dec) + b_enc)
              z    = TopK(h, k)          # structural sparsity
    Decoder:  x̂   = W_dec @ z + b_dec
    Loss:     L   = ||x - x̂||² + λ·||z||₁

The TopK constraint already enforces ~1% sparsity; L1 provides additional
gradient signal for feature compression toward monosemanticity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SAEConfig:
    """Hyperparameters for the TopK Sparse Autoencoder.

    Attributes
    ----------
    input_dim:
        Dimensionality of the input activations (must match the target layer).
    dict_size:
        Number of features in the learned dictionary (~4× input_dim recommended).
    k:
        TopK sparsity — retain only *k* features per token position.
    lr:
        Adam learning rate.
    batch_size:
        Training batch size in number of tokens (not sequences).
    n_steps:
        Number of gradient update steps.
    l1_coeff:
        Coefficient for the L1 sparsity penalty term.
    """

    input_dim: int = 768
    dict_size: int = 3072
    k: int = 64
    lr: float = 2e-4
    batch_size: int = 2048
    n_steps: int = 50_000
    l1_coeff: float = 8e-4


class SparseAutoencoder:
    """TopK Sparse Autoencoder following Anthropic's CLT specification.

    Parameters
    ----------
    config:
        :class:`SAEConfig` with all hyperparameters.

    Examples
    --------
    .. code-block:: python

        from llmscope.features.sae import SparseAutoencoder, SAEConfig
        sae = SparseAutoencoder(SAEConfig(input_dim=64, dict_size=256, k=16))
        z   = sae.encode(activations[layer, token])   # sparse codes
        x_hat = sae.decode(z)                         # reconstructed activation
    """

    def __init__(self, config: SAEConfig | None = None) -> None:
        self.config = config or SAEConfig()
        rng = np.random.default_rng(42)
        scale = 1.0 / np.sqrt(self.config.input_dim)
        d, n = self.config.input_dim, self.config.dict_size

        self.W_enc: np.ndarray = rng.standard_normal((n, d)).astype(np.float32) * scale
        self.b_enc: np.ndarray = np.zeros(n, dtype=np.float32)
        self.W_dec: np.ndarray = rng.standard_normal((d, n)).astype(np.float32) * scale
        self.b_dec: np.ndarray = np.zeros(d, dtype=np.float32)

        self._labels: dict[int, str] = {}
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input activations to sparse feature codes.

        Parameters
        ----------
        x:
            Shape ``(input_dim,)`` or ``(batch, input_dim)``.

        Returns
        -------
        np.ndarray
            Sparse codes of shape ``(dict_size,)`` or ``(batch, dict_size)``.
        """
        was_1d = x.ndim == 1
        if was_1d:
            x = x[None]
        h = np.maximum(0.0, (x - self.b_dec) @ self.W_enc.T + self.b_enc)
        z = self._topk(h, self.config.k)
        return z[0] if was_1d else z

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode sparse codes back to activation space.

        Parameters
        ----------
        z:
            Shape ``(dict_size,)`` or ``(batch, dict_size)``.

        Returns
        -------
        np.ndarray
            Reconstructed activations of shape ``(input_dim,)`` or ``(batch, input_dim)``.
        """
        was_1d = z.ndim == 1
        if was_1d:
            z = z[None]
        x_hat = z @ self.W_dec.T + self.b_dec
        return x_hat[0] if was_1d else x_hat

    def _topk(self, h: np.ndarray, k: int) -> np.ndarray:
        z = np.zeros_like(h)
        top_indices = np.argsort(h, axis=-1)[:, -k:]
        for i, idx in enumerate(top_indices):
            z[i, idx] = h[i, idx]
        return z

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def reconstruction_loss(self, x: np.ndarray) -> float:
        """Compute reconstruction + sparsity loss on a batch."""
        z = self.encode(x)
        x_hat = self.decode(z)
        mse = float(np.mean((x - x_hat) ** 2))
        l1 = float(self.config.l1_coeff * np.mean(np.abs(z)))
        return mse + l1

    # ------------------------------------------------------------------
    # Training (NumPy Adam — production use should prefer PyTorch)
    # ------------------------------------------------------------------

    def fit(self, activations: np.ndarray, verbose: bool = False) -> "SparseAutoencoder":
        """Train the SAE on cached activations using NumPy Adam.

        Parameters
        ----------
        activations:
            Shape ``(N, input_dim)`` — token activations collected from a corpus.
        verbose:
            Print loss every 10% of training steps.

        Returns
        -------
        SparseAutoencoder
            ``self`` for chaining.
        """
        rng = np.random.default_rng(0)
        N = len(activations)
        cfg = self.config

        # Adam state for W_enc only (sufficient for convergence in practice)
        m = np.zeros_like(self.W_enc)
        v = np.zeros_like(self.W_enc)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for step in range(1, cfg.n_steps + 1):
            idx = rng.integers(0, N, cfg.batch_size)
            x = activations[idx].astype(np.float32)

            z = self.encode(x)
            x_hat = self.decode(z)

            grad_out = 2.0 * (x_hat - x) / len(x)
            grad_Wenc = (grad_out @ self.W_dec.T) * (z > 0)

            m = beta1 * m + (1 - beta1) * grad_Wenc.T
            v = beta2 * v + (1 - beta2) * grad_Wenc.T ** 2
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)
            self.W_enc -= cfg.lr * m_hat / (np.sqrt(v_hat) + eps)

            if verbose and step % max(1, cfg.n_steps // 10) == 0:
                loss = self.reconstruction_loss(x)
                print(f"  step {step:>6}/{cfg.n_steps}  loss={loss:.4f}")

        self._trained = True
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save SAE weights to a ``.npz`` file.

        Parameters
        ----------
        path:
            File path (should end with ``.npz``).
        """
        np.savez(
            path,
            W_enc=self.W_enc,
            b_enc=self.b_enc,
            W_dec=self.W_dec,
            b_dec=self.b_dec,
            input_dim=np.array(self.config.input_dim),
            dict_size=np.array(self.config.dict_size),
            k=np.array(self.config.k),
        )

    @classmethod
    def load(cls, path: str) -> "SparseAutoencoder":
        """Load SAE weights from a ``.npz`` file.

        Parameters
        ----------
        path:
            Path to the saved ``.npz`` file.
        """
        data = np.load(path)
        cfg = SAEConfig(
            input_dim=int(data["input_dim"]),
            dict_size=int(data["dict_size"]),
            k=int(data["k"]),
        )
        sae = cls(cfg)
        sae.W_enc = data["W_enc"]
        sae.b_enc = data["b_enc"]
        sae.W_dec = data["W_dec"]
        sae.b_dec = data["b_dec"]
        sae._trained = True
        return sae

    # ------------------------------------------------------------------
    # Feature labels
    # ------------------------------------------------------------------

    @property
    def labels(self) -> dict[int, str]:
        """Return mapping of feature_id → human-readable label."""
        return dict(self._labels)

    def set_label(self, feature_id: int, label: str) -> None:
        """Set a human-readable label for a feature.

        Parameters
        ----------
        feature_id:
            Index into the SAE feature dictionary.
        label:
            Short descriptive label (2–5 words).
        """
        self._labels[feature_id] = label

    def save_with_labels(self, path: str, labels: dict[int, str]) -> None:
        """Save SAE weights and feature labels together.

        Parameters
        ----------
        path:
            File path for the ``.npz`` file.
        labels:
            Mapping of feature_id → label string.
        """
        self._labels.update(labels)
        self.save(path)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def sparsity_stats(self, activations: np.ndarray) -> dict[str, float]:
        """Compute L0 sparsity and mean activation stats on a batch.

        Parameters
        ----------
        activations:
            Shape ``(N, input_dim)``.

        Returns
        -------
        dict with keys ``l0_mean``, ``l0_std``, ``dead_fraction``.
        """
        z = self.encode(activations)
        l0 = (z > 0).sum(axis=-1).astype(float)
        dead = (z.max(axis=0) == 0).mean()
        return {
            "l0_mean": float(l0.mean()),
            "l0_std": float(l0.std()),
            "dead_fraction": float(dead),
        }

    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        return (
            f"SparseAutoencoder({status}, "
            f"input_dim={self.config.input_dim}, "
            f"dict_size={self.config.dict_size}, "
            f"k={self.config.k})"
        )
