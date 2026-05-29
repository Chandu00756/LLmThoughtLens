"""TopK Sparse Autoencoder — real PyTorch implementation.

Architecture (Anthropic CLT-style):

    Encoder:  h    = ReLU(W_enc (x − b_dec) + b_enc)
              z    = TopK(h, k)                    # exact structural sparsity
    Decoder:  x̂   = W_dec z + b_dec
    Loss:     L   = ||x − x̂||² + λ ||z||₁

Extra production details from the literature:
* Decoder columns are renormalised to unit L2 after each optimiser step
  (otherwise the L1 sparsity penalty is trivially defeated by scaling
  features down and absorbing the magnitude in the decoder).
* Dead-feature resampling: features that haven't fired for
  ``dead_window`` steps are re-initialised toward residual examples from
  the training batch.
* Real diagnostics: reconstruction MSE, fraction of variance explained,
  mean L0 sparsity, dead-feature percentage, density histogram.

The class accepts NumPy inputs externally and converts to torch tensors
internally so that callers never need a torch dependency at use sites
outside of training.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SAEConfig:
    """Hyperparameters for the TopK Sparse Autoencoder."""

    input_dim: int = 768
    dict_size: int = 3072
    k: int = 64
    lr: float = 2e-4
    batch_size: int = 2048
    n_steps: int = 5_000
    l1_coeff: float = 8e-4
    seed: int = 0
    dead_window: int = 1_000
    log_every: int = 200
    device: str = "auto"


class SparseAutoencoder:
    """TopK Sparse Autoencoder with a real PyTorch training loop.

    Parameters
    ----------
    config:
        :class:`SAEConfig` instance.  Use the keyword form to override
        only the hyperparameters you care about::

            sae = SparseAutoencoder(SAEConfig(input_dim=64, dict_size=256, k=16))
    """

    def __init__(self, config: SAEConfig | None = None) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover — gated via extras
            raise ImportError(
                "SparseAutoencoder needs torch. "
                "Install with: pip install 'LLmThoughtLens[huggingface]'"
            ) from exc

        self.config = config or SAEConfig()
        self._labels: dict[int, str] = {}
        self._trained: bool = False
        self._steps_run: int = 0
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        import torch

        cfg = self.config
        gen = torch.Generator().manual_seed(cfg.seed)
        scale = 1.0 / math.sqrt(cfg.input_dim)

        self.W_enc = (
            torch.randn((cfg.dict_size, cfg.input_dim), generator=gen, dtype=torch.float32) * scale
        )
        self.b_enc = torch.zeros(cfg.dict_size, dtype=torch.float32)
        # Initialise decoder as transpose of encoder; renormalise columns.
        self.W_dec = self.W_enc.T.clone().contiguous()
        self._renormalise_decoder()
        self.b_dec = torch.zeros(cfg.input_dim, dtype=torch.float32)

        # Track how many steps since each feature last fired (for resampling).
        self._steps_since_active = torch.zeros(cfg.dict_size, dtype=torch.int64)

    def _renormalise_decoder(self) -> None:
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-9)
        self.W_dec.div_(norms)

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> Any:
        import torch

        req = self.config.device
        if req != "auto":
            return torch.device(req)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _to(self, device: Any) -> None:
        self.W_enc = self.W_enc.to(device)
        self.b_enc = self.b_enc.to(device)
        self.W_dec = self.W_dec.to(device)
        self.b_dec = self.b_dec.to(device)

    # ------------------------------------------------------------------
    # Encode / decode (numpy in/out for downstream extractor)
    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode activations into sparse codes.

        Parameters
        ----------
        x:
            ``(input_dim,)`` or ``(batch, input_dim)``.

        Returns
        -------
        np.ndarray
            ``(dict_size,)`` or ``(batch, dict_size)``.
        """
        import torch

        was_1d = x.ndim == 1
        xt = torch.as_tensor(np.atleast_2d(x), dtype=torch.float32, device=self.W_enc.device)
        with torch.no_grad():
            z = self._encode_torch(xt)
        out = z.cpu().numpy()
        return out[0] if was_1d else out

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct activations from sparse codes."""
        import torch

        was_1d = z.ndim == 1
        zt = torch.as_tensor(np.atleast_2d(z), dtype=torch.float32, device=self.W_enc.device)
        with torch.no_grad():
            xh = zt @ self.W_dec.T + self.b_dec
        out = xh.cpu().numpy()
        return out[0] if was_1d else out

    def encode_torch(self, x: Any) -> Any:
        """Torch-native encode for use inside autograd / hooks."""
        return self._encode_torch(x)

    def decode_torch(self, z: Any) -> Any:
        """Torch-native decode."""
        return z @ self.W_dec.T + self.b_dec

    # ------------------------------------------------------------------
    # Internal forward (torch tensors)
    # ------------------------------------------------------------------

    def _encode_torch(self, x: Any) -> Any:
        import torch

        h = torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)
        return _topk_tensor(h, self.config.k)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        activations: np.ndarray,
        verbose: bool = False,
    ) -> SparseAutoencoder:
        """Train the SAE on cached activations using real PyTorch autograd.

        Parameters
        ----------
        activations:
            ``(N, input_dim)`` — token activations collected from a corpus.
        verbose:
            Print loss + sparsity stats every ``config.log_every`` steps.
        """
        import torch

        cfg = self.config
        device = self._resolve_device()
        self._to(device)

        x_all = torch.as_tensor(activations, dtype=torch.float32, device=device)
        n = x_all.shape[0]
        if n == 0:
            raise ValueError("activations array is empty")
        if x_all.shape[1] != cfg.input_dim:
            raise ValueError(
                f"activations have d={x_all.shape[1]}, expected input_dim={cfg.input_dim}"
            )

        # b_dec is initialised to the mean of the data (CLT recommendation).
        with torch.no_grad():
            self.b_dec.copy_(x_all.mean(dim=0))

        # Optimiser holds W_enc, b_enc, W_dec, b_dec as leaf tensors with grad.
        for t in (self.W_enc, self.b_enc, self.W_dec, self.b_dec):
            t.requires_grad_(True)
        opt = torch.optim.Adam(
            [self.W_enc, self.b_enc, self.W_dec, self.b_dec],
            lr=cfg.lr,
            betas=(0.9, 0.999),
        )

        steps_since_active = self._steps_since_active.to(device)

        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed + 1)

        for step in range(1, cfg.n_steps + 1):
            idx = torch.randint(0, n, (cfg.batch_size,), generator=gen)
            x = x_all[idx]

            z = self._encode_torch(x)
            x_hat = z @ self.W_dec.T + self.b_dec
            mse = (x - x_hat).pow(2).mean()
            l1 = z.abs().mean() * cfg.l1_coeff
            loss = mse + l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                self._renormalise_decoder()

                active = (z != 0).any(dim=0)
                steps_since_active = torch.where(
                    active, torch.zeros_like(steps_since_active), steps_since_active + 1
                )

                if cfg.dead_window > 0 and step % cfg.dead_window == 0:
                    self._resample_dead_features(steps_since_active, x_all, rng)
                    steps_since_active.zero_()

            if verbose and step % max(1, cfg.log_every) == 0:
                with torch.no_grad():
                    l0 = (z != 0).float().sum(dim=-1).mean().item()
                    print(
                        f"  step {step:>6}/{cfg.n_steps} "
                        f"mse={mse.item():.4f} l1={l1.item():.4f} "
                        f"l0={l0:.1f}"
                    )

            self._steps_run += 1

        # Detach parameters once training completes.
        for t in (self.W_enc, self.b_enc, self.W_dec, self.b_dec):
            t.requires_grad_(False)
        self._steps_since_active = steps_since_active.detach().cpu()
        self._trained = True
        return self

    def _resample_dead_features(
        self,
        steps_since_active: Any,
        x_all: Any,
        rng: np.random.Generator,
    ) -> None:
        import torch

        dead = (steps_since_active >= self.config.dead_window).nonzero(as_tuple=True)[0]
        if dead.numel() == 0:
            return
        # Sample residuals — examples where current reconstruction is worst.
        n_sample = min(2048, x_all.shape[0])
        idx = torch.randint(0, x_all.shape[0], (n_sample,), device=x_all.device)
        sample = x_all[idx]
        x_hat = self._encode_torch(sample) @ self.W_dec.T + self.b_dec
        residual = (sample - x_hat).detach()
        norms = residual.norm(dim=-1)
        order = torch.argsort(-norms)
        chosen = residual[order[: dead.numel()]]
        # Encoder rows pointing toward worst residuals (with unit norm).
        chosen_unit = chosen / chosen.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self.W_enc.data[dead] = chosen_unit
        self.W_dec.data[:, dead] = chosen_unit.T
        # Reset their bias so they re-enter via ReLU again.
        self.b_enc.data[dead] = 0.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def reconstruction_loss(self, x: np.ndarray) -> float:
        """Return ``MSE + λ·||z||₁`` for the given batch."""
        z = self.encode(x)
        x_hat = self.decode(z)
        mse = float(np.mean((x - x_hat) ** 2))
        l1 = float(self.config.l1_coeff * np.mean(np.abs(z)))
        return mse + l1

    def sparsity_stats(self, activations: np.ndarray) -> dict[str, float]:
        """Return L0 sparsity stats and dead-feature fraction on a batch."""
        z = self.encode(activations)
        l0 = (z != 0).sum(axis=-1).astype(float)
        feature_active = (z != 0).any(axis=0)
        dead_fraction = float(1.0 - feature_active.mean())
        mse = float(np.mean((activations - self.decode(z)) ** 2))
        explained = 1.0 - mse / float(np.var(activations) + 1e-12)
        return {
            "l0_mean": float(l0.mean()),
            "l0_std": float(l0.std()),
            "dead_fraction": dead_fraction,
            "mse": mse,
            "explained_variance": float(max(0.0, min(1.0, explained))),
        }

    def feature_density(self, activations: np.ndarray, bins: int = 20) -> dict[str, list[float]]:
        """Return density histogram of how often each feature fires across a batch."""
        z = self.encode(activations)
        firing_rate = (z != 0).mean(axis=0)
        hist, edges = np.histogram(firing_rate, bins=bins, range=(0.0, 1.0))
        return {
            "edges": edges.astype(float).tolist(),
            "counts": hist.astype(int).tolist(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save SAE weights + config + labels to a single ``.pt`` file."""
        import torch

        path = Path(path)
        payload = {
            "config": asdict(self.config),
            "W_enc": self.W_enc.detach().cpu(),
            "b_enc": self.b_enc.detach().cpu(),
            "W_dec": self.W_dec.detach().cpu(),
            "b_dec": self.b_dec.detach().cpu(),
            "labels": self._labels,
            "trained": self._trained,
            "steps_run": self._steps_run,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> SparseAutoencoder:
        """Load weights + config + labels from a ``.pt`` file."""
        import torch

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = SAEConfig(**payload["config"])
        sae = cls(cfg)
        sae.W_enc = payload["W_enc"].to(torch.float32)
        sae.b_enc = payload["b_enc"].to(torch.float32)
        sae.W_dec = payload["W_dec"].to(torch.float32)
        sae.b_dec = payload["b_dec"].to(torch.float32)
        sae._labels = dict(payload.get("labels", {}))
        sae._trained = bool(payload.get("trained", True))
        sae._steps_run = int(payload.get("steps_run", 0))
        return sae

    def save_with_labels(self, path: str | Path, labels: dict[int, str]) -> None:
        """Persist labels alongside weights (merges with existing)."""
        self._labels.update(labels)
        self.save(path)

    def export_labels(self, path: str | Path) -> None:
        """Dump just the labels dict as JSON (handy for sharing)."""
        Path(path).write_text(json.dumps(self._labels, indent=2))

    # ------------------------------------------------------------------
    # Label management
    # ------------------------------------------------------------------

    @property
    def labels(self) -> dict[int, str]:
        return dict(self._labels)

    def set_label(self, feature_id: int, label: str) -> None:
        self._labels[int(feature_id)] = str(label)

    # ------------------------------------------------------------------
    # Feature direction (used by FeatureIntervention for white-box hooks)
    # ------------------------------------------------------------------

    def feature_direction(self, feature_id: int) -> np.ndarray:
        """Return the (unit) decoder column for *feature_id*.

        Interventions clamp/amplify the activation along this direction in
        the residual stream, rather than mutating a single coordinate.
        """
        fid = int(feature_id) % self.config.dict_size
        col = self.W_dec[:, fid].detach().cpu().numpy()
        norm = float(np.linalg.norm(col))
        if norm < 1e-9:
            return col
        return col / norm

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        return (
            f"SparseAutoencoder({status}, "
            f"input_dim={self.config.input_dim}, "
            f"dict_size={self.config.dict_size}, "
            f"k={self.config.k}, steps={self._steps_run})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _topk_tensor(h: Any, k: int) -> Any:
    """Exact TopK along the last dim — keeps top *k* magnitudes, zeros the rest.

    Operates on a torch tensor without breaking autograd.
    """
    import torch

    if k >= h.shape[-1]:
        return h
    topk_vals, topk_idx = torch.topk(h, k, dim=-1)
    mask = torch.zeros_like(h)
    mask.scatter_(-1, topk_idx, 1.0)
    return h * mask
