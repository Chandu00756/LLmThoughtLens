"""ActivationCache — collect and persist transformer activations for SAE training.

Stores ``(N, d_model)`` flattened token activations from a chosen layer of
a white-box provider.  Persists to ``.npz`` (compressed) or to a memory-mapped
``.npy`` for very large corpora.  No PyTorch dependency at use sites: the
collector receives already-numpy activations from
:class:`~LLmThoughtLens.providers.huggingface_provider.HuggingFaceProvider`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from LLmThoughtLens.providers.base import BaseProvider


class ActivationCache:
    """Collect per-token residual-stream activations from a provider.

    Parameters
    ----------
    provider:
        A white-box :class:`~LLmThoughtLens.providers.base.BaseProvider`
        (typically :class:`~LLmThoughtLens.providers.huggingface_provider.HuggingFaceProvider`).
    layer:
        Index of the transformer block whose activations to keep.
    max_tokens:
        Optional hard cap on total tokens stored.  ``None`` means no cap.
    dtype:
        NumPy dtype for the stored array (``float32`` by default).
    """

    def __init__(
        self,
        provider: BaseProvider,
        layer: int = 0,
        max_tokens: int | None = None,
        dtype: Any = np.float32,
    ) -> None:
        if not provider.supports_internals:
            raise ValueError(
                f"provider {provider.name!r} does not expose internal activations; "
                "ActivationCache requires a white-box provider."
            )
        self.provider = provider
        self.layer = int(layer)
        self.max_tokens = max_tokens
        self.dtype = np.dtype(dtype)
        self._chunks: list[np.ndarray] = []
        self._tokens_seen: int = 0
        self._d_model: int | None = None

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(
        self,
        prompts: Iterable[str],
        verbose: bool = False,
    ) -> ActivationCache:
        """Run each prompt through the provider and accumulate layer activations.

        Returns ``self`` for chaining.
        """
        for i, prompt in enumerate(prompts):
            if self.max_tokens is not None and self._tokens_seen >= self.max_tokens:
                break
            out = self.provider.run(prompt)
            if out.activations is None:
                raise RuntimeError(
                    "provider returned no activations; ensure capture_internals=True"
                )
            if not 0 <= self.layer < out.n_layers:
                raise IndexError(f"layer={self.layer} out of range for n_layers={out.n_layers}")
            chunk = out.activations[self.layer].astype(self.dtype)  # (T, D)
            if self.max_tokens is not None:
                remaining = self.max_tokens - self._tokens_seen
                chunk = chunk[:remaining]
            self._chunks.append(chunk)
            self._tokens_seen += chunk.shape[0]
            if self._d_model is None:
                self._d_model = chunk.shape[1]
            if verbose:
                print(f"  prompt {i + 1}: +{chunk.shape[0]} tokens (total {self._tokens_seen})")
        return self

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def array(self) -> np.ndarray:
        """Return the accumulated ``(N, d_model)`` array."""
        if not self._chunks:
            return np.empty((0, self._d_model or 0), dtype=self.dtype)
        return np.concatenate(self._chunks, axis=0)

    def __len__(self) -> int:
        return self._tokens_seen

    @property
    def d_model(self) -> int:
        return self._d_model or 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the accumulated activations to a ``.npz`` (compressed)."""
        arr = self.array()
        meta: dict[str, Any] = {
            "layer": int(self.layer),
            "n_tokens": int(arr.shape[0]),
            "d_model": int(arr.shape[1]) if arr.size else int(self.d_model),
            "provider": self.provider.name,
            "model_id": self.provider.model_id,
        }
        np.savez_compressed(
            Path(path), activations=arr, **{f"meta__{k}": v for k, v in meta.items()}
        )

    @classmethod
    def load(cls, path: str | Path) -> dict[str, Any]:
        """Load a previously saved ``.npz`` and return ``{activations, meta}``."""
        data = np.load(Path(path), allow_pickle=False)
        out: dict[str, Any] = {"activations": data["activations"]}
        for k in data.files:
            if k.startswith("meta__"):
                out.setdefault("meta", {})[k.removeprefix("meta__")] = data[k].item()
        return out
