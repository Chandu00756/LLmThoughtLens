"""ActivationCache — collect and persist transformer activations for SAE training.

Stores ``(N, d_model)`` flattened token activations from a chosen layer of
a white-box provider and persists them to a compressed ``.npz``.  No PyTorch
dependency at use sites: the collector receives already-numpy activations from
:class:`~LLmThoughtLens.providers.huggingface_provider.HuggingFaceProvider`.

Alongside the activations the cache records, for every stored row, the
provider's *own* token string (``ProviderOutput.tokens`` — BPE pieces for
HuggingFace models) and where it came from: the index of the prompt it
belongs to and its position inside that prompt.  Downstream consumers such as
:class:`~LLmThoughtLens.features.labeler.FeatureLabeler` rebuild token contexts
from these (:meth:`ActivationCache.token_streams`) instead of re-tokenising
the corpus, which would silently misalign with the activations.  Files
written before token storage existed still load; they simply lack those keys.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
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
        # Per stored row: provider token string, prompt index, position in prompt.
        self._tokens: list[str] = []
        self._token_prompt: list[int] = []
        self._token_pos: list[int] = []
        # Per collected prompt: its text and (optionally) its source line number.
        self._prompts: list[str] = []
        self._prompt_lines: list[int] = []
        self._tokens_aligned = True

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(
        self,
        prompts: Iterable[str],
        verbose: bool = False,
        *,
        line_numbers: Sequence[int] | None = None,
    ) -> ActivationCache:
        """Run each prompt through the provider and accumulate layer activations.

        Parameters
        ----------
        prompts:
            Prompts to run, in order.
        verbose:
            Print one progress line per prompt.
        line_numbers:
            Optional source line number of each prompt (parallel to
            *prompts*, e.g. 1-based corpus lines), persisted by :meth:`save`
            as ``prompt_lines`` so rows can be traced back to the corpus.

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
            tokens = self._provider_tokens(out.tokens, chunk.shape[0], prompt)
            if self.max_tokens is not None:
                remaining = self.max_tokens - self._tokens_seen
                chunk = chunk[:remaining]
                tokens = tokens[: chunk.shape[0]]
            prompt_idx = len(self._prompts)
            self._prompts.append(prompt)
            self._prompt_lines.append(self._line_number(line_numbers, i))
            self._tokens.extend(tokens)
            self._token_prompt.extend([prompt_idx] * len(tokens))
            self._token_pos.extend(range(len(tokens)))
            self._chunks.append(chunk)
            self._tokens_seen += chunk.shape[0]
            if self._d_model is None:
                self._d_model = chunk.shape[1]
            if verbose:
                print(f"  prompt {i + 1}: +{chunk.shape[0]} tokens (total {self._tokens_seen})")
        return self

    def _provider_tokens(self, tokens: list[str], n_rows: int, prompt: str) -> list[str]:
        """The provider's token strings for one prompt, one per activation row.

        A provider whose ``tokens`` don't match its activation rows can't be
        aligned exactly: we warn, pad/truncate with ``"<?>"`` and record the
        cache as not token-aligned (``meta["tokens_aligned"] = False``).
        """
        toks = [str(t) for t in tokens]
        if len(toks) == n_rows:
            return toks
        self._tokens_aligned = False
        warnings.warn(
            f"provider {self.provider.name!r} returned {len(toks)} tokens but {n_rows} "
            f"activation rows for prompt {prompt[:40]!r}; stored token strings are "
            "padded/truncated and contexts for this prompt may be misaligned.",
            RuntimeWarning,
            stacklevel=3,
        )
        return (toks + ["<?>"] * n_rows)[:n_rows]

    @staticmethod
    def _line_number(line_numbers: Sequence[int] | None, i: int) -> int:
        if line_numbers is None:
            return -1
        if i >= len(line_numbers):
            raise ValueError(
                f"line_numbers has {len(line_numbers)} entries but prompt #{i + 1} was given; "
                "it must be parallel to prompts"
            )
        return int(line_numbers[i])

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
        """Save activations, provider tokens and the row mapping to a ``.npz``.

        Keys: ``activations`` ``(N, D)``; ``tokens`` ``(N,)`` provider token
        strings; ``token_prompt`` / ``token_pos`` ``(N,)`` prompt index and
        position of each row; ``prompts`` ``(P,)`` the prompt texts;
        ``prompt_lines`` ``(P,)`` source line numbers (only when
        :meth:`collect` was given ``line_numbers``); plus ``meta__*`` scalars.
        """
        arr = self.array()
        meta: dict[str, Any] = {
            "layer": int(self.layer),
            "n_tokens": int(arr.shape[0]),
            "d_model": int(arr.shape[1]) if arr.size else int(self.d_model),
            "provider": self.provider.name,
            "model_id": self.provider.model_id,
            "tokens_aligned": bool(self._tokens_aligned),
        }
        arrays: dict[str, Any] = {
            "activations": arr,
            "tokens": np.asarray(self._tokens, dtype=np.str_),
            "token_prompt": np.asarray(self._token_prompt, dtype=np.int64),
            "token_pos": np.asarray(self._token_pos, dtype=np.int64),
            "prompts": np.asarray(self._prompts, dtype=np.str_),
        }
        if any(n >= 0 for n in self._prompt_lines):
            arrays["prompt_lines"] = np.asarray(self._prompt_lines, dtype=np.int64)
        arrays.update({f"meta__{k}": v for k, v in meta.items()})
        np.savez_compressed(Path(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> dict[str, Any]:
        """Load a previously saved ``.npz``.

        Always returns ``activations`` and ``meta``.  Files written with token
        storage also return ``tokens`` (``list[str]``), ``token_prompt`` /
        ``token_pos`` (int arrays), ``prompts`` (``list[str]``) and, when
        recorded, ``prompt_lines``; older files simply lack these keys (use
        :meth:`has_tokens` to check).
        """
        with np.load(Path(path), allow_pickle=False) as data:
            out: dict[str, Any] = {"activations": data["activations"], "meta": {}}
            for k in data.files:
                if k.startswith("meta__"):
                    out["meta"][k.removeprefix("meta__")] = data[k].item()
            if "tokens" in data.files:
                out["tokens"] = [str(t) for t in data["tokens"].tolist()]
                out["token_prompt"] = data["token_prompt"].astype(np.int64)
                out["token_pos"] = data["token_pos"].astype(np.int64)
                out["prompts"] = [str(t) for t in data["prompts"].tolist()]
            if "prompt_lines" in data.files:
                out["prompt_lines"] = data["prompt_lines"].astype(np.int64)
        return out

    @staticmethod
    def has_tokens(data: dict[str, Any]) -> bool:
        """True if a :meth:`load` result carries the provider's token strings."""
        return "tokens" in data

    @staticmethod
    def token_streams(data: dict[str, Any]) -> tuple[list[list[str]], list[tuple[int, int]]]:
        """Rebuild per-prompt token streams and the row mapping from :meth:`load` output.

        Returns ``(streams, positions)`` where ``streams[p]`` is the provider's
        token list for prompt ``p`` and ``positions[i] == (p, t)`` locates
        activation row ``i`` — exactly what
        :meth:`~LLmThoughtLens.features.labeler.FeatureLabeler.label_all`
        takes.

        Raises
        ------
        ValueError
            If the file predates token storage or its arrays are inconsistent.
        """
        if not ActivationCache.has_tokens(data):
            raise ValueError(
                "this activation cache has no stored tokens (written by an older "
                "LLmThoughtLens); re-run `LLmThoughtLens cache-activations` to record "
                "the provider's own tokens alongside the activations"
            )
        tokens: list[str] = list(data["tokens"])
        prompt_idx = [int(p) for p in np.asarray(data["token_prompt"]).tolist()]
        pos_idx = [int(t) for t in np.asarray(data["token_pos"]).tolist()]
        n_rows = int(np.asarray(data["activations"]).shape[0])
        if not len(tokens) == len(prompt_idx) == len(pos_idx) == n_rows:
            raise ValueError(
                f"corrupt activation cache: {n_rows} activation rows but {len(tokens)} tokens, "
                f"{len(prompt_idx)} prompt indices and {len(pos_idx)} positions"
            )
        n_prompts = max(len(data.get("prompts", [])), max(prompt_idx, default=-1) + 1)
        streams: list[list[str]] = [[] for _ in range(n_prompts)]
        for tok, p, t in zip(tokens, prompt_idx, pos_idx, strict=True):
            if p < 0 or t != len(streams[p]):
                raise ValueError(
                    f"corrupt activation cache: row for prompt {p} position {t} is out of order"
                )
            streams[p].append(tok)
        return streams, list(zip(prompt_idx, pos_idx, strict=True))
