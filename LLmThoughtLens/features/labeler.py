"""FeatureLabeler — auto-label SAE features via an LLM labelling provider.

For each feature in the SAE dictionary, we:
1. Run a corpus of contexts through the SAE encoder.
2. Pick the top-N contexts where this feature activates most.
3. Ask a labelling LLM (any :class:`BaseProvider`) to propose a short label.
4. Cache the label back into the SAE.

The labelling prompt mirrors the protocol used in the Anthropic CLT paper —
20 contexts in, a 2–5 word label out.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from LLmThoughtLens.features.sae import SparseAutoencoder
    from LLmThoughtLens.providers.base import BaseProvider

_PROMPT_TEMPLATE = (
    "You will be shown {n} short text snippets. They all strongly activate the same "
    "neural feature in a language model. Propose a SHORT label (2-5 words) describing "
    "the common concept. Reply with the label only — no quotes, no punctuation, no "
    "explanation.\n\n"
    "Snippets:\n{contexts}\n\nLabel:"
)

_FALLBACK_LABEL = "unlabelled_feature"


class FeatureLabeler:
    """Auto-label SAE features using an LLM labelling provider."""

    def __init__(
        self,
        sae: SparseAutoencoder,
        labeler_provider: BaseProvider,
        top_n_contexts: int = 20,
        context_window: int = 8,
    ) -> None:
        self.sae = sae
        self.labeler = labeler_provider
        self.top_n_contexts = int(top_n_contexts)
        self.context_window = int(context_window)

    # ------------------------------------------------------------------
    # Core: label a single feature
    # ------------------------------------------------------------------

    def label_feature(
        self,
        feature_id: int,
        activations: np.ndarray,
        token_streams: list[list[str]],
    ) -> str:
        """Label one feature given pre-collected activations + token streams.

        Parameters
        ----------
        feature_id:
            SAE feature dictionary index.
        activations:
            ``(N, d_model)`` per-token activations matching the flattened
            token_streams (in order).
        token_streams:
            One token list per prompt that was fed to build *activations*.
            Used to extract human-readable contexts around top-activating tokens.
        """
        codes = self.sae.encode(activations)  # (N, dict_size)
        feat_col = codes[:, feature_id]
        top_idx = np.argsort(-feat_col)[: self.top_n_contexts]

        positions = self._flat_to_positions(token_streams)
        contexts: list[str] = []
        for idx in top_idx:
            if feat_col[idx] <= 0:
                continue
            stream_idx, tok_idx = positions[int(idx)]
            stream = token_streams[stream_idx]
            lo = max(0, tok_idx - self.context_window)
            hi = min(len(stream), tok_idx + self.context_window + 1)
            snippet = " ".join(stream[lo:hi])
            highlighted = (
                " ".join(stream[lo:tok_idx])
                + " «"
                + stream[tok_idx]
                + "» "
                + " ".join(stream[tok_idx + 1 : hi])
            )
            contexts.append(highlighted.strip() or snippet)

        if not contexts:
            return _FALLBACK_LABEL

        prompt = _PROMPT_TEMPLATE.format(
            n=len(contexts),
            contexts="\n".join(f"- {c}" for c in contexts[: self.top_n_contexts]),
        )
        out = self.labeler.run(prompt)
        text = "".join(out.tokens) if out.tokens else out.meta.get("completion", "")
        label = _clean_label(text) or _FALLBACK_LABEL
        self.sae.set_label(feature_id, label)
        return label

    # ------------------------------------------------------------------
    # Bulk labelling
    # ------------------------------------------------------------------

    def label_all(
        self,
        activations: np.ndarray,
        token_streams: list[list[str]],
        feature_ids: Iterable[int] | None = None,
        verbose: bool = False,
    ) -> dict[int, str]:
        """Label every (or *feature_ids*) feature whose activations are non-zero.

        Returns the resulting ``{feature_id: label}`` dict.
        """
        codes = self.sae.encode(activations)
        density = (codes != 0).mean(axis=0)
        ids = (
            list(feature_ids)
            if feature_ids is not None
            else [int(i) for i in np.where(density > 0)[0]]
        )

        results: dict[int, str] = {}
        for fid in ids:
            label = self.label_feature(fid, activations, token_streams)
            results[fid] = label
            if verbose:
                print(f"  feature {fid:>5} → {label}")
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_labels(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.sae.labels, indent=2))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flat_to_positions(self, token_streams: list[list[str]]) -> list[tuple[int, int]]:
        """Map each flat index 0..N-1 back to (stream_idx, token_idx)."""
        positions: list[tuple[int, int]] = []
        for s_idx, stream in enumerate(token_streams):
            for t_idx in range(len(stream)):
                positions.append((s_idx, t_idx))
        return positions


# ---------------------------------------------------------------------------
# Label sanitiser
# ---------------------------------------------------------------------------

_CLEAN_RE = re.compile(r"[^A-Za-z0-9 _\-]+")


def _clean_label(text: str) -> str:
    text = text.strip().split("\n")[0]
    text = _CLEAN_RE.sub("", text).strip().lower()
    if not text:
        return ""
    # Keep at most 5 words and 60 chars.
    words = text.split()[:5]
    return " ".join(words)[:60]
