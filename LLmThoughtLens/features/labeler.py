"""FeatureLabeler — auto-label SAE features via an LLM labelling provider.

For each feature in the SAE dictionary, we:
1. Run a corpus of contexts through the SAE encoder.
2. Pick the top-N contexts where this feature activates most.
3. Ask a labelling LLM (any :class:`BaseProvider`) to propose a short label.
4. Cache the label back into the SAE.

The labelling prompt mirrors the protocol used in the Anthropic CLT paper —
20 contexts in, a 2–5 word label out.

Activation rows and token contexts must align exactly: row ``i`` of the
activations is the token at ``positions[i] == (stream, index)``.  Build the
streams from the *provider's own* tokens (see
:meth:`~LLmThoughtLens.features.cache.ActivationCache.token_streams`), not by
re-tokenising the corpus; a length mismatch raises :class:`ValueError`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from LLmThoughtLens.features.sae import SparseAutoencoder
    from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput

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
        positions: Sequence[tuple[int, int]] | None = None,
    ) -> str:
        """Label one feature given pre-collected activations + token streams.

        Parameters
        ----------
        feature_id:
            SAE feature dictionary index.
        activations:
            ``(N, d_model)`` per-token activations.
        token_streams:
            Token lists used to extract human-readable contexts around the
            top-activating tokens.
        positions:
            ``(stream_idx, token_idx)`` of each activation row.  Defaults to
            the flattened order of *token_streams* (row ``i`` = ``i``-th token).

        Raises
        ------
        ValueError
            If ``len(positions) != activations.shape[0]`` or a position falls
            outside *token_streams* — contexts would be misaligned.
        """
        resolved = self._resolve_positions(activations, token_streams, positions)
        codes = self.sae.encode(activations)  # (N, dict_size)
        return self._label_from_codes(feature_id, codes, token_streams, resolved)

    def _label_from_codes(
        self,
        feature_id: int,
        codes: np.ndarray,
        token_streams: list[list[str]],
        positions: list[tuple[int, int]],
    ) -> str:
        feat_col = codes[:, feature_id]
        top_idx = np.argsort(-feat_col)[: self.top_n_contexts]

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
        label = _clean_label(_reply_text(out)) or _FALLBACK_LABEL
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
        positions: Sequence[tuple[int, int]] | None = None,
    ) -> dict[int, str]:
        """Label every (or *feature_ids*) feature whose activations are non-zero.

        *positions* has the same meaning as in :meth:`label_feature`; the
        alignment is validated once and the SAE encodes the activations once.

        Returns the resulting ``{feature_id: label}`` dict.
        """
        resolved = self._resolve_positions(activations, token_streams, positions)
        codes = self.sae.encode(activations)
        density = (codes != 0).mean(axis=0)
        ids = (
            list(feature_ids)
            if feature_ids is not None
            else [int(i) for i in np.where(density > 0)[0]]
        )

        results: dict[int, str] = {}
        for fid in ids:
            label = self._label_from_codes(fid, codes, token_streams, resolved)
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

    def _resolve_positions(
        self,
        activations: np.ndarray,
        token_streams: list[list[str]],
        positions: Sequence[tuple[int, int]] | None,
    ) -> list[tuple[int, int]]:
        """Validate (or derive) the row -> (stream, token) mapping."""
        n_rows = int(np.shape(activations)[0])
        if positions is None:
            resolved = self._flat_to_positions(token_streams)
            if len(resolved) != n_rows:
                raise ValueError(
                    f"activations has {n_rows} rows but token_streams contain {len(resolved)} "
                    "tokens; they must align 1:1 (row i = i-th token of the flattened streams). "
                    "Build the streams from the provider's own tokens — e.g. "
                    "ActivationCache.token_streams() on a cache written by "
                    "`LLmThoughtLens cache-activations` — not by re-tokenising the corpus."
                )
            return resolved
        resolved = [(int(s_idx), int(t_idx)) for s_idx, t_idx in positions]
        if len(resolved) != n_rows:
            raise ValueError(
                f"activations has {n_rows} rows but positions has {len(resolved)} entries; "
                "every activation row needs exactly one (stream_idx, token_idx)"
            )
        for row, (s_idx, t_idx) in enumerate(resolved):
            if not (0 <= s_idx < len(token_streams) and 0 <= t_idx < len(token_streams[s_idx])):
                raise ValueError(
                    f"positions[{row}] = ({s_idx}, {t_idx}) is outside token_streams "
                    f"({len(token_streams)} streams)"
                )
        return resolved


# ---------------------------------------------------------------------------
# Reply extraction + label sanitiser
# ---------------------------------------------------------------------------


def _reply_text(out: ProviderOutput) -> str:
    """The labelling model's reply as text.

    * ``meta["completion"]`` — the verbatim reply of chat/black-box providers
      (OpenAI, Anthropic, Ollama) — wins whenever it is non-empty.
    * Otherwise black-box ``tokens`` are a *whitespace split* of the reply, so
      they are re-joined with spaces ("proper nouns", not "propernouns").
    * White-box providers (HuggingFace, mock) put the *prompt's* tokens in
      ``tokens``; echoing those back would label every feature with the
      prompt, so their reply is the model's next-token prediction.
    """
    completion = out.meta.get("completion") if isinstance(out.meta, dict) else None
    if isinstance(completion, str) and completion.strip():
        return completion
    if out.evidence_kind == "white_box":
        return out.output_token
    return " ".join(str(t) for t in out.tokens)


_CLEAN_RE = re.compile(r"[^A-Za-z0-9 _\-]+")


def _clean_label(text: str) -> str:
    text = text.strip().split("\n")[0]
    text = _CLEAN_RE.sub("", text).strip().lower()
    if not text:
        return ""
    # Keep at most 5 words and 60 chars.
    words = text.split()[:5]
    return " ".join(words)[:60]
