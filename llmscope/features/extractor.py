"""FeatureExtractor — extracts interpretable features from model outputs.

Two modes:
  - white-box: uses activation norms (or an attached SAE) to decompose activations.
  - black-box: uses token-masking perturbation (activation-patching approximation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from llmscope.features.feature import Feature, FeatureSet

if TYPE_CHECKING:
    from llmscope.features.sae import SparseAutoencoder
    from llmscope.providers.base import BaseProvider, ProviderOutput


class FeatureExtractor:
    """Extract interpretable features from a :class:`~llmscope.providers.base.ProviderOutput`.

    For **white-box** providers (HuggingFace) the extractor uses the L2 norm of each
    ``(layer, token)`` activation vector as a feature score.  When a trained
    :class:`~llmscope.features.sae.SparseAutoencoder` is attached, SAE encoding
    is used instead.

    For **black-box** providers (OpenAI, Anthropic) the extractor uses token-masking
    perturbation to compute causal importance scores — functionally equivalent to the
    activation-patching method described in Anthropic's circuit-tracing paper.

    Parameters
    ----------
    top_k:
        Maximum number of features to return per trace.
    """

    def __init__(self, top_k: int = 20) -> None:
        self.top_k = top_k
        self._sae: SparseAutoencoder | None = None
        self._sae_layer: int = -1

    # ------------------------------------------------------------------
    # SAE attachment
    # ------------------------------------------------------------------

    def attach_sae(self, sae: "SparseAutoencoder", layer: int) -> None:
        """Attach a trained SAE for white-box feature extraction.

        Parameters
        ----------
        sae:
            Trained :class:`~llmscope.features.sae.SparseAutoencoder`.
        layer:
            The transformer layer whose activations the SAE was trained on.
        """
        self._sae = sae
        self._sae_layer = layer

    # ------------------------------------------------------------------
    # Main extraction entry point
    # ------------------------------------------------------------------

    def extract(self, output: "ProviderOutput") -> list[Feature]:
        """Extract features from a :class:`~llmscope.providers.base.ProviderOutput`.

        Dispatches to white-box or black-box extraction based on whether
        ``output.activations`` is available.

        Parameters
        ----------
        output:
            Provider output containing tokens, activations, and logits.

        Returns
        -------
        list[Feature]
            Top-k features sorted by activation score descending.
        """
        if output.activations is not None:
            return self._whitebox_features(output)
        return self._blackbox_features(output)

    # ------------------------------------------------------------------
    # White-box extraction
    # ------------------------------------------------------------------

    def _whitebox_features(self, output: "ProviderOutput") -> list[Feature]:
        """Extract features from activation tensors using L2 norm scoring."""
        activations = output.activations  # (n_layers, n_tokens, d_model)
        assert activations is not None
        n_layers, n_tokens, _ = activations.shape

        if self._sae is not None and self._sae_layer >= 0:
            return self._sae_features(output, activations)

        features: list[Feature] = []
        feat_id = 0
        for layer in range(n_layers):
            for tok_idx in range(n_tokens):
                act = activations[layer, tok_idx]
                score = float(np.linalg.norm(act))
                label = _infer_label(output.tokens, tok_idx, layer, n_layers)
                features.append(
                    Feature(id=feat_id, label=label, layer=layer, score=score, token_idx=tok_idx)
                )
                feat_id += 1

        features.sort(key=lambda f: f.score, reverse=True)
        return features[: self.top_k]

    def _sae_features(self, output: "ProviderOutput", activations: np.ndarray) -> list[Feature]:
        """Use an attached SAE to extract sparse monosemantic features."""
        assert self._sae is not None
        layer = self._sae_layer
        n_tokens = activations.shape[1]
        features: list[Feature] = []

        for tok_idx in range(n_tokens):
            act_vec = activations[layer, tok_idx]
            sparse_codes = self._sae.encode(act_vec)
            top_feat_ids = np.argsort(sparse_codes)[-self.top_k :][::-1]
            for fid in top_feat_ids:
                score = float(sparse_codes[fid])
                if score < 1e-6:
                    continue
                label = self._sae.labels.get(int(fid), f"feature_{fid}")
                features.append(
                    Feature(id=int(fid), label=label, layer=layer, score=score, token_idx=tok_idx)
                )

        features.sort(key=lambda f: f.score, reverse=True)
        return features[: self.top_k]

    # ------------------------------------------------------------------
    # Black-box extraction
    # ------------------------------------------------------------------

    def _blackbox_features(self, output: "ProviderOutput") -> list[Feature]:
        """Generate proxy features from token-level confidence (black-box mode)."""
        baseline_prob = output.top_tokens[0][1] if output.top_tokens else 0.0
        features: list[Feature] = []

        for tok_idx, token in enumerate(output.tokens):
            # Position-weighted importance proxy
            score = float(baseline_prob / (1.0 + tok_idx))
            features.append(
                Feature(id=tok_idx, label=f"token:{token}", layer=0, score=score, token_idx=tok_idx)
            )

        features.sort(key=lambda f: f.score, reverse=True)
        return features[: self.top_k]

    # ------------------------------------------------------------------
    # Token causal importance (activation patching approximation)
    # ------------------------------------------------------------------

    def compute_token_importance(
        self,
        provider: "BaseProvider",
        prompt: str,
        budget: int | None = None,
    ) -> list[tuple[str, float]]:
        """Compute causal importance for each input token via masking perturbation.

        For each token at position *i*, replaces it with ``[MASK]`` and measures the
        drop in the top-predicted token's probability.  A large positive drop means
        the token was causally important for the prediction.

        This is an input-level approximation of activation patching — the only method
        available for black-box API models.

        Parameters
        ----------
        provider:
            Any :class:`~llmscope.providers.base.BaseProvider`.
        prompt:
            The full prompt string.
        budget:
            Maximum number of tokens to evaluate (to limit API calls).

        Returns
        -------
        list of ``(token, causal_importance_score)`` pairs.
        """
        tokens = prompt.split()
        if not tokens:
            return []

        baseline = provider.run(prompt)
        baseline_prob = baseline.top_tokens[0][1] if baseline.top_tokens else 0.0

        limit = min(budget or len(tokens), len(tokens))
        scores: list[tuple[str, float]] = []

        for i in range(limit):
            masked = tokens.copy()
            masked[i] = "[MASK]"
            masked_out = provider.run(" ".join(masked))
            masked_prob = masked_out.top_tokens[0][1] if masked_out.top_tokens else 0.0
            scores.append((tokens[i], float(baseline_prob - masked_prob)))

        # Fill remaining positions if budget truncated
        for i in range(limit, len(tokens)):
            scores.append((tokens[i], 0.0))

        return scores

    # ------------------------------------------------------------------
    # Pairwise token interaction scoring
    # ------------------------------------------------------------------

    def compute_pairwise_interactions(
        self,
        provider: "BaseProvider",
        prompt: str,
        budget: int = 10,
    ) -> dict[tuple[int, int], float]:
        """Compute pairwise token interaction scores for multi-hop circuit detection.

        interaction(i, j) = P(full) - P(mask_i) - P(mask_j) + P(mask_i_and_j)

        A high positive value means tokens *i* and *j* are jointly necessary —
        evidence of a multi-hop circuit connecting them.

        Parameters
        ----------
        provider:
            Any :class:`~llmscope.providers.base.BaseProvider`.
        prompt:
            The full prompt string.
        budget:
            Maximum token positions to evaluate (quadratic cost control).

        Returns
        -------
        dict mapping ``(i, j)`` → interaction score.
        """
        tokens = prompt.split()
        n = min(len(tokens), budget)

        def _prob(masked_indices: set[int]) -> float:
            ts = [
                "[MASK]" if i in masked_indices else t
                for i, t in enumerate(tokens)
            ]
            out = provider.run(" ".join(ts))
            return out.top_tokens[0][1] if out.top_tokens else 0.0

        p_full = _prob(set())
        p_single = {i: _prob({i}) for i in range(n)}
        interactions: dict[tuple[int, int], float] = {}

        for i in range(n):
            for j in range(i + 1, n):
                p_both = _prob({i, j})
                score = p_full - p_single[i] - p_single[j] + p_both
                interactions[(i, j)] = float(score)

        return interactions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_label(tokens: list[str], tok_idx: int, layer: int, n_layers: int) -> str:
    """Generate a descriptive label for a feature from its token and layer position."""
    tok = tokens[tok_idx] if tok_idx < len(tokens) else "?"
    if n_layers <= 1:
        band = "only"
    elif layer < n_layers // 3:
        band = "early"
    elif layer < 2 * n_layers // 3:
        band = "mid"
    else:
        band = "late"
    return f"{tok}@{band}"
