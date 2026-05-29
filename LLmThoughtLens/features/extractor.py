"""FeatureExtractor — real white-box (SAE/L2) and black-box (token-masking) features.

White-box mode (``output.activations is not None``):
- With an SAE attached → use SAE-encoded sparse codes as per-token features.
- Without SAE → use L2 norm of each ``(layer, token)`` activation as score.

Black-box mode (``output.activations is None``):
- Run real token-masking forward passes (``prob_baseline − prob_masked``).
- Optionally compute pairwise interactions for multi-hop detection.

Returns :class:`~LLmThoughtLens.features.feature.Feature` objects tagged
with the correct ``evidence_kind`` so the report can colour and caveat
them appropriately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from LLmThoughtLens.features.feature import Feature
from LLmThoughtLens.utils.tokenizer_utils import mask_positions, token_join, whitespace_tokens

if TYPE_CHECKING:
    from LLmThoughtLens.features.sae import SparseAutoencoder
    from LLmThoughtLens.providers.base import BaseProvider, ProviderOutput


class FeatureExtractor:
    """Extract :class:`Feature` objects from a :class:`ProviderOutput`.

    Parameters
    ----------
    top_k:
        Maximum number of features to return per trace.
    blackbox_budget:
        Cap on number of token-masking forward passes per trace
        (each mask is one API call).  ``None`` means "all tokens".
    blackbox_cache:
        When ``True``, memoise masking results inside a single extractor
        instance so the tracer and probe runner share work.
    """

    def __init__(
        self,
        top_k: int = 20,
        blackbox_budget: int | None = 16,
        blackbox_cache: bool = True,
    ) -> None:
        self.top_k = int(top_k)
        self.blackbox_budget = blackbox_budget
        self._cache: dict[str, float] = {} if blackbox_cache else {}
        self._cache_enabled = blackbox_cache
        self._sae: SparseAutoencoder | None = None
        self._sae_layer: int = -1
        # Provider used during the last extract() call — needed for black-box masking.
        self._last_provider: BaseProvider | None = None

    # ------------------------------------------------------------------
    # SAE attachment
    # ------------------------------------------------------------------

    def attach_sae(self, sae: SparseAutoencoder, layer: int) -> None:
        self._sae = sae
        self._sae_layer = int(layer)

    @property
    def sae(self) -> SparseAutoencoder | None:
        return self._sae

    @property
    def sae_layer(self) -> int:
        return self._sae_layer

    # ------------------------------------------------------------------
    # Main dispatch
    # ------------------------------------------------------------------

    def extract(
        self,
        output: ProviderOutput,
        provider: BaseProvider | None = None,
    ) -> list[Feature]:
        """Top-level extraction dispatch.

        Parameters
        ----------
        output:
            The provider's :class:`ProviderOutput` for the prompt.
        provider:
            The provider instance — required for black-box masking (so we
            can issue masked forward passes).  When ``None`` and the
            output is black-box, we fall back to a single-pass importance
            heuristic that is clearly labelled as such.
        """
        self._last_provider = provider
        if output.has_internals:
            return self._whitebox_features(output)
        return self._blackbox_features(output, provider)

    # ------------------------------------------------------------------
    # White-box: SAE-based features when available, else L2 norm scoring.
    # ------------------------------------------------------------------

    def _whitebox_features(self, output: ProviderOutput) -> list[Feature]:
        activations = output.activations
        assert activations is not None
        n_layers, n_tokens, _ = activations.shape

        if self._sae is not None and 0 <= self._sae_layer < n_layers:
            return self._sae_features(output, activations)

        features: list[Feature] = []
        feat_id = 0
        for layer in range(n_layers):
            for tok_idx in range(n_tokens):
                act = activations[layer, tok_idx]
                score = float(np.linalg.norm(act))
                features.append(
                    Feature(
                        id=feat_id,
                        label=_layer_band_label(output.tokens, tok_idx, layer, n_layers),
                        layer=layer,
                        score=score,
                        token_idx=tok_idx,
                        node_type="feature",
                        evidence_kind="white_box",
                        meta={"method": "l2_norm"},
                    )
                )
                feat_id += 1
        features.sort(key=lambda f: f.score, reverse=True)
        return features[: self.top_k]

    def _sae_features(self, output: ProviderOutput, activations: np.ndarray) -> list[Feature]:
        assert self._sae is not None
        layer = self._sae_layer
        n_tokens = activations.shape[1]
        features: list[Feature] = []

        # Encode every token at the target layer in a single batched call.
        codes = self._sae.encode(activations[layer])  # (n_tokens, dict_size)
        local_top = min(self.top_k, codes.shape[1])
        for tok_idx in range(n_tokens):
            row = codes[tok_idx]
            if not np.any(row != 0):
                continue
            top_ids = np.argpartition(-row, local_top - 1)[:local_top]
            top_ids = top_ids[np.argsort(-row[top_ids])]
            for fid in top_ids:
                s = float(row[fid])
                if s <= 0.0:
                    continue
                label = self._sae.labels.get(int(fid), f"feature_{int(fid)}")
                features.append(
                    Feature(
                        id=int(fid),
                        label=label,
                        layer=layer,
                        score=s,
                        token_idx=tok_idx,
                        node_type="feature",
                        evidence_kind="white_box",
                        meta={"method": "sae"},
                    )
                )
        features.sort(key=lambda f: f.score, reverse=True)
        return features[: self.top_k]

    # ------------------------------------------------------------------
    # Black-box: real token-masking importance.
    # ------------------------------------------------------------------

    def _blackbox_features(
        self, output: ProviderOutput, provider: BaseProvider | None
    ) -> list[Feature]:
        tokens = output.tokens
        if not tokens:
            return []

        # If we don't have a provider handle, we cannot actually mask.  Return
        # one feature per token with a *clearly approximated* heuristic score
        # taken from the surface position so downstream code degrades gracefully.
        if provider is None:
            heuristic_score = output.output_prob
            features = [
                Feature(
                    id=i,
                    label=f"token:{t}",
                    layer=0,
                    score=float(heuristic_score / (1.0 + i)),
                    token_idx=i,
                    node_type="input_token",
                    evidence_kind="black_box",
                    meta={"method": "position_heuristic", "approximation": True},
                )
                for i, t in enumerate(tokens)
            ]
            features.sort(key=lambda f: f.score, reverse=True)
            return features[: self.top_k]

        importance = self.compute_token_importance(
            provider=provider, prompt=output.prompt, baseline=output
        )
        # importance is [(token, score)] — convert to Feature list.
        features = [
            Feature(
                id=i,
                label=f"token:{tok}",
                layer=0,
                score=float(score),
                token_idx=i,
                node_type="input_token",
                evidence_kind="black_box",
                meta={"method": "token_masking"},
            )
            for i, (tok, score) in enumerate(importance)
        ]
        features.sort(key=lambda f: f.score, reverse=True)
        return features[: self.top_k]

    # ------------------------------------------------------------------
    # Black-box: real masking + pairwise interactions.
    # ------------------------------------------------------------------

    def compute_token_importance(
        self,
        provider: BaseProvider,
        prompt: str,
        baseline: ProviderOutput | None = None,
    ) -> list[tuple[str, float]]:
        """Compute per-token causal importance via masking.

        ``score_i = prob_baseline(top) − prob_masked_i(top)`` with
        ``top`` fixed to the baseline's argmax.  Positive ⇒ token was
        load-bearing; negative ⇒ token was actively suppressing that prediction.
        """
        tokens = whitespace_tokens(prompt)
        if not tokens:
            return []

        if baseline is None:
            baseline = provider.run(prompt)
        target_token = baseline.output_token
        baseline_prob = baseline.output_prob

        limit = min(self.blackbox_budget or len(tokens), len(tokens))
        scores: list[tuple[str, float]] = []
        for i in range(limit):
            masked = mask_positions(tokens, [i])
            masked_prompt = token_join(masked)
            masked_prob = self._masked_prob(provider, masked_prompt, target_token)
            scores.append((tokens[i], float(baseline_prob - masked_prob)))
        for i in range(limit, len(tokens)):
            scores.append((tokens[i], 0.0))
        return scores

    def compute_pairwise_interactions(
        self,
        provider: BaseProvider,
        prompt: str,
        budget: int = 8,
    ) -> dict[tuple[int, int], float]:
        """Pairwise interaction scores for multi-hop circuit detection.

        ``interaction(i, j) = P_full(top) − P_mask_i(top) − P_mask_j(top) + P_mask_{i,j}(top)``

        Positive ⇒ tokens i and j are *jointly* required.
        """
        tokens = whitespace_tokens(prompt)
        n = min(len(tokens), int(budget))
        baseline = provider.run(prompt)
        target = baseline.output_token

        p_full = baseline.output_prob
        p_single = {
            i: self._masked_prob(provider, token_join(mask_positions(tokens, [i])), target)
            for i in range(n)
        }
        interactions: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                p_both = self._masked_prob(
                    provider, token_join(mask_positions(tokens, [i, j])), target
                )
                interactions[(i, j)] = float(p_full - p_single[i] - p_single[j] + p_both)
        return interactions

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _masked_prob(self, provider: BaseProvider, prompt: str, target_token: str) -> float:
        """Return P(target_token | masked_prompt) using *provider*.

        Looks up *target_token* in the masked run's ``top_tokens`` list; if
        the target isn't in the top-k we approximate its probability as 0.0
        (the model has effectively given up on that prediction at that mask).
        """
        if self._cache_enabled and prompt in self._cache:
            base_prob = self._cache[prompt]
            return base_prob if not target_token else base_prob

        out = provider.run(prompt)
        top = dict(out.top_tokens)
        prob = float(top.get(target_token, 0.0)) if target_token else out.output_prob
        if self._cache_enabled:
            self._cache[prompt] = prob
        return prob


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer_band_label(tokens: list[str], tok_idx: int, layer: int, n_layers: int) -> str:
    tok = tokens[tok_idx] if 0 <= tok_idx < len(tokens) else "?"
    if n_layers <= 1:
        band = "only"
    elif layer < n_layers // 3:
        band = "early"
    elif layer < 2 * n_layers // 3:
        band = "mid"
    else:
        band = "late"
    return f"{tok}@{band}"
