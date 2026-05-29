"""SupernodeGrouper — clusters features into supernodes by cosine similarity.

A supernode is a :class:`~llmscope.features.feature.FeatureSet` containing
features whose activation vectors are mutually similar — analogous to the
supernodes in Anthropic's attribution-graph visualisations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from llmscope.features.feature import Feature, FeatureSet

if TYPE_CHECKING:
    from llmscope.providers.base import ProviderOutput


class SupernodeGrouper:
    """Group features into higher-level supernodes using cosine similarity.

    Parameters
    ----------
    similarity_threshold:
        Features with pairwise cosine similarity above this threshold are
        merged into the same supernode.
    """

    def __init__(self, similarity_threshold: float = 0.8) -> None:
        self.similarity_threshold = similarity_threshold

    def group(
        self,
        features: list[Feature],
        output: "ProviderOutput",
    ) -> list[FeatureSet]:
        """Cluster *features* into :class:`~llmscope.features.feature.FeatureSet` supernodes.

        Parameters
        ----------
        features:
            List of :class:`~llmscope.features.feature.Feature` objects to cluster.
        output:
            :class:`~llmscope.providers.base.ProviderOutput` providing activations
            for similarity computation.

        Returns
        -------
        list[FeatureSet]
            Each item is a supernode containing one or more related features.
        """
        if output.activations is None or len(features) == 0:
            return self._group_by_label(features)
        return self._group_by_activation(features, output.activations)

    # ------------------------------------------------------------------
    # Activation-based clustering (white-box)
    # ------------------------------------------------------------------

    def _group_by_activation(
        self,
        features: list[Feature],
        activations: np.ndarray,
    ) -> list[FeatureSet]:
        """Greedy cosine-similarity clustering."""
        assigned = [False] * len(features)
        groups: list[FeatureSet] = []

        for i, feat_i in enumerate(features):
            if assigned[i]:
                continue
            fset = FeatureSet(name=feat_i.label)
            fset.add(feat_i)
            assigned[i] = True

            vec_i = activations[feat_i.layer, feat_i.token_idx]
            for j, feat_j in enumerate(features[i + 1 :], i + 1):
                if assigned[j]:
                    continue
                vec_j = activations[feat_j.layer, feat_j.token_idx]
                sim = _cosine_sim(vec_i, vec_j)
                if sim >= self.similarity_threshold:
                    fset.add(feat_j)
                    assigned[j] = True

            groups.append(fset)

        return groups

    # ------------------------------------------------------------------
    # Label-based grouping (black-box fallback)
    # ------------------------------------------------------------------

    def _group_by_label(self, features: list[Feature]) -> list[FeatureSet]:
        """Group by token prefix when activations are unavailable."""
        seen: dict[str, FeatureSet] = {}
        for feat in features:
            key = feat.label.split("@")[0] if "@" in feat.label else feat.label
            if key not in seen:
                seen[key] = FeatureSet(name=key)
            seen[key].add(feat)
        return list(seen.values())


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
