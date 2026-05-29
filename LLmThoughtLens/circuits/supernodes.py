"""SupernodeGrouper — cluster features into supernodes by activation similarity.

A supernode is a :class:`~LLmThoughtLens.features.feature.FeatureSet` of
features whose activation directions are mutually close (cosine).  When an
SAE is attached the grouper uses the SAE *decoder directions* directly,
which gives much sharper, label-aligned clusters than raw activations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from LLmThoughtLens.features.feature import Feature, FeatureSet
from LLmThoughtLens.utils.math_utils import cosine_sim

if TYPE_CHECKING:
    from LLmThoughtLens.features.sae import SparseAutoencoder
    from LLmThoughtLens.providers.base import ProviderOutput


class SupernodeGrouper:
    """Greedy cosine-similarity clusterer."""

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        sae: SparseAutoencoder | None = None,
    ) -> None:
        self.similarity_threshold = float(similarity_threshold)
        self.sae = sae

    def group(
        self,
        features: list[Feature],
        output: ProviderOutput,
    ) -> list[FeatureSet]:
        if not features:
            return []
        if self.sae is not None:
            return self._group_by_sae_direction(features)
        if output.activations is not None:
            return self._group_by_activation(features, output.activations)
        return self._group_by_label(features)

    # ------------------------------------------------------------------
    # SAE-direction clustering (sharpest grouping)
    # ------------------------------------------------------------------

    def _group_by_sae_direction(self, features: list[Feature]) -> list[FeatureSet]:
        assert self.sae is not None
        directions = {f.id: self.sae.feature_direction(f.id) for f in features}
        return self._greedy_cluster(features, lambda f: directions[f.id])

    # ------------------------------------------------------------------
    # Activation clustering (white-box without SAE)
    # ------------------------------------------------------------------

    def _group_by_activation(
        self, features: list[Feature], activations: np.ndarray
    ) -> list[FeatureSet]:
        return self._greedy_cluster(features, lambda f: activations[f.layer, f.token_idx])

    # ------------------------------------------------------------------
    # Label clustering (black-box fallback)
    # ------------------------------------------------------------------

    def _group_by_label(self, features: list[Feature]) -> list[FeatureSet]:
        seen: dict[str, FeatureSet] = {}
        for f in features:
            key = f.label.split("@")[0] if "@" in f.label else f.label
            if key not in seen:
                seen[key] = FeatureSet(name=key)
            seen[key].add(f)
        return list(seen.values())

    # ------------------------------------------------------------------
    # Greedy cluster shared by activation + SAE paths
    # ------------------------------------------------------------------

    def _greedy_cluster(self, features: list[Feature], vec_of) -> list[FeatureSet]:
        assigned = [False] * len(features)
        groups: list[FeatureSet] = []
        for i, fi in enumerate(features):
            if assigned[i]:
                continue
            fset = FeatureSet(name=fi.label or f"supernode_{i}")
            fset.add(fi)
            assigned[i] = True
            vi = np.asarray(vec_of(fi))
            for j in range(i + 1, len(features)):
                if assigned[j]:
                    continue
                vj = np.asarray(vec_of(features[j]))
                if cosine_sim(vi, vj) >= self.similarity_threshold:
                    fset.add(features[j])
                    assigned[j] = True
            fset.meta["representative_layer"] = fset.representative_layer()
            fset.meta["representative_token"] = fset.representative_token()
            groups.append(fset)
        return groups
