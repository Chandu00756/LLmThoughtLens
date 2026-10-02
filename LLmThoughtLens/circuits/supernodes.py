"""SupernodeGrouper — cluster features into supernodes by activation similarity.

A supernode is a :class:`~LLmThoughtLens.features.feature.FeatureSet` of
features whose activation directions are mutually close (cosine).  When an
SAE is attached the grouper uses the SAE *decoder directions* directly,
which gives much sharper, label-aligned clusters than raw activations.

SAE features identify their dictionary entry through
``meta["sae_feature_id"]`` (``Feature.id`` is unique per position and SAE,
not the dictionary index).  With several SAEs pass ``sae`` as a mapping
``{attachment name: sae}`` (``FeatureExtractor.sae_map``); features are
matched through ``meta["sae_name"]``.  A single SAE paired with features from
several SAEs is ambiguous, so the grouper then clusters by activation instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

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
        sae: SparseAutoencoder | Mapping[str, SparseAutoencoder] | None = None,
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
        if self.sae is not None and (isinstance(self.sae, Mapping) or _single_source(features)):
            return self._group_by_sae_direction(features, output.activations)
        if output.activations is not None:
            return self._group_by_activation(features, output.activations)
        return self._group_by_label(features)

    # ------------------------------------------------------------------
    # SAE-direction clustering (sharpest grouping)
    # ------------------------------------------------------------------

    def _group_by_sae_direction(
        self, features: list[Feature], activations: np.ndarray | None = None
    ) -> list[FeatureSet]:
        assert self.sae is not None
        directions = [self._direction(f, activations) for f in features]
        index = {id(f): i for i, f in enumerate(features)}
        return self._greedy_cluster(features, lambda f: directions[index[id(f)]])

    def _direction(self, f: Feature, activations: np.ndarray | None) -> Any:
        """Decoder direction of *f*, else its activation vector (else zeros)."""
        fid = f.meta.get("sae_feature_id")
        sae: Any = self.sae
        if isinstance(sae, Mapping):
            sae = sae.get(f.meta.get("sae_name")) if fid is not None else None
        if sae is not None:
            return sae.feature_direction(int(fid) if fid is not None else f.id)
        if activations is not None and 0 <= f.layer < activations.shape[0]:
            return activations[f.layer, f.token_idx]
        return np.zeros(1)

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


def _single_source(features: list[Feature]) -> bool:
    """True when every SAE feature comes from the same attached SAE (or none carry a name)."""
    names = {f.meta.get("sae_name") for f in features if f.meta.get("sae_feature_id") is not None}
    return len(names) <= 1
